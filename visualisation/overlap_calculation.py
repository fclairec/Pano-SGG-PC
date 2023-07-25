import json
import os
import os.path as osp
from functools import partial
import pickle
import cv2
import numpy as np
from PIL import Image
import open3d as o3d

from Colmap_test.python.read_write_model import read_images_binary, read_cameras_binary
from Colmap_test.python.read_write_dense import read_array

from io_functions.sgg import load_sgg_data, load_sgg_data_visual
from io_functions.plyfile import PlyDataReader
from visualisation.colmap_2_3d import create_mask_for_depth, draw_point_cloud, unproject


def compute_overlaps():
    # load the photogrammetry point cloud
    plyDataReader= PlyDataReader()
    plyDataReader.read_ply('/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/fused.ply')
    points = plyDataReader.pos # (n, 3)
    # choose m base points
    base_point_ind = np.random.choice(points.shape[0], 6000, replace=False)
    base_pts = points[base_point_ind]

    # build kd tree for base points
    base_pts_pc = o3d.geometry.PointCloud()
    base_pts_pc.points = o3d.utility.Vector3dVector(base_pts)
    pcd_tree = o3d.geometry.KDTreeFlann(base_pts_pc)

    # load colmap and sgg data
    parameter_path_parameter = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/colmap/dense/sparse'
    path_depth_maps = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/colmap/dense/stereo/depth_maps'
    path_color_images = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/colmap/images'
    custom_prediction_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/custom_prediction.json'
    custom_data_info_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/custom_data_info.json'
    output_dir_center_points = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/'
    # (w,h) should be the same size as images used in sg prediction
    resize = (1996, 1500)  
    # resize = None
    # load sg prediction
    prediction_info_dict = load_sgg_data(8, 10, custom_prediction_path, custom_data_info_path)
    paths = {'color': osp.join(path_color_images, '{}'),
             # 'depth': osp.join(path_depth_maps, '{}.photometric.bin'),
             'depth': osp.join(path_depth_maps, '{}.geometric.bin'),
             'pose': osp.join(parameter_path_parameter, 'images.bin'),
             'camera_intrinsics': osp.join(parameter_path_parameter, 'cameras.bin'),
             }
    # "Camera", ["id", "model", "width", "height", "params"])
    cameras_dic = read_cameras_binary(paths['camera_intrinsics'])
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    image_info_dic = read_images_binary(paths['pose'])


    # initialize the overlap matrix
    frame_ids = [key for key in image_info_dic.keys()]
    max_key = max(frame_ids) # 168
    # overlaps = np.zeros([len(base_point_ind), max_key+1], dtype=bool)
    overlaps = np.zeros([max_key+1, len(base_point_ind)], dtype=bool)

    # project
    for i, image in image_info_dic.items():

        # load pose #(rotation matrix and translation vector)
        rotation_matrix = image.qvec2rotmat()
        t = image.tvec

        # load intrinsic
        fx = cameras_dic[image.camera_id].params[0]
        fy = cameras_dic[image.camera_id].params[1]
        cx = cameras_dic[image.camera_id].params[2]
        cy = cameras_dic[image.camera_id].params[3]
        # build intrinsic matrix
        cam_matrix = np.zeros(shape=(3,3))
        cam_matrix[0] = [fx,0,cx]
        cam_matrix[1] = [0,fy,cy]
        cam_matrix[2] = [0,0,1]

        # load depth map (H,W,C)
        depth= read_array(paths['depth'].format(image.name))

        if resize:
            # adjust intrinsic matrix
            depth_map_size = (depth.shape[1], depth.shape[0]) # (w,h)
            cam_matrix = cam_matrix.copy()  # avoid overwriting
            cam_matrix[0] /= depth_map_size[0] / resize[0]
            cam_matrix[1] /= depth_map_size[1] / resize[1]
        
        # numpy -> Image
        depth_im = Image.fromarray(depth) #(w,h)?
        if resize: 
            depth_im = depth_im.resize(resize, Image.NEAREST)
        
        # depth = np.asarray(depth_im, dtype=np.float32) / 1000.
        depth = np.asarray(depth_im, dtype=np.float32)

        # get image name
        image_name = image.name
        # find corresponding bounding boxes in sg prediction
        bboxes = prediction_info_dict[image_name]['boxes']

        # create mask for depth map
        mask = create_mask_for_depth(depth, bboxes)
        # apply mask to depth, only project bounding boxes
        depth_masked = depth * mask
        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth_masked)
        # apply pose to unprojected points
        unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)

        # for each point of unprojected point cloud find nearest neighbor (only one!) in base point cloud
        for j in range(len(unproj_pts)):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(unproj_pts[j, :3], 0.1, 1)
            if found:
                # i is key of image_dic
                # overlaps[idx_point, i] = True

                overlaps[i, idx_point] = True
        
        # visualize
        debug = False
        if debug:
            base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
            overlap_base_pts_vis = draw_point_cloud(base_pts[overlaps[:, i]], colors=[0., 1., 0.])
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            path = "/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/visualization/"
            if not os.path.exists(path):
                os.makedirs(path)
            o3d.io.write_point_cloud(path + image_name + 'base_points' + '.ply', base_pts_vis)
            o3d.io.write_point_cloud(path + image_name + 'overlap_base_pts_vis' + '.ply', overlap_base_pts_vis)
            o3d.io.write_point_cloud(path + image_name + 'unproj_pts_vis' + '.ply', unproj_pts_vis)
    
    def get_overlapping_mask_indices(mask_arrays, overlap_threshold):
        mask_arrays = [np.array(mask) for mask in mask_arrays]
        n = len(mask_arrays)
        
        # Create a overlap matrix
        overlap_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                overlap_matrix[i, j] = np.sum(np.logical_and(mask_arrays[i], mask_arrays[j]))
                overlap_matrix[j, i] = overlap_matrix[i, j]
        
        # Find the masks that have an overlap more than the threshold
        mask_to_remove = np.any(overlap_matrix > overlap_threshold, axis=0)
        
        # Get the indices of the masks that have an overlap more than the threshold
        mask_indices = [i for i, to_remove in enumerate(mask_to_remove) if to_remove]
        
        return mask_indices
    
    overlap_threshold = 1000
    filtered_mask_arrays = get_overlapping_mask_indices(overlaps, overlap_threshold)


    # remove images
    for i in filtered_mask_arrays:
        print('removing image: ', i)
        image_info_dic.pop(str(i), None)
    with open(f'saved_img_dictionary_threshold_{overlap_threshold}.pkl', 'wb') as f:
        pickle.dump(image_info_dic, f)
    return image_info_dic

if __name__ == '__main__':
    compute_overlaps()
    with open('saved_img_dictionary.pkl', 'rb') as f:
        x = pickle.load(f)
        print(x)
