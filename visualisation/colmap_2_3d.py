import json
import os
import os.path as osp
from functools import partial
import cv2
import numpy as np
from PIL import Image
import open3d as o3d

from Colmap_test.python.read_write_model import read_images_binary, read_cameras_binary
from Colmap_test.python.read_write_dense import read_array

from io_functions.sgg import load_sgg_data, load_sgg_data_visual

from io_functions.plyfile import PlyDataReader

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

def unproject(k, depth_map, mask=None):
    if mask is None:
        # only consider points where we have a depth value
        mask = depth_map > 0
    # create xy coordinates from image position
    v, u = np.indices(depth_map.shape)
    v = v[mask]
    u = u[mask]
    depth = depth_map[mask].ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    points_3d_xyz = (np.linalg.inv(k[:3, :3]).dot(uv1_points.T) * depth).T
    return points_3d_xyz

def draw_point_cloud(points, colors=None, normals=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc

# ----------------------------------------------------------------------------- #
# Worker function
# ----------------------------------------------------------------------------- #

def find_nearest_pixel(img, target):
    #  nonzero[:,:,0]: X axis is horizontal axis (left to right)
    #  nonzero[:,:,1]: Y axis is vertical axis (top to bottom)
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    # [[x,y]]
    return nonzero[nearest_index]


def create_mask_for_depth(depth, boundingboxes):
    mask = np.zeros(depth.shape)
    for boundingbox in boundingboxes:
        x1 = int(boundingbox[0])
        y1 = int(boundingbox[1])
        x2 = int(boundingbox[2])
        y2 = int(boundingbox[3])
        centric_x = int((x1+x2)/2)
        centric_y = int((y1+y2)/2)
        point = find_nearest_pixel(depth, (centric_x, centric_y))
        mask[point[0][1], point[0][0]] = 1
        # this order is right
        # mask[y1:y2, x1:x2] = 1
    return mask

def project_depth_one_by_one(cam_matrix, depth, boundingbox, t, rotation_matrix):
    depth = depth.copy() # avoid overwriting
    mask = np.zeros(depth.shape)
    x1 = int(boundingbox[0])
    y1 = int(boundingbox[1])
    x2 = int(boundingbox[2])
    y2 = int(boundingbox[3])
    centric_x = int((x1+x2)/2)
    centric_y = int((y1+y2)/2)
    point = find_nearest_pixel(depth, (centric_x, centric_y))
    mask[point[0][1], point[0][0]] = 1
    depth_masked = depth * mask
    unproj_pts = unproject(cam_matrix, depth_masked)
    unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)
    return unproj_pts

def project_depth_in_bbox_one_by_one(cam_matrix, depth, boundingbox, t, rotation_matrix):
    depth = depth.copy() # avoid overwriting
    mask = np.zeros(depth.shape)
    x1 = int(boundingbox[0])
    y1 = int(boundingbox[1])
    x2 = int(boundingbox[2])
    y2 = int(boundingbox[3])
    mask[y1:y2, x1:x2] = 1
    depth_masked = depth * mask
    unproj_pts = unproject(cam_matrix, depth_masked)
    unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)
    return unproj_pts

def create_mask_for_color(depth, boundingboxes):
    mask = np.zeros(depth.shape)
    for boundingbox in boundingboxes:
        x1 = int(boundingbox[0])
        y1 = int(boundingbox[1])
        x2 = int(boundingbox[2])
        y2 = int(boundingbox[3])
        centric_x = int((x1+x2)/2)
        centric_y = int((y1+y2)/2)
        point = find_nearest_pixel(depth, (centric_x, centric_y))
        mask[point[0][1], point[0][0]] = 1
        # mask[y1:y2, x1:x2] = 1
    mask_3d = np.stack((mask,mask,mask),axis=2) #3 channel mask
    return mask_3d

def test():

    parameter_path_parameter = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/sparse'
    path_depth_maps = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/stereo/depth_maps'
    path_color_images = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/images'
    custom_prediction_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_sg_pred/custom_prediction.json'
    custom_data_info_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_sg_pred/custom_data_info.json'
    output_dir_center_points = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_center_point/'
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

    # # for testing
    # frame_ids = [1, 2, 3] 
    # for i, frame_id in enumerate(frame_ids):
    #     # load pose #(rotation matrix and translation vector)
    #     rotation_matrix = image_info_dic[frame_id].qvec2rotmat()
    #     t = image_info_dic[frame_id].tvec
    #     # load intrinsic
    #     fx = cameras_dic[image_info_dic[frame_id].camera_id].params[0]
    #     fy = cameras_dic[image_info_dic[frame_id].camera_id].params[1]
    #     cx = cameras_dic[image_info_dic[frame_id].camera_id].params[2]
    #     cy = cameras_dic[image_info_dic[frame_id].camera_id].params[3]

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
        
        # # load color image, just for debugging purpose
        # color_im = Image.open(paths['color'].format(image.name))
        # if resize: 
        #     color_im = color_im.resize(resize, Image.NEAREST)
        # color = np.asarray(color_im)
        # # create mask for color image
        # mask_3d = create_mask_for_color(depth, bboxes)
        # color_masked = color * mask_3d
        # # masked_img = cv2.findNonZero(color_masked)
        # color_im = Image.fromarray(color_masked.astype(np.uint8))
        # color_im.save('/home/dchangyu/Pano-SGG-PC/visualisation/data/' + str(image_name))

        # This is the original code for parallel processing of depth map, but we cant keep track of the box labels
        # ----------------------------------------------------------------------------- #
        # # create mask for depth map
        # mask = create_mask_for_depth(depth, bboxes)
        # # apply mask to depth
        # depth_masked = depth * mask
        # # un-project point cloud from depth map
        # unproj_pts = unproject(cam_matrix, depth_masked)
        # # apply pose to unprojected points
        # unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)
        # ----------------------------------------------------------------------------- #

        # Do the projection one by one instead of parallelism to keep track of the box labels
        unproj_pts = []
        for i, bbox in enumerate(bboxes):
            # project depth map to 3d points
            unproj_pts_one_by_one = project_depth_one_by_one(cam_matrix, depth, bbox, t, rotation_matrix)
            unproj_pts.append(unproj_pts_one_by_one)
        
        unproj_pts = np.asarray(unproj_pts)
        unproj_pts = unproj_pts.reshape(-1,3)

        # get box labels in index
        box_labels = prediction_info_dict[image_name]['box_labels']
        box_labels_index = [i for i in range(len(box_labels))] # [0,1,2,3,4,5,6,7]
        
        # TODO: try using io_functions.plyfile pcd2ply and writeply
        plyDataReader = PlyDataReader()
        plyDataReader.pcd2ply(unproj_pts, box_labels_index)
        if not os.path.exists(output_dir_center_points):
            os.makedirs(output_dir_center_points)
        plyDataReader.write_ply(output_dir_center_points + image_name + '.ply')
        
        # for i, label in enumerate(box_labels):
        #     print (f"{i} : {label}")
        print('finish writing ply file for image: ' + image_name)

def visualization():
    parameter_path_parameter = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway_vis'
    path_depth_maps = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway_vis'
    path_color_images = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway_vis'
    custom_prediction_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway_vis/custom_prediction_subset_2.json'
    custom_data_info_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway_vis/custom_data_info.json'
    output_dir_center_points = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway_vis/output/'
    # (w,h) should be the same size as images used in sg prediction
    resize = (1996, 1500)  
    # resize = None

    # load sg prediction
    prediction_info_dict = load_sgg_data_visual(custom_prediction_path, custom_data_info_path)


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

    # # for testing
    # frame_ids = [1, 2, 3] 
    # for i, frame_id in enumerate(frame_ids):
    #     # load pose #(rotation matrix and translation vector)
    #     rotation_matrix = image_info_dic[frame_id].qvec2rotmat()
    #     t = image_info_dic[frame_id].tvec
    #     # load intrinsic
    #     fx = cameras_dic[image_info_dic[frame_id].camera_id].params[0]
    #     fy = cameras_dic[image_info_dic[frame_id].camera_id].params[1]
    #     cx = cameras_dic[image_info_dic[frame_id].camera_id].params[2]
    #     cy = cameras_dic[image_info_dic[frame_id].camera_id].params[3]

    for i, image in image_info_dic.items():

        if image.name not in prediction_info_dict.keys():
            continue

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

        # Do the projection one by one instead of parallelism to keep track of the box labels
        unproj_center_pts = []
        unproj_bbox_pts = []
        for i, bbox in enumerate(bboxes):
            # project depth map to 3d points
            unproj_pts_one_by_one = project_depth_one_by_one(cam_matrix, depth, bbox, t, rotation_matrix)
            unproj_center_pts.append(unproj_pts_one_by_one)
            # project depth map in bbox to 3d points
            unproj_pts_bbox = project_depth_in_bbox_one_by_one(cam_matrix, depth, bbox, t, rotation_matrix)
            unproj_bbox_pts.append(unproj_pts_bbox)
        
        unproj_center_pts = np.asarray(unproj_center_pts)
        unproj_center_pts = unproj_center_pts.reshape(-1,3)
        
        # create scalar list for visalization
        scalar_list = [30 for i in range(unproj_bbox_pts[0].shape[0])] + [40 for i in range(unproj_bbox_pts[1].shape[0])]
        unproj_bbox_pts = np.concatenate(unproj_bbox_pts, axis=0)

        # get box labels in index
        box_labels = prediction_info_dict[image_name]['box_labels']
        box_labels_index = [i for i in range(len(box_labels))] # [0,1,2,3,4,5,6,7]
        
        plyDataReader = PlyDataReader()
        plyDataReader.pcd2ply(unproj_center_pts, [0, 20])
        if not os.path.exists(output_dir_center_points):
            os.makedirs(output_dir_center_points)
        plyDataReader.write_ply(output_dir_center_points + image_name + '.ply')

        plyDataReader_bbox = PlyDataReader()
        plyDataReader_bbox.pcd2ply(unproj_bbox_pts, scalar_list) # random number just for visualization
        if not os.path.exists(output_dir_center_points):
            os.makedirs(output_dir_center_points)
        plyDataReader_bbox.write_ply(output_dir_center_points + image_name + '_bbox_projection.ply')



        
        # for i, label in enumerate(box_labels):
        #     print (f"{i} : {label}")
        print('finish writing ply file for image: ' + image_name)

if __name__ == '__main__':
    visualization()
