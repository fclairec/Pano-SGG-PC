# @inproceedings{jaritz2019multi,
# 	title={Multi-view PointNet for 3D Scene Understanding},
# 	author={Jaritz, Maximilian and Gu, Jiayuan and Su, Hao},
# 	booktitle={ICCV Workshop 2019},
# 	year={2019}
# }

import json
import os
import os.path as osp
import pickle
import time
import argparse
from functools import partial
import glob
import cv2

import numpy as np


from PIL import Image
import open3d as o3d

from Colmap_test.python.read_write_model import read_images_binary, read_cameras_binary
from Colmap_test.python.read_write_dense import read_array


Colmap_root = '/home/dchangyu/MV-KPConv/colmap/'

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def get_data_paths(colmap_root):
    return {
        'color': osp.join(colmap_root, 'color', '{}.jpg'),
        'depth': osp.join(colmap_root, 'depth', '{}.png'),
        'pose': osp.join(colmap_root, 'pose', '{}.txt'),
        'intrinsics_depth': osp.join(colmap_root, 'intrinsic', 'intrinsic_depth.txt'),
    }


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
    pc.points = o3d.utility.Vector3dVector(points)
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


#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#

def compute_rgbd_knn_colmap(cloud_name, cameras_dic,  image_info_dic, paths, subsampeled_scene_pts,
                     num_base_pts=2000, resize = (380, 200), debug =False):
    # choose m base points
    base_point_ind = np.random.choice(len(subsampeled_scene_pts), num_base_pts, replace=False)
    base_pts = subsampeled_scene_pts[base_point_ind]

    # initialize output
    # frame_ids is key list, start from 15, end at 183, but not continuous
    frame_ids = [key for key in image_info_dic.keys()]
    max_key = max(frame_ids) # 183
    overlaps = np.zeros([len(base_point_ind), max_key+1], dtype=bool)

    # build kd tree for base points
    base_pts_pc = o3d.geometry.PointCloud()
    base_pts_pc.points = o3d.utility.Vector3dVector(base_pts)
    pcd_tree = o3d.geometry.KDTreeFlann(base_pts_pc)

    # "Camera", ["id", "model", "width", "height", "params"])
    # cameras_dic = read_cameras_binary(paths['camera_intrinsics'])
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    # image_info_dic = read_images_binary(paths['pose'])

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
        depth = read_array(paths['depth'].format(image.name))

        # # resize = (160, 120) # w,h
        # resize = (380, 200)  # w,h # we dont need to resize to 80 60 because the depth map from colmap is already very sparse
        # # resize = (80, 60)  # w,h  # for overlap compute we use 80*60 depth map to speed up

        if resize:
            # original size around w=1920, h= 1080 but not stable, differ from images to images
            # Note that we may use 1900x1000 depth maps; however, camera matrix here is irrelevant to that.
            # adjust intrinsic matrix
            depth_map_size = (depth.shape[1], depth.shape[0]) # (w,h)
            cam_matrix = cam_matrix.copy()  # avoid overwriting
            cam_matrix[0] /= depth_map_size[0] / resize[0]
            cam_matrix[1] /= depth_map_size[1] / resize[1]
        # numpy -> Image
        depth_im = Image.fromarray(depth) #(w,h)?
        if resize: #resize to (380,200)
            depth_im = depth_im.resize(resize, Image.NEAREST)
        # depth = np.asarray(depth_im, dtype=np.float32) / 1000.
        depth = np.asarray(depth_im, dtype=np.float32)

        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth)

        # apply pose to unprojected points
        unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)

        # aligns unproject points with scan point cloud
        matrix = np.loadtxt(paths['translation_matrix_for_images'], dtype=np.float32)
        ones = np.ones(shape=unproj_pts.shape[0])
        unproj_pts = np.append(unproj_pts.T, [ones], axis=0)
        unproj_pts = np.matmul(unproj_pts.T, matrix.T)
        unproj_pts = np.delete(unproj_pts,3, axis=1)

        # rot_matrix = matrix[:3,:3]
        # t_matrix = matrix[:3,3]
        # unproj_pts = np.matmul(unproj_pts, rot_matrix) - t_matrix

        # extrinsic matrix
        # x = np.column_stack((rotation_matrix,t))
        # row = np.array([0,0,0,1])
        # x = np.append(x,[row],axis=0)
        #
        # ones = np.ones(shape=unproj_pts.shape[0])
        # unproj_pts = np.append(unproj_pts.T, [ones], axis=0)
        #
        # unproj_pts = np.matmul(unproj_pts.T, x.T)
        #
        # unproj_pts = np.delete(unproj_pts,3, axis=1)


        # for each point of unprojected point cloud find nearest neighbor (only one!) in whole scene point cloud(whole scene base points)
        for j in range(len(unproj_pts)):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(unproj_pts[j, :3], 0.1, 1)
            if found:
                # i is key of image_dic
                overlaps[idx_point, i] = True

        # visualize
        if debug:
            pts_vis = draw_point_cloud(subsampeled_scene_pts)
            base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
            overlap_base_pts_vis = draw_point_cloud(base_pts[overlaps[:, i]], colors=[0., 1., 0.])
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            # o3d.visualization.draw_geometries([overlap_base_pts_vis, unproj_pts_vis, base_pts_vis, pts_vis])
            # o3d.visualization.draw_geometries([base_pts_vis, unproj_pts_vis, overlap_base_pts_vis])
            path = '/home/dchangyu/MV-KPConv/'
            o3d.io.write_point_cloud(path + cloud_name + 'base_points' + '.ply', base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'overlap_base_pts_vis' + '.ply', overlap_base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'unproj_pts_vis' + '.ply', unproj_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'subsampeled_scene_pts' + '.ply', pts_vis)

    return base_point_ind, overlaps, frame_ids



# ----------------------------------------------------------------------------- #
# Worker function
# ----------------------------------------------------------------------------- #


def find_nearest_pixel(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

def load_sgg_data(box_topk=8, rel_topk=10):
    '''
    # parameters
    box_topk = 8 # select top k bounding boxes
    rel_topk = 10 # select top k relationships
    '''
    # load the following to files from DETECTED_SGG_DIR
    custom_prediction = json.load(open('/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_sg_pred/custom_prediction.json'))
    custom_data_info = json.load(open('/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_sg_pred/custom_data_info.json'))
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']
    prediction_info_dict = {}
    for image_idx in range(len(custom_data_info['idx_to_files'])):

        img_name = os.path.basename(custom_data_info['idx_to_files'][image_idx])
        boxes = custom_prediction[str(image_idx)]['bbox'][:box_topk]
        box_labels = custom_prediction[str(image_idx)]['bbox_labels'][:box_topk]
        box_scores = custom_prediction[str(image_idx)]['bbox_scores'][:box_topk]
        all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
        all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
        all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']

        for i in range(len(box_labels)):
            box_labels[i] = ind_to_classes[box_labels[i]]

        rel_labels = []
        rel_scores = []
        for i in range(len(all_rel_pairs)):
            if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
                rel_scores.append(all_rel_scores[i])
                label = str(all_rel_pairs[i][0]) + '_' + box_labels[all_rel_pairs[i][0]] + ' => ' + ind_to_predicates[all_rel_labels[i]] + ' => ' + str(all_rel_pairs[i][1]) + '_' + box_labels[all_rel_pairs[i][1]]
                rel_labels.append(label)

        rel_labels = rel_labels[:rel_topk]
        rel_scores = rel_scores[:rel_topk]

        prediction_info_dict[img_name] = {'boxes': boxes, 'box_labels': box_labels, 'box_scores': box_scores, 'rel_labels': rel_labels, 'rel_scores': rel_scores, 'image_idx': image_idx}
    
    return prediction_info_dict



def create_mask_for_depth(depth, boundingboxes):
    mask = np.zeros(depth.shape)
    for boundingbox in boundingboxes:
        x1 = int(boundingbox[0])
        y1 = int(boundingbox[1])
        x2 = int(boundingbox[2])
        y2 = int(boundingbox[3])
        cenric_x = int((x1+x2)/2)
        centric_y = int((y1+y2)/2)
        point = find_nearest_pixel(depth, (centric_y,cenric_x))
        mask[point[0][0], point[0][1]] = 1
        # this order is right
        # mask[y1:y2, x1:x2] = 1
    return mask

def create_mask_for_color(depth, boundingboxes):
    mask = np.zeros(depth.shape)
    for boundingbox in boundingboxes:
        x1 = int(boundingbox[0])
        y1 = int(boundingbox[1])
        x2 = int(boundingbox[2])
        y2 = int(boundingbox[3])
        cenric_x = int((x1+x2)/2)
        centric_y = int((y1+y2)/2)
        point = find_nearest_pixel(depth, (centric_y,cenric_x))
        mask[point[0][0], point[0][1]] = 1
        # mask[y1:y2, x1:x2] = 1
    mask_3d = np.stack((mask,mask,mask),axis=2) #3 channel mask
    return mask_3d

def test():

    # load sg prediction
    prediction_info_dict = load_sgg_data(box_topk=8, rel_topk=10)

    path_parameter = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/sparse'
    path_depth_maps = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/stereo/depth_maps'
    path_color_images = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/images'


    paths = {'color': osp.join(path_color_images, '{}'),
             # 'depth': osp.join(path_depth_maps, '{}.photometric.bin'),
             'depth': osp.join(path_depth_maps, '{}.geometric.bin'),
             'pose': osp.join(path_parameter, 'images.bin'),
             'camera_intrinsics': osp.join(path_parameter, 'cameras.bin'),
             }

    # "Camera", ["id", "model", "width", "height", "params"])
    cameras_dic = read_cameras_binary(paths['camera_intrinsics'])
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    image_info_dic = read_images_binary(paths['pose'])

    # frame_ids = [133, 134, 135]
    frame_ids = [1, 2, 3] # im_133, im_134, im_135

    # # choose m base points
    # num_base_pts = 4000
    # base_point_ind = np.random.choice(len(subsampeled_scene_pts), num_base_pts, replace=False)
    # base_pts = subsampeled_scene_pts[base_point_ind]

    # initialize output
    # overlaps = np.zeros([len(base_point_ind), len(frame_ids)], dtype=bool)

    # # build kd tree for base points
    # base_pts_pc = o3d.geometry.PointCloud()
    # base_pts_pc.points = o3d.utility.Vector3dVector(base_pts)
    # pcd_tree = o3d.geometry.KDTreeFlann(base_pts_pc)

    last_time = time.time()
    for i, frame_id in enumerate(frame_ids):

        # load pose #(rotation matrix and translation vector)
        rotation_matrix = image_info_dic[frame_id].qvec2rotmat()
        t = image_info_dic[frame_id].tvec

        # load intrinsic
        fx = cameras_dic[image_info_dic[frame_id].camera_id].params[0]
        fy = cameras_dic[image_info_dic[frame_id].camera_id].params[1]
        cx = cameras_dic[image_info_dic[frame_id].camera_id].params[2]
        cy = cameras_dic[image_info_dic[frame_id].camera_id].params[3]
        # build intrinsic matrix
        cam_matrix = np.zeros(shape=(3,3))
        cam_matrix[0] = [fx,0,cx]
        cam_matrix[1] = [0,fy,cy]
        cam_matrix[2] = [0,0,1]

        # load depth map (H,W,C)
        depth= read_array(paths['depth'].format(image_info_dic[frame_id].name))

        # resize = (160, 120) # w,h
        resize = (798, 600)  # w,h # we dont need to resize to 80 60 because the depth map from colmap is already very sparse
        # resize = (80, 60)  # w,h  # for overlap compute we use 80*60 depth map to speed up
        # resize = None

        if resize:
            # original size around w=1920, h= 1080 but not stable, differ from images to images
            # Note that we may use 1900x1000 depth maps; however, camera matrix here is irrelevant to that.
            # adjust intrinsic matrix
            depth_map_size = (depth.shape[1], depth.shape[0]) # (w,h)
            cam_matrix = cam_matrix.copy()  # avoid overwriting
            cam_matrix[0] /= depth_map_size[0] / resize[0]
            cam_matrix[1] /= depth_map_size[1] / resize[1]
        
        # numpy -> Image
        depth_im = Image.fromarray(depth) #(w,h)?
        if resize: #resize to (380,200)
            depth_im = depth_im.resize(resize, Image.NEAREST)
        
        # depth = np.asarray(depth_im, dtype=np.float32) / 1000.
        depth = np.asarray(depth_im, dtype=np.float32)

        # get image name
        image_name = image_info_dic[frame_id].name
        # find corresponding bounding boxes in sg prediction
        bboxes = prediction_info_dict[image_name]['boxes']

        
        # create mask for depth map
        mask = create_mask_for_depth(depth, bboxes)
        # apply mask to depth
        depth_masked = depth * mask


        # load color image
        # color_im = Image.open(paths['color'].format(image_info_dic[frame_id].name))
        # if resize: 
        #     color_im = color_im.resize(resize, Image.NEAREST)
        # color = np.asarray(color_im)
        # # create mask for color image
        # mask_3d = create_mask_for_color(depth, bboxes)
        # color_masked = color * mask_3d
        # # masked_img = cv2.findNonZero(color_masked)
        # color_im = Image.fromarray(color_masked.astype(np.uint8))
        # color_im.save('/home/dchangyu/Pano-SGG-PC/visualisation/data/' + str(image_name))

        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth_masked)

        # apply pose to unprojected points
        unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)

        # extrinsic matrix
        # x = np.column_stack((rotation_matrix,t))
        # row = np.array([0,0,0,1])
        # x = np.append(x,[row],axis=0)
        #
        # ones = np.ones(shape=unproj_pts.shape[0])
        # unproj_pts = np.append(unproj_pts.T, [ones], axis=0)
        #
        # unproj_pts = np.matmul(unproj_pts.T, x.T)
        #
        # unproj_pts = np.delete(unproj_pts,3, axis=1)

        # # for each point of unprojected point cloud find nearest neighbor (only one!) in whole scene point cloud(whole scene base points)
        # for j in range(len(unproj_pts)):
        #     # find a neighbor that is at most 1cm away
        #     found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(unproj_pts[j, :3], 0.1, 1)
        #     if found:
        #         overlaps[idx_point, i] = True

        # visualize
        debug = True
        if debug:
            # pts_vis = draw_point_cloud(subsampeled_scene_pts)
            # base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
            # overlap_base_pts_vis = draw_point_cloud(base_pts[overlaps[:, i]], colors=[0., 1., 0.])
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            # o3d.visualization.draw_geometries([overlap_base_pts_vis, unproj_pts_vis, base_pts_vis, pts_vis])
            # o3d.visualization.draw_geometries([base_pts_vis, unproj_pts_vis, overlap_base_pts_vis])
            path = '/home/dchangyu/Pano-SGG-PC/visualisation/data/'
            cloud_name = 'colmap_test_image'+ str(frame_id)
            # o3d.io.write_point_cloud(path + cloud_name + 'base_points' + '.ply', base_pts_vis)
            # o3d.io.write_point_cloud(path + cloud_name + 'overlap_base_pts_vis' + '.ply', overlap_base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'unproj_pts_vis' + '.ply', unproj_pts_vis)
            # o3d.io.write_point_cloud(path + cloud_name + 'subsampeled_scene_pts' + '.ply', pts_vis)

# from colmap, qvec stands for quaternion vector [w,x,y,z]
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    


if __name__ == '__main__':
    test()