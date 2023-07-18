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

from io_functions.sgg import load_sgg_data


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

    path_parameter = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/sparse'
    path_depth_maps = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/stereo/depth_maps'
    path_color_images = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_reconstruction/colmap/dense/images'
    custom_prediction_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_sg_pred/custom_prediction.json'
    custom_data_info_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_sg_pred/custom_data_info.json'

    # load sg prediction
    prediction_info_dict = load_sgg_data(box_topk=8, rel_topk=10, custom_prediction_path, custom_data_info_path)


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
        if resize: 
            depth_im = depth_im.resize(resize, Image.NEAREST)
        
        # depth = np.asarray(depth_im, dtype=np.float32) / 1000.
        depth = np.asarray(depth_im, dtype=np.float32)

        # get image name
        image_name = image.name
        # find corresponding bounding boxes in sg prediction
        bboxes = prediction_info_dict[image_name]['boxes']
        # get box labels
        box_labels = prediction_info_dict[image_name]['box_labels']
        
        # create mask for depth map
        mask = create_mask_for_depth(depth, bboxes)
        # apply mask to depth
        depth_masked = depth * mask


        # # load color image
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

        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth_masked)

        # apply pose to unprojected points
        unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)

        box_labels_index = range(len(box_labels))
        # add additional box label list index to point cloud
        unproj_pts = np.concatenate([unproj_pts, [[i] for i in box_labels_index]], axis=1)

        # visualize

        # TODO: try using io_functions.plyfile pcd2ply and writeply
        debug = True
        if debug:
            # can not store class label index with open3d...need to find a way to store it
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            path = '/home/dchangyu/Pano-SGG-PC/visualisation/data/'
            o3d.io.write_point_cloud(path + image_name  + '.ply', unproj_pts_vis)



if __name__ == '__main__':
    test()
