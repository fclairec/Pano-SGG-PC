import numpy as np
import open3d as o3d
from io_functions.read_write_dense import read_array
from PIL import Image
import os.path as osp
from visualisation.utils_3d import compute_and_write_hull, write_and_show_pcd
from plotting import plot_depths
import cv2


def load_reconstruction_parameters(image, camera):
    # load pose #(rotation matrix and translation vector)
    rotation_matrix = image.qvec2rotmat()
    t = image.tvec

    # load intrinsic
    fx = camera.params[0]
    fy = camera.params[1]
    cx = camera.params[2]
    cy = camera.params[3]
    # build intrinsic matrix
    cam_matrix = np.zeros(shape=(3, 3))
    cam_matrix[0] = [fx, 0, cx]
    cam_matrix[1] = [0, fy, cy]
    cam_matrix[2] = [0, 0, 1]

    return rotation_matrix, t, cam_matrix


def load_depth(use_rgbd, deph_path, image_name):
    mode = 'sensor'
    # load depth map (H,W,C)
    if use_rgbd:
        depth_raw = o3d.io.read_image(deph_path.format(image_name.split("_color")[0]))
        # scaling factor
        s = 0.453
        depth = np.asarray(depth_raw).astype(np.float32) / 1000 * s
    else:
        "use colmap depth map"
        depth = read_array(deph_path.format(image_name))
        # numpy -> Image
        depth_im = Image.fromarray(depth)
        depth = np.asarray(depth_im, dtype=np.float32)

    return depth


def unproject(k, depth_map, t, rotation_matrix, mask=None, color_image=None):
    """ returns a structured array with 3d points and color values from a depth map and a color image
    :param k: intrinsic camera matrix
    :param depth_map: depth map
    :param mask: mask to apply to the depth map
    :param color_image: color image"""
    if mask is None:
        # only consider points where we have a depth value and the depth value is smaller than 1.5m
        mask = depth_map > 0
    # create xy coordinates from image position
    v, u = np.indices(depth_map.shape)
    v = v[mask]
    u = u[mask]
    depth = depth_map[mask].ravel()

    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    points_3d_xyz = (np.linalg.inv(k[:3, :3]).dot(uv1_points.T) * depth).T

    point_cloud_dtype = np.dtype([('points', np.float64, (3,)), ('color', np.uint8, (3,))])
    # Create an empty structured array
    structured_point_array = np.empty(len(points_3d_xyz), dtype=point_cloud_dtype)
    structured_point_array['points'] = points_3d_xyz
    structured_point_array["points"] = np.matmul(structured_point_array["points"] - t, rotation_matrix)

    if color_image is not None:
        color_image = color_image[mask]
        structured_point_array['color'] = color_image
        return structured_point_array
    return structured_point_array

def image_wise_projection(output_dir, cam_matrix, depth, color_image, rotation_matrix, t, image_name, downsample=0.05):
    point_array = unproject(cam_matrix, depth, t, rotation_matrix, color_image=color_image)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(point_array['points'])
    o3d_pcd.colors = o3d.utility.Vector3dVector(point_array['color'] / 255)
    if downsample:
        voxel_size = downsample
        o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Assign colors to the point cloud
    write_and_show_pcd(output_dir, image_name, o3d_pcd)


def instance_wise_projection(output_dir_bbox, output_dir_pcd, cam_matrix, depth_map, color_image, label_info, masks, rotation_matrix, t, image_name, legen_colors, downsample=False):
    """ loops over all instances and projects them into 3d space"""
    label_info = label_info[1:]
    for i, (mask, info, legend_color) in enumerate(zip(masks, label_info, legen_colors)):
        print(f"processing instance {info.value}/{len(label_info)}: {info.label}")
        depth_masked = depth_map * mask
        point_array = unproject(cam_matrix, depth_masked, t, rotation_matrix, color_image=color_image)
        if len(point_array) < 50:
            print("not enough points for instance")
            continue

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(point_array['points'])
        # Assign colors to the point cloud
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_array['color'] / 255)
        if downsample:
            voxel_size = downsample
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

        instance_name_str = str(info.label)+"_"+str(info.value)

        

        compute_and_write_hull(output_dir_bbox, image_name, o3d_pcd, legend_color,  instance_name_str)
        write_and_show_pcd(output_dir_pcd, image_name, o3d_pcd, instance_name_str)

    return

def statistical_filtering(masked_depth, threshold):
    # statistical outlier filtering

    mean_depth = np.mean(masked_depth[masked_depth > 0])
    std_depth = np.std(masked_depth[masked_depth > 0])

    # Filter out outliers
    lower_bound = mean_depth - threshold * std_depth
    upper_bound = mean_depth + threshold * std_depth
    filtered_depth = np.where((masked_depth > lower_bound) & (masked_depth < upper_bound), masked_depth, 0)
    return filtered_depth

def custom_semantic_filtering(masked_depth, label, s):
    # custom semantic-wise filtering
    if label.label == "ceiling" or label.label == "floor":
        max_distance = 100
        threshold_depth = max_distance * s
    elif label.label == "person":
        max_distance = 0
        threshold_depth = max_distance * s
    else:
        max_distance = 5
        threshold_depth = max_distance * s

    masked_depth[masked_depth > threshold_depth] = 0

    return masked_depth

def shrink_mask(mask):
    # shrink mask to remove outliers
    kernel = np.ones((5, 5), np.uint8)
    small_mask = cv2.erode(mask, kernel, iterations=1)
    return small_mask



def filter_depth(depth, masks, label_info, image_name, stats_dir):
    # TODO should contain the scaling, the distance threshhold and mask.
    # TODO for now only mask
    s = 0.453


    depth_masks = []

    for i, mask in enumerate(masks):

        small_mask = shrink_mask(mask)

        # Apply mask to depth
        masked_depth = depth * small_mask

        masked_depth = statistical_filtering(masked_depth, 2)

        masked_depth = custom_semantic_filtering(masked_depth, label_info[i + 1], s)

        depth_masks.append(masked_depth)

    legend_colors = plot_depths(depth_masks, label_info, image_name, stats_dir)
    # concatenate all depths maps into one depth map
    depth = np.sum(depth_masks, axis=0)

    return depth, legend_colors
