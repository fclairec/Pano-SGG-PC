
import json
import os
import os.path as osp
from functools import partial
import pickle
import cv2
import numpy as np
from PIL import Image
import open3d as o3d
import json
import os
import os.path as osp
from io_functions.colmap import read_images_binary, read_cameras_binary
from io_functions.read_write_dense import read_array
from io_functions.load_new_prediction_data import load_predictions


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

def image_wise_projection(output_dir, cam_matrix, depth, color_image, rotation_matrix, t, image_name):
    point_array = unproject(cam_matrix, depth, t, rotation_matrix, color_image=color_image)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(point_array['points'])
    # Assign colors to the point cloud
    o3d_pcd.colors = o3d.utility.Vector3dVector(point_array['color'] / 255)
    write_and_show_pcd(output_dir, image_name, point_array)


def instance_wise_projection(output_dir_bbox, output_dir_pcd, cam_matrix, depth_map, color_image, label_info, masks, rotation_matrix, t, image_name):
    """ loops over all instances and projects them into 3d space"""
    for i, (mask, info) in enumerate(zip(masks, label_info)):
        print(f"processing instance {i}/{len(label_info)}: {info.label}")
        depth_masked = depth_map * mask
        point_array = unproject(cam_matrix, depth_masked, t, rotation_matrix, color_image=color_image)

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(point_array['points'])
        # Assign colors to the point cloud
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_array['color'] / 255)

        compute_and_write_hull(output_dir_bbox, image_name, o3d_pcd, info.value)
        write_and_show_pcd(output_dir_pcd, image_name, point_array, info.value)

    return


def write_and_show_pcd(point_clouds_path, image_name, point_array, inst_i=None):
    # point cloud to xyz point cloud file
    if inst_i is not None:
        xyz_path = osp.join(point_clouds_path, image_name.split("_color")[0] + f'_{inst_i}.xyz')
    else:
        xyz_path = osp.join(point_clouds_path, image_name.split("_color")[0] + '.xyz')
    # Manually format each point and its color into a string
    formatted_lines = [
        f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}"
        for point, color in zip(point_array['points'], point_array['color'])
    ]
    # Join the lines into a single string
    data_str = "\n".join(formatted_lines)

    # Write to the XYZ file
    with open(xyz_path, 'w') as file:
        file.write(data_str)




def compute_and_write_hull(output_dir_center_bbox, image_name, o3d_pcd, inst_i=None):
    # Compute the convex hull
    try:
        nb_neighbors = 10
        std_ratio = 2.0
        cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        inlier_cloud = o3d_pcd.select_by_index(ind)

        hull, _ = inlier_cloud.compute_convex_hull()
    except:
        print("no hull found")
        return
    # hull.paint_uniform_color([1, 0, 0, 0.5])  # Example: Red color with 50% transparency

    lines = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    lines.paint_uniform_color([1, 0, 0])

    # o3d.visualization.draw_geometries([o3d_pcd, lines])

    # Convert the hull to a mesh
    hull_mesh = o3d.geometry.TriangleMesh(hull.vertices, hull.triangles)

    if inst_i is not None:
        hull_obj_path = osp.join(output_dir_center_bbox, image_name.split("_color")[0] + f'_{inst_i}.obj')
        hull_mtl_path = os.path.join(output_dir_center_bbox, image_name.split("_color")[0] + f'_{inst_i}.mtl')
    else:
        hull_obj_path = osp.join(output_dir_center_bbox, image_name.split("_color")[0] + '.obj')
        hull_mtl_path = os.path.join(output_dir_center_bbox, image_name.split("_color")[0] + '.mtl')

    o3d.io.write_triangle_mesh(hull_obj_path, hull_mesh)

    mtl_content = (
        "newmtl TransparentMaterial\n"
        "Kd 0.4 0.4 0.4\n"
        "d 0.5  # Opacity: 0.5 means 50% transparent\n"
    )

    # Write the MTL content to the file
    with open(hull_mtl_path, 'w') as mtl_file:
        mtl_file.write(mtl_content)

    with open(hull_obj_path, 'r') as file:
        obj_content = file.readlines()

    # Insert the MTL file reference after the first line
    obj_content.insert(1, f'mtllib {os.path.basename(hull_mtl_path)}\n')
    # Insert usemtl TransparentMaterial before the first 'f' line
    for i, line in enumerate(obj_content):
        if line.startswith('f'):
            obj_content.insert(i, 'usemtl TransparentMaterial\n')
            break
    # Write the modified content back to the OBJ file
    with open(hull_obj_path, 'w') as file:
        file.writelines(obj_content)


def seperate_masks(mask_image):
    num_masks = np.max(mask_image)
    separated_masks = []
    for i in range(1, num_masks + 1):  # Starting from 1 to exclude the background
        mask = np.where(mask_image == i, 1, 0).astype(np.uint8)
        separated_masks.append(mask)
    return separated_masks


def centerpoints_2d_to_3d():
    parameter_path_parameter = 'C:/projects/01_resources/coop_photog/info'
    path_depth_maps = 'C:/projects/01_resources/coop_photog/depth_maps'
    path_color_images = 'C:/projects/01_resources/coop_photog/images'
    path_depth_sensor = 'C:/projects/01_resources/coop_photog/depth_sensor'
    custom_prediction_path = '...'
    custom_data_info_path = '...'
    point_clouds_path = 'C:/projects/01_resources/coop_photog/0412_sensor_point_clouds'
    alignment_checks = 'C:/projects/01_resources/coop_photog/0412_alignment_checks'
    output_dir_center_points = 'C:/projects/01_resources/coop_photog/center_points'
    output_dir_bbox = 'C:/projects/01_resources/coop_photog/0412_box_sensor_point_clouds'
    prediction_masks = 'C:/projects/01_resources/coop_photog/preds_seminar2'
    #image_info_dic_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/img_info_dic_threshold_200.pkl'

    paths = {'color': osp.join(path_color_images, '{}'),
             'depth': osp.join(path_depth_maps, '{}.geometric.bin'),
             'pose': osp.join(parameter_path_parameter, 'images.bin'),
             'camera_intrinsics': osp.join(parameter_path_parameter, 'cameras.bin'),
             'depth_sensor': osp.join(path_depth_sensor, '{}_depth.png'),
             'prediction_masks': osp.join(prediction_masks, '{}_mask.png'),
             'label_info': osp.join(prediction_masks, '{}_label.json')
             }

    # "Camera", ["id", "model", "width", "height", "params"])
    cameras_dic = read_cameras_binary(paths['camera_intrinsics'])
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    image_info_dic = read_images_binary(paths['pose'])

    image_names = [image.name for image in image_info_dic.values()]
    # showcase 112527 # 112933 112925
    #image_names = ["20231108_112532_color.jpg", "20231108_112527_color.jpg", "20231108_112933_color.jpg",  "20231108_112925_color.jpg"]
    label_info = load_predictions(paths["label_info"], image_names)

    for j, (image_idx, image) in enumerate(image_info_dic.items()):
        print("processing image: ", image.name)

        """if j>5:
            break"""
        if image.name not in ["20231108_112532_color.jpg", "20231108_112527_color.jpg", "20231108_112933_color.jpg",  "20231108_112925_color.jpg"]:
            continue

        # load color image
        color_image = cv2.imread(paths['color'].format(image.name))

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

        mode = 'sensor'

        # load depth map (H,W,C)
        if mode=='colmap':
            depth= read_array(paths['depth'].format(image.name))
            # numpy -> Image
            depth_im = Image.fromarray(depth)  # (w,h)?

            depth = np.asarray(depth_im, dtype=np.float32)
        elif mode=='sensor':
            depth_raw = o3d.io.read_image(paths['depth_sensor'].format(image.name.split("_color")[0]))
            # scaling factor
            s = 0.453
            depth = np.asarray(depth_raw).astype(np.float32) / 1000 * s
        else:
            raise ValueError('no valid depth input mode')

        # get image name
        image_name = image.name

        image_wise_projection(alignment_checks, cam_matrix, depth, color_image, rotation_matrix, t, image_name)

        mask_image = cv2.imread(paths["prediction_masks"].format(image.name), cv2.IMREAD_GRAYSCALE)
        masks = seperate_masks(mask_image)
        instance_wise_projection(output_dir_bbox, point_clouds_path, cam_matrix, depth, color_image, label_info[image_name], masks, rotation_matrix, t, image_name)


        """o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(point_array['points'])
        # Assign colors to the point cloud
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_array['color'] / 255)


        compute_and_write_hull(output_dir_center_bbox, image_name, o3d_pcd)
        write_and_show_pcd(point_clouds_path, image_name, point_array)"""




if __name__ == '__main__':
    centerpoints_2d_to_3d()