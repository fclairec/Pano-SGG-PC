import cv2

import os
import os.path as osp
from io_functions.colmap import read_images_binary, read_cameras_binary
from io_functions.load_new_prediction_data import load_predictions
from reconstruction import load_reconstruction_parameters, load_depth, filter_depth
from reconstruction import image_wise_projection, instance_wise_projection
from utils_2d import seperate_masks
from plotting import recolor_masks
from pathlib import Path

def generate_output_dirs(output_root_dir, point_clouds_path, alignment_checks, output_dir_bbox, projected_mask_2d, stats_dir):
    # create output directories
    if not os.path.exists(output_root_dir):
        os.mkdir(output_root_dir)
    if not os.path.exists(point_clouds_path):
        os.mkdir(point_clouds_path)
    if not os.path.exists(alignment_checks):
        os.mkdir(alignment_checks)
    if not os.path.exists(output_dir_bbox):
        os.mkdir(output_dir_bbox)
    if not os.path.exists(projected_mask_2d):
        os.mkdir(projected_mask_2d)
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)


def centerpoints_2d_to_3d():
    root_dir = Path('C:/projects/01_resources/coop_photog')
    parameter_path_parameter = root_dir / 'info'
    path_depth_maps = root_dir /'depth_maps'
    path_color_images = root_dir /'images'
    path_depth_sensor = root_dir / 'depth_sensor'
    path_predictions = 'C:/projects/01_resources/coop_photog/predictions'
    output_root_dir = root_dir / 'output'
    point_clouds_path = output_root_dir / 'point_clouds'
    alignment_checks = output_root_dir / 'alignment_checks'
    output_dir_bbox = output_root_dir / 'convex_hulls'
    projected_mask_2d = output_root_dir / 'mask_2d'
    stats_dir = output_root_dir / 'stats'

    paths = {'color': osp.join(path_color_images, '{}'),
             'depth': osp.join(path_depth_maps, '{}.geometric.bin'),
             'pose': osp.join(parameter_path_parameter, 'images.bin'),
             'camera_intrinsics': osp.join(parameter_path_parameter, 'cameras.bin'),
             'depth_sensor': osp.join(path_depth_sensor, '{}_depth.png'),
             'path_predictions': osp.join(path_predictions, '{}_mask.png'),
             'projected_mask_2d': osp.join(projected_mask_2d, '{}_projected_masks.png'),
             'label_info': osp.join(path_predictions, '{}_label.json')
             }

    generate_output_dirs(output_root_dir, point_clouds_path, alignment_checks, output_dir_bbox, projected_mask_2d, stats_dir)

    # "Camera", ["id", "model", "width", "height", "params"])
    cameras_dic = read_cameras_binary(paths['camera_intrinsics'])
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    image_info_dic = read_images_binary(paths['pose'])

    image_names = [image.name for image in image_info_dic.values()]
    # showcase 112527 # 112933 112925
    #image_names = ["20231108_112532_color.jpg", "20231108_112527_color.jpg", "20231108_112933_color.jpg",  "20231108_112925_color.jpg"]
    label_info = load_predictions(paths["label_info"], image_names)

    for j, (image_idx, image) in enumerate(image_info_dic.items()):


        """if os.path.exists(output_file):
            print(f"already processed {image.name}")
            continue"""
        """if j>5:
            break"""

        """["20231108_112532_color.jpg", "20231108_112527_color.jpg", "20231108_112933_color.jpg",
         "20231108_112925_color.jpg", "20231108_112534_color.jpg", "20231108_112533_color.jpg",
         "20231108_112526_color.jpg", "20231108_112912_color.jpg", "20231108_112919_color.jpg",
         "20231108_112921_color.jpg", "20231108_112930_color.jpg"]:"""

        # ["20231108_112919_color.jpg"]
        if image.name not in ["20231108_112532_color.jpg", "20231108_112527_color.jpg", "20231108_112933_color.jpg",
         "20231108_112925_color.jpg", "20231108_112534_color.jpg", "20231108_112533_color.jpg",
         "20231108_112526_color.jpg", "20231108_112912_color.jpg", "20231108_112919_color.jpg",
         "20231108_112921_color.jpg", "20231108_112930_color.jpg"]:

            continue
        print("processing image: ", image.name)
        # load color image
        if image.name == "20231108_113107_color.jpg":
            # this image causes an error somehow
            continue

        color_image = cv2.imread(paths['color'].format(image.name))

        rotation_matrix, t, cam_matrix = load_reconstruction_parameters(image, cameras_dic[image.camera_id])

        use_rgbd = True
        depth_path = paths['depth_sensor'] if use_rgbd else paths['colmap']
        depth = load_depth(use_rgbd, depth_path, image.name)



        # RECONSTRUCTION CHECK: this outputs the image_wise scaled point clouds, load them all into cloud compare to check the alignment. Comment out if you are sure the reconstruction is fine
        output_file = os.path.join(alignment_checks, image.name.split("_color")[0] + '.xyz')
        #image_wise_projection(alignment_checks, cam_matrix, depth, color_image, rotation_matrix, t, image_name, downsample=0.05*s)

        mask_image = cv2.imread(paths["path_predictions"].format(image.name), cv2.IMREAD_GRAYSCALE)
        masks = seperate_masks(mask_image)

        s = 0.453
        depth, legend_colors = filter_depth(depth, masks, label_info[image.name], image.name, stats_dir, s)

        instance_wise_projection(output_dir_bbox, point_clouds_path, cam_matrix, depth, color_image, label_info[image.name], masks, rotation_matrix, t, image.name, legend_colors, downsample=0.01*s)

        # recolor masks on images
        recolor_masks(masks, label_info[image.name], legend_colors, paths["projected_mask_2d"].format(image.name), color_image)






if __name__ == '__main__':
    centerpoints_2d_to_3d()