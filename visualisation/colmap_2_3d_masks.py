import cv2

import os
import os.path as osp
from io_functions.colmap import read_images_binary, read_cameras_binary
from io_functions.load_new_prediction_data import load_predictions
from reconstruction import load_reconstruction_parameters, load_depth, filter_depth
from reconstruction import image_wise_projection, instance_wise_projection
from utils_2d import seperate_masks




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
    stats_dir = 'C:/projects/01_resources/coop_photog/0412_stats'


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


        """if os.path.exists(output_file):
            print(f"already processed {image.name}")
            continue"""
        """if j>5:
            break"""
        if image.name not in ["20231108_112532_color.jpg", "20231108_112527_color.jpg", "20231108_112933_color.jpg",  "20231108_112925_color.jpg"]:
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

        mask_image = cv2.imread(paths["prediction_masks"].format(image.name), cv2.IMREAD_GRAYSCALE)
        masks = seperate_masks(mask_image)

        depth = filter_depth(depth, masks, label_info[image.name], image.name, os.path.join(stats_dir, "depth_histo"))

        instance_wise_projection(output_dir_bbox, point_clouds_path, cam_matrix, depth, color_image, label_info[image.name], masks, rotation_matrix, t, image.name)





if __name__ == '__main__':
    centerpoints_2d_to_3d()