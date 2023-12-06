import os
import os.path as osp
from io_functions.load_new_prediction_data import load_predictions
import os
from collections import defaultdict
import open3d as o3d
import numpy as np
import json
import cv2
from visualisation.colmap_2_3d_masks import seperate_masks
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_bbox_intersections(all_bbox):
    iou = np.zeros((len(all_bbox), len(all_bbox)))
    intersecting_rects = []

    for i, bbox1 in enumerate(all_bbox):
        for j, bbox2 in enumerate(all_bbox):
            if i == j:
                continue
            else:
                x1 = max(bbox1[0], bbox2[0])
                y1 = max(bbox1[1], bbox2[1])
                x2 = min(bbox1[2], bbox2[2])
                y2 = min(bbox1[3], bbox2[3])
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                intersection = w * h
                union = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection
                iou[i, j] = intersection / union
                interArea = max(0, x2 - x1) * max(0, y2 - y1)


    intersecting_rect = np.where(iou > 0.5)





    return iou, intersecting_rect



def get_paths():
    # list all labels with their fequency
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
    prediction_masks = 'C:/projects/01_resources/coop_photog/predictions'
    stats_dir = 'C:/projects/01_resources/coop_photog/stats'
    # image_info_dic_path = '/mnt/c/Users/ge25yak/Desktop/SG_test_data/office_and_hallway/img_info_dic_threshold_200.pkl'

    paths = {'color': osp.join(path_color_images, '{}'),
             'depth': osp.join(path_depth_maps, '{}.geometric.bin'),
             'pose': osp.join(parameter_path_parameter, 'images.bin'),
             'camera_intrinsics': osp.join(parameter_path_parameter, 'cameras.bin'),
             'depth_sensor': osp.join(path_depth_sensor, '{}_depth.png'),
             'prediction_masks': osp.join(prediction_masks, '{}_mask.png'),
             'label_info': osp.join(prediction_masks, '{}_label.json')
             }

    return paths, stats_dir

def compute_statistics(json_data, aggregated_stats):
    for item in json_data['mask']:
        label = item['label']
        aggregated_stats[label]['count'] += 1

        if 'logit' in item:
            aggregated_stats[label]['total_logit'] += item['logit']

        if 'box' in item:
            box = item['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            aggregated_stats[label]['total_area'] += area

def compute_stats():
    paths, stats_dir = get_paths()

    color_image_names = os.listdir(paths['color'].format(''))

    label_infos = load_predictions(paths["label_info"], color_image_names)

    # Dictionary to aggregate statistics
    aggregated_stats = defaultdict(lambda: {'count': 0, 'total_logit': 0, 'total_area': 0})
    all_label_strings = []
    count = 0

    for image_name in color_image_names:
        """if image_name not in ["20231108_112532_color.jpg", "20231108_112527_color.jpg", "20231108_112933_color.jpg",  "20231108_112925_color.jpg"]:
            continue"""
        count += 1
        """if count > 500:
            break"""

        label_info = label_infos[image_name]
        labels = [label.label.replace(",", " ") for label in label_info]
        # if labels contain a "," remove it

        all_label_strings.extend(labels)
        # get depth image
        depth_raw = o3d.io.read_image(paths['depth_sensor'].format(image_name.split("_color")[0]))
        # scaling factor
        s = 0.453
        depth = np.asarray(depth_raw).astype(np.float32) / 1000 * s
        filter_depth = 3
        threshold_depth = filter_depth
        depth[depth > threshold_depth] = 0
        # get prediction mask
        mask_image = cv2.imread(paths["prediction_masks"].format(image_name), cv2.IMREAD_GRAYSCALE)
        masks = seperate_masks(mask_image)
        depth_masks = []
        for mask in masks:
            depth_masks.append(depth * mask)
        # make histogram of depth values for each mask, merge them into one histogram and show plot
        colors = cm.get_cmap('tab20', len(depth_masks))
        plt.figure(figsize=(10, 6))

        for i, mask in enumerate(depth_masks):
            plt.hist(mask[mask > 0], color=colors(i), bins=100, label=f'{str(label_info[i+1].label) + str(label_info[i+1].value)}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'{image_name}')
        plt.tight_layout()
        histogram_path = os.path.join(stats_dir, f'{image_name}_histogram.png')
        plt.savefig(histogram_path)
        plt.show()

        # check bbox overlap
        # get point cloud
        all_bbox = [label.box for label in label_info]

        #iou, intersecting_rect = get_bbox_intersections(all_bbox)



        #compute_statistics(label_info, aggregated_stats)

    """for label in aggregated_stats:
        if aggregated_stats[label]['count'] > 0:
            aggregated_stats[label]['average_logit'] = aggregated_stats[label]['total_logit'] / \
                                                       aggregated_stats[label]['count']
            aggregated_stats[label]['average_area'] = aggregated_stats[label]['total_area'] / \
                                                      aggregated_stats[label]['count']"""

    a=0
    # save to stats_dir as csv

    # save all_label_strings to stats_dir as csv
    all_label_strings = np.array(all_label_strings)
    np.savetxt(os.path.join(stats_dir, 'all_label_strings.csv'), all_label_strings, delimiter=',', fmt='%s')

    return aggregated_stats


if __name__ == '__main__':
    aggregated_stats = compute_stats()