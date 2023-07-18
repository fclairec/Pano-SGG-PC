import json
import os



def load_sgg_data(box_topk=8, rel_topk=10, custom_prediction_path=None, custom_data_info_path=None):
    '''
    # parameters
    box_topk = 8 # select top k bounding boxes
    rel_topk = 10 # select top k relationships
    '''
    # load the following to files from DETECTED_SGG_DIR
    custom_prediction = json.load(open(custom_prediction_path))
    custom_data_info = json.load(open(custom_data_info_path))
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
                label = str(all_rel_pairs[i][0]) + '_' + box_labels[all_rel_pairs[i][0]] + ' => ' + ind_to_predicates[
                    all_rel_labels[i]] + ' => ' + str(all_rel_pairs[i][1]) + '_' + box_labels[all_rel_pairs[i][1]]
                rel_labels.append(label)

        rel_labels = rel_labels[:rel_topk]
        rel_scores = rel_scores[:rel_topk]

        prediction_info_dict[img_name] = {'boxes': boxes, 'box_labels': box_labels, 'box_scores': box_scores,
                                          'rel_labels': rel_labels, 'rel_scores': rel_scores, 'image_idx': image_idx}

    return prediction_info_dict