import os
import collections
import numpy as np
import struct
import argparse
import json

Predictions = collections.namedtuple(
    "predictions", ["value", "label", "logit", "box"]
)
def load_predictions(path, image_names):
    pred_info = {}
    for image_name in image_names:
        prediction_file = path.format(image_name)
        # load Json
        try:
            with open(prediction_file, 'r') as file:
                data = json.load(file)
                mask_data = data['mask']  # Extracting the 'mask' list
                predictions_list = [Predictions(value=item['value'], label=item['label'], logit=item.get('logit', None), box=item.get('box', None)) for item in mask_data]
                pred_info[image_name] = predictions_list

        except:
            continue

    return pred_info

