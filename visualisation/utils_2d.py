import numpy as np
def seperate_masks(mask_image):
    num_masks = np.max(mask_image)
    separated_masks = []
    for i in range(1, num_masks + 1):  # Starting from 1 to exclude the background
        mask = np.where(mask_image == i, 1, 0).astype(np.uint8)
        separated_masks.append(mask)
    return separated_masks

