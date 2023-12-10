import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cv2
import numpy as np

def plot_depths(depth_masks, label_info, image_name, stats_dir):
    # make histogram of depth values for each mask, merge them into one histogram and show plot
    colors = cm.get_cmap('tab20', len(depth_masks))
    plt.figure(figsize=(10, 6))
    legend_colors = []
    for i, mask in enumerate(depth_masks):
        plt.hist(mask[mask > 0], color=colors(i), bins=100, alpha=0.6,
                 label=f'{str(label_info[i + 1].label) + str(label_info[i + 1].value)}')
        legend_colors.append(colors(i))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{image_name}')
    plt.tight_layout()
    histogram_path = os.path.join(stats_dir, f'{image_name}_histogram.png')
    plt.savefig(histogram_path)
    plt.show()

    return legend_colors


def show_box(box, ax, label, color):
    x0, y0 = int(box[0]), int(box[1])
    w, h = int(box[2]) - int(box[0]), int(box[3]) - int(box[1])
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def show_mask(mask, ax, color):
    color = np.array(color[:3])
    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 4))  # Initialize a transparent layer
    mask_image[:, :, 0] = color[0]
    mask_image[:, :, 1] = color[1]
    mask_image[:, :, 2] = color[2]
    mask_image[:, :, 3] = np.where(mask, 1, 0)  # Apply alpha value for mask


    ax.imshow(mask_image, interpolation='nearest', alpha=0.45)  # Overlay with some transparency


def recolor_masks(masks, label_info, legend_colors, mask_image, color_image):
    """ colors the masks onto the color image. plots the bounding box, semantic label and logit value on the image"""
    print(f"recoloring masks {len(masks)}")
    label_info = label_info[1:]
    # Create a figure and axis for drawing
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(color_image)

    # Iterate over masks, label_info, and colors
    for mask, info, color in zip(masks, label_info, legend_colors):
        bbox, logit = info.box, info.logit
        label = f"{info.label}_{info.value} ({logit:.2f})"


        # Draw bounding box and label
        show_box(bbox, ax, label, color=color)
        show_mask(mask, ax, color=color)

    ax.axis('off')
    plt.savefig(mask_image,
        bbox_inches="tight", dpi=100, pad_inches=0.0
    )

    return