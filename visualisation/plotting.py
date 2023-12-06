import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def plot_depths(depth_masks, label_info, image_name, stats_dir):
    # make histogram of depth values for each mask, merge them into one histogram and show plot
    colors = cm.get_cmap('tab20', len(depth_masks))
    plt.figure(figsize=(10, 6))

    for i, mask in enumerate(depth_masks):
        plt.hist(mask[mask > 0], color=colors(i), bins=100, alpha=0.6,
                 label=f'{str(label_info[i + 1].label) + str(label_info[i + 1].value)}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{image_name}')
    plt.tight_layout()
    histogram_path = os.path.join(stats_dir, f'{image_name}_histogram.png')
    plt.savefig(histogram_path)
    plt.show()