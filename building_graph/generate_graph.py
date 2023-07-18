from pathlib import Path
import json
from io_functions.sgg import load_sgg_data




class PathConfigurator:
    def __init__(self, root_dir: Path, project_name: str):
        self.root_dir = root_dir
        self.sg_dir = root_dir / (project_name + '_sg_pred')
        self.custom_data_info = self.sg_dir / 'custom_data_info.json'
        self.image_paths, self.ind_to_classes, self.ind_to_predicates = self.parse_custom_data_info()
        self.custom_prediction = self.sg_dir / 'custom_prediction.json'
        self.center_point_dir = root_dir / (project_name + '_center_point')

    def parse_custom_data_info(self):
        with open(self.custom_data_info, 'r') as f:
            data_info = json.load(f)
        return data_info['idx_to_files'], data_info['ind_to_classes'], data_info['ind_to_predicates']









def test():
    root_dir = Path('X:/99_exchange/Du/SG_test_data')
    project_name = 'hallway'
    input_info = PathConfigurator(root_dir, project_name)


    # Given: for each image file, there is a centric points file, which contains the x y z and labels of the centric points
    # Goal: for the project one single graph should be generated.
    #       centerpoints from the different perspecives are aggregated if they have the same semantic label and lie close to each other.
    #       the edges are the relationships between the centerpoints,
    # for each image file
    # 1. find the corresponding centric points file, and parse the centric points file
    # 2. find relationships between the centric points via the labels
    # 3. intermediate: formulate networkx graphs and plot them in plotly
    # 4. TO THINK: how to merge the graphs of different images?


    # merge to one point cloud file
    # look u




















if __name__ == '__main__':
    test()