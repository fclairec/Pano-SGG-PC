from pathlib import Path
import json
from io_functions.sgg import load_sgg_data, load_sgg_data_pred_only
import networkx as nx
import os
from io_functions.plyfile import PlyDataReader





class PathConfigurator:
    def __init__(self, root_dir: Path, project_name: str):
        self.root_dir = root_dir
        self.sg_dir = root_dir / (project_name + '_sg_pred')
        self.custom_data_info = self.sg_dir / 'custom_data_info.json'
        self.center_point_dir = root_dir / (project_name + '_center_point')
        self.image_paths, self.ind_to_classes, self.ind_to_predicates = self.parse_custom_data_info(debug=True)
        self.custom_prediction = self.sg_dir / 'custom_prediction.json'


    def parse_custom_data_info(self, debug=False):
        with open(self.custom_data_info, 'r') as f:
            data_info = json.load(f)
        all_images =data_info['idx_to_files']
        if debug:
            # check the point clouds available in self.center_point_dir. image names are in the filename.
            images = os.listdir(self.center_point_dir)
            image_names = [os.path.splitext(image)[0]+".jpg" for image in images]
            # set all indices in all_images to 0 that are not in image_names
            for i in range(len(all_images)):
                if all_images[i].split("/")[-1].split(".")[0]+".jpg"  not in image_names:
                    all_images[i] = 0

        return all_images, data_info['ind_to_classes'], data_info['ind_to_predicates']


def assemble_nx_graph(nodes, adjacencies, node_attributes, edge_attributes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(adjacencies)
    nx.set_node_attributes(G, node_attributes)
    nx.set_edge_attributes(G, edge_attributes)
    return G



def merge_nodes():
    a=0










def test():
    root_dir = Path('X:/99_exchange/Du/SG_test_data')
    project_name = 'hallway'
    input_info = PathConfigurator(root_dir, project_name)

    prediction_info_dict = load_sgg_data_pred_only(8, 10, input_info.custom_prediction, input_info.image_paths, input_info.ind_to_classes, input_info.ind_to_predicates)

    # load the center points from ply files
    pcd = PlyDataReader()

    for images_id, prediction_info in prediction_info_dict.items():
        # load the center points from ply files
        center_point_file = input_info.center_point_dir / (images_id.split(".")[-2] + ".ply")
        # load ply point cloud


        # Read the initial PLY file
        pcd.read_ply(center_point_file)

        a=0






    a=0


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