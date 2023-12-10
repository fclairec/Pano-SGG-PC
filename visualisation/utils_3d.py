import os
import os.path as osp
import open3d as o3d
import numpy as np

def compute_and_write_hull(output_dir_center_bbox, image_name, o3d_pcd, legend_color, inst_i=None):
    # Compute the convex hull
    try:
        nb_neighbors = 50
        std_ratio = 2.0
        cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        inlier_cloud = o3d_pcd.select_by_index(ind)

        hull, _ = inlier_cloud.compute_convex_hull()
    except:
        print("no hull found")
        return
    # hull.paint_uniform_color([1, 0, 0, 0.5])  # Example: Red color with 50% transparency

    lines = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    lines.paint_uniform_color([1, 0, 0])

    # o3d.visualization.draw_geometries([o3d_pcd, lines])

    # Convert the hull to a mesh
    hull_mesh = o3d.geometry.TriangleMesh(hull.vertices, hull.triangles)

    if inst_i is not None:
        hull_obj_path = osp.join(output_dir_center_bbox, image_name.split("_color")[0] + f'_{inst_i}.obj')
        hull_mtl_path = os.path.join(output_dir_center_bbox, image_name.split("_color")[0] + f'_{inst_i}.mtl')
    else:
        hull_obj_path = osp.join(output_dir_center_bbox, image_name.split("_color")[0] + '.obj')
        hull_mtl_path = os.path.join(output_dir_center_bbox, image_name.split("_color")[0] + '.mtl')

    o3d.io.write_triangle_mesh(hull_obj_path, hull_mesh)

    mtl_content = (
        "newmtl TransparentMaterial\n"
        f"Kd {legend_color[0]} {legend_color[1]} {legend_color[2]}\n"
        "d 0.6  # Opacity: 0.5 means 50% transparent\n"
    )

    # Write the MTL content to the file
    with open(hull_mtl_path, 'w') as mtl_file:
        mtl_file.write(mtl_content)

    with open(hull_obj_path, 'r') as file:
        obj_content = file.readlines()

    # Insert the MTL file reference after the first line
    obj_content.insert(1, f'mtllib {os.path.basename(hull_mtl_path)}\n')
    # Insert usemtl TransparentMaterial before the first 'f' line
    for i, line in enumerate(obj_content):
        if line.startswith('f'):
            obj_content.insert(i, 'usemtl TransparentMaterial\n')
            break
    # Write the modified content back to the OBJ file
    with open(hull_obj_path, 'w') as file:
        file.writelines(obj_content)


def write_and_show_pcd(point_clouds_path, image_name, pcd: o3d.geometry.PointCloud, inst_i=None):
    # point cloud to xyz point cloud file
    if inst_i is not None:
        xyz_path = osp.join(point_clouds_path, image_name.split("_color")[0] + f'_{inst_i}.xyz')
    else:
        xyz_path = osp.join(point_clouds_path, image_name.split("_color")[0] + '.xyz')
    # Manually format each point and its color into a string
    formatted_lines = [
        f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}"
        for point, color in zip(pcd.points, np.array(pcd.colors)*255)
    ]
    # Join the lines into a single string
    data_str = "\n".join(formatted_lines)

    # Write to the XYZ file
    with open(xyz_path, 'w') as file:
        file.write(data_str)
