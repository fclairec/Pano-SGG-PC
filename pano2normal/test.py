import math
import os
from pathlib import Path
import numpy as np
from PIL import Image
import utils


def e2c(e_img, face_w=256, mode='bilinear', cube_format='dice'):
    '''
    e_img:  ndarray in shape of [H, W, *]
    face_w: int, the length of each face of the cubemap
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = utils.xyzcube(face_w)
    uv = utils.xyz2uv(xyz)
    coor_xy = utils.uv2coor(uv, h, w)

    cubemap = np.stack([
        utils.sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_h2list(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_h2dict(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_h2dice(cubemap)
    else:
        raise NotImplementedError()

    return cubemap


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear'):
    '''
    e_img:   ndarray in shape of [H, W, *]
    fov_deg: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]


    h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180

    in_rot = in_rot_deg * np.pi / 180

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180
    xyz = utils.xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = utils.xyz2uv(xyz)
    coor_xy = utils.uv2coor(uv, h, w)

    pers_img = np.stack([
        utils.sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    return pers_img

def using_e2cube():
    img_path = '/mnt/c/Users/ge25yak/Desktop/Navis_CMS/pano/00000-pano.jpg'
    e_img = np.array(Image.open(img_path))
    cubes = e2c(e_img, face_w=1024, mode='bilinear', cube_format='list')
    for i, cube in enumerate(cubes):
        save_path = os.path.join(f"{os.path.basename(img_path)}_{i}.jpg")
        Image.fromarray(cube).save(save_path, quality=95, subsampling=0)
        print(f"saved cubemap {save_path}!")

def using_e2p(img_path):
    e_img = np.array(Image.open(img_path))
    face_centers = np.array(
    [
        [180, 0],
        [-90, 0],
        [0, 0],
        [90, 0],
        [0, 90],
        [0, -90],
    ])   

    cubes = []
    for center in face_centers:
        cubes.append(e2p(e_img, (90, 90), center[0], center[1], (1024, 1024)))

    img_name = os.path.basename(img_path)[:-4]
    root_dir = Path(img_path).parent.parent.absolute()
    cubmap_dir = os.path.join(root_dir, 'cube_maps', img_name)
    if not os.path.exists(cubmap_dir):
        os.mkdir(cubmap_dir)

    for i, cube in enumerate(cubes):
        save_path = os.path.join(cubmap_dir, f"{img_name}_{i}.jpg")
        Image.fromarray(cube).save(save_path, quality=95, subsampling=0)
        print(f"saved cubemap {save_path}!")

def getFileList(dir, Filelist, ext="jpg"):
    """
    Get the image path recursively
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist

if __name__ == "__main__":
    img_path_list = getFileList('/mnt/c/Users/ge25yak/Desktop/Navis_CMS/pano', [])
    for img_path in img_path_list:
        using_e2p(img_path)

    # with current setting, e2p and e2c generate same results
    # using_e2cube()