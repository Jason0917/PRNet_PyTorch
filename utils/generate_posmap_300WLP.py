# coding: UTF-8
'''
Generate uv position map of 300W_LP.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
import argparse
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return uv_coords


def run_posmap_300W_LP(bfm, image_path, mat_path, save_folder,  idx, uv_h=256, uv_w=256, image_h=256, image_w=256):
    # 1. load image and fitted parameters
    image_name = image_path.strip().split('/')[-1]
    image = io.imread(image_path) / 255.
    [h, w, c] = image.shape

    info = sio.loadmat(mat_path)
    print(info)
    pose_para = info['Pose_Para'].T.astype(np.float32)
    shape_para = info['Shape_Para'].astype(np.float32)
    exp_para = info['Exp_Para'].astype(np.float32)

    # 2. generate mesh
    # generate shape
    vertices = bfm.generate_vertices(shape_para, exp_para)
    # transform mesh
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1

    # 3. crop image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0,
                       bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top) / 2
    size = int(old_size * 1.5)
    # random pertube. you can change the numbers
    marg = old_size * 0.1
    t_x = np.random.rand() * marg * 2 - marg
    t_y = np.random.rand() * marg * 2 - marg
    center[0] = center[0] + t_x
    center[1] = center[1] + t_y
    size = size * (np.random.rand() * 0.2 + 0.9)

    # crop and record the transform parameters
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))

    # transform face position(image vertices) along with 2d facial image
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2] * tform.params[0, 0]  # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z

    # 4. uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c=3)

    # 5. save files
    if not os.path.exists(os.path.join(save_folder, str(idx) + '/')):
        os.mkdir(os.path.join(save_folder, str(idx) + '/'))

    io.imsave('{}/{}/{}'.format(save_folder, idx, 'original.jpg'), np.squeeze(cropped_image))
    np.save('{}/{}/{}'.format(save_folder, idx, image_name.replace('jpg', 'npy')), uv_position_map)
    io.imsave('{}/{}/{}'.format(save_folder, idx, 'uv_posmap.jpg'),
              (uv_position_map) / max(image_h, image_w))  # only for show

    # --verify
    # import cv2
    # uv_texture_map_rec = cv2.remap(cropped_image, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_tex.jpg')), np.squeeze(uv_texture_map_rec))


def generate_batch_sample(input_dir, save_folder, mat_dir):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # set para
    uv_h = uv_w = 256

    # load uv coords
    global uv_coords
    uv_coords = face3d.morphable_model.load.load_uv_coords('BFM_UV.mat')  #
    uv_coords = process_uv(uv_coords, uv_h, uv_w)

    # load bfm
    bfm = MorphabelModel('BFM/BFM.mat')

    # Batch generating uv_map Dataset
    """
    @date: 2019/07/19
    Train Dataset:
        AFW. 10413.
        HELEN. 75351.
        LFPW. 33111.
    Test Dataset:
        IBUG. 3571.

    """
    base = 0
    id = 0
    for idx, item in enumerate(os.listdir(input_dir)):
        if 'jpg' in item:
            ab_path = os.path.join(input_dir, item)
            img_path = ab_path

            mat_path = os.path.join(mat_dir,item)
            mat_path = mat_path.replace('jpg', 'mat')
            #save_folder = os.path.join(save_folder,tem+item)
            run_posmap_300W_LP(bfm, img_path, mat_path, save_folder, id + base)
            print("Number {} uv_pos_map was generated!".format(idx))
            id = id+1


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--save_dir", help="specify output uv_map directory.")
    # parser.add_argument("--input_dir", help="specify input origin mat & image directory.")
    # args = parser.parse_args()
    input_dir = '../Ibug'



    for dir in os.listdir(input_dir):
         # input_path = os.path.join(args.input_dir, dir, 'vid')
         # output_path = os.path.join(args.save_dir, dir)
         input_path = os.path.join(input_dir, dir)
         print(input_path)
         output_path = os.path.join('../IBUG_UV_Maps', dir)
         mat_path = os.path.join(input_dir,dir)
         generate_batch_sample(save_folder=output_path, input_dir=input_path, mat_dir = mat_path)