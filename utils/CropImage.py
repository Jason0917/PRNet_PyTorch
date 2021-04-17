import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform

# sys.path.append('.')
# import face3d
# from face3d import mesh
# from face3d.morphable_model import MorphabelModel

image_h=256
image_w=256

root_path = "../300VW"

for i in range(1, 1838):
    image_path = root_path + "/vid/" + str(i) + ".jpg"
    mat_path = root_path + "/landmarks/" + str(i).zfill(6) + ".mat"
    image = io.imread(image_path) / 255.
    [h, w, c] = image.shape

    # Load 68 keypoints
    info = sio.loadmat(mat_path)
    kpt = info['Y']
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

    io.imsave('{}/{}/{}'.format("../300VW", 'cropped', str(i) + '.jpg'), np.squeeze(cropped_image))