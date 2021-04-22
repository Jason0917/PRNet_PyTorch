import os.path as osp
import numpy as np
from math import sqrt
from utils.io import _load
import scipy.io as sio
from api import PRN

# main()
d = './IBUG_UV_Maps/1/0'
h = './Ibug/1/IBUG_image_003_1_0.mat'
yaw_list = _load(osp.join(d, 'IBUG_image_003_1_10.npy'))
# print('yaw_list', yaw_list)
print('yaw_list', yaw_list.shape)

prnet = PRN('./results/latest_for_IBUG1.pth1')
pre_ver68 = prnet.get_landmarks(yaw_list)
print(pre_ver68)
print(pre_ver68.shape)

mat = sio.loadmat(h)
# print(mat)
print('mat', mat['pt2d'].shape)