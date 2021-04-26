import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim

from api import PRN
from model.resfcn256 import ResFCN256

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize
from tools.prnet_loss import WeightMaskLoss, INFO

from config.config import FLAGS

from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import math


# Set random seem for reproducibility
manualSeed = 5
INFO("Random Seed", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.enabled = False


def main(data_dir):
    # 1) Create Dataset of 300_WLP & Dataloader.
    wlp300 = PRNetDataset(root_dir=data_dir,
                          transform=transforms.Compose([ToTensor(),
                                                        ToNormalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])]))

    wlp300_dataloader = DataLoader(dataset=wlp300, batch_size=FLAGS['batch_size'], shuffle=True, num_workers=0)

    # 2) Intermediate Processing.
    transform_img = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])
    ])

    # # 3) Create PRNet model.
    # start_epoch, target_epoch = FLAGS['start_epoch'], FLAGS['target_epoch']
    # model = ResFCN256()
    #
    # # Load the pre-trained weight
    # if FLAGS['resume'] and os.path.exists(os.path.join(FLAGS['images'], "3channels.pth")):
    #     state = torch.load(os.path.join(FLAGS['images'], "3channels.pth"))
    #     model.load_state_dict(state['prnet'])
    #     start_epoch = state['start_epoch']
    #     INFO("Load the pre-trained weight! Start from Epoch", start_epoch)
    #
    # model.to("cuda")
    prn = PRN(os.path.join(FLAGS['images'], "6channels.pth"))

    bar = tqdm(wlp300_dataloader)
    nme_list = []
    for i, sample in enumerate(bar):
        uv_map, origin = sample['uv_map'].to(FLAGS['device']), sample['origin'].to(FLAGS['device'])
        # print(origin.shape)
        # Inference.
        # origin = cv2.resize(origin, (256, 256))
        # origin = transform_img(origin)
        # origin = origin.unsqueeze(0)
        default_uv_map = torch.zeros([FLAGS['batch_size'], 3, 256, 256]).to(FLAGS['device'])
        input = torch.cat((origin.cuda(), default_uv_map), 1)
        uv_map_result = prn.net_forward(input)
        final_input = torch.cat((origin.cuda(), uv_map_result), 1)
        uv_map_result = prn.net_forward(final_input)
        out = uv_map_result.cpu().detach().numpy()
        uv_map_result = np.squeeze(out)
        cropped_pos = uv_map_result * 255
        uv_map_result = cropped_pos.transpose(1, 2, 0)

        out = uv_map.cpu().detach().numpy()
        uv_map = np.squeeze(out)
        cropped_pos = uv_map * 255
        uv_map = cropped_pos.transpose(1, 2, 0)

        kpt_predicted = prn.get_landmarks(uv_map_result)[:, :2]
        kpt_gt = prn.get_landmarks(uv_map)[:, :2]

        nme_sum = 0
        for j in range(kpt_gt.shape[0]):
            x = kpt_gt[j][0] - kpt_predicted[j][0]
            y = kpt_gt[j][1] - kpt_predicted[j][1]
            L2_norm = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
            # bounding box size has been fixed to 256x256
            d = 256*256
            error = L2_norm/d
            nme_sum += error
        nme_list.append(nme_sum/68)

    print(np.mean(nme_list))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", help="specify input directory.")
    args = parser.parse_args()
    main(args.test_dir)