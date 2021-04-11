#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from api import PRN
import time
import numpy as np
from glob import glob

from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw

from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex
import argparse


def extract_param(checkpoint_fp, root='', filelists=None, num_classes=62, device_ids=[0],
                  batch_size=1, num_workers=0):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    # checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(device_ids[0])
    model = PRN(checkpoint_fp)
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # model.load_state_dict(checkpoint)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    # model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            inputs = inputs.cuda()

            # Get the output landmarks
            pos = model.net_forward(inputs)

            out = pos.cpu().detach().numpy()
            pos = np.squeeze(out)
            cropped_pos = pos * 255
            pos = cropped_pos.transpose(1, 2, 0)

            if pos is None:
                continue

            # print(pos.shape)
            output = model.get_landmarks(pos)
            # print(output.shape)

            outputs.append(output)

        outputs = np.array(outputs, dtype=np.float32)
        print("outputs",outputs.shape)
    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def _benchmark_aflw(outputs):
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


# def benchmark_alfw_params(params):
#     outputs = []
#     for i in range(params.shape[0]):
#         lm = reconstruct_vertex(params[i])
#         outputs.append(lm[:2, :])
#     return _benchmark_aflw(outputs)


# def benchmark_aflw2000_params(params):
#     outputs = []
#     for i in range(params.shape[0]):
#         lm = reconstruct_vertex(params[i])
#         outputs.append(lm[:2, :])
#     return _benchmark_aflw2000(outputs)


def benchmark_pipeline(checkpoint_fp):
    device_ids = [0]

    def aflw():
        landmarks = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW_GT_crop',
            filelists='test.data/AFLW_GT_crop.list',
            device_ids=device_ids,
            batch_size=1)

        _benchmark_aflw(landmarks)

    def aflw2000():
        landmarks = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW2000-3D_crop',
            filelists='test.data/AFLW2000-3D_crop.list',
            device_ids=device_ids,
            batch_size=1)

        _benchmark_aflw2000(landmarks)

    aflw2000()
    aflw()


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    # parser.add_argument('--arch', default='mobilenet_1', type=str)
    parser.add_argument('-model', default='results/6channels.pth', type=str,
                        help='model path')
    args = parser.parse_args()

    benchmark_pipeline(args.model)


if __name__ == '__main__':
    main()