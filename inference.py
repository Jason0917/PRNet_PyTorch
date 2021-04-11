# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from glob import glob
import scipy.io as sio
import argparse
import ast

from api import PRN
from torchvision import transforms, utils, models


from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
# from utils.render_app import get_visibility, get_uv_mask, get_depth_image
# from utils.write import write_obj_with_colors, write_obj_with_texture

from config.config import FLAGS

def main(args):
    if args.isShow or args.isTexture:
        import cv2
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

    # ---- transform
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])
    ])



    # ---- init PRN
    prn = PRN(args.model)
    # ------------- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    print("#" * 25)
    print("[PRNet Inference] {} picture were under processing~".format(total_num))
    print("#"*25)

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]

        # read image
        image = cv2.imread(image_path)
        [h, w, c] = image.shape

        # the core: regress position map
        image = cv2.resize(image, (256, 256))
        image_t = transform_img(image)
        image_t = image_t.unsqueeze(0)
        print(image_t.shape)
        pos = prn.net_forward(image_t.cuda())  # input image has been cropped to 256x256

        out = pos.cpu().detach().numpy()
        pos = np.squeeze(out)
        cropped_pos = pos * 255
        pos = cropped_pos.transpose(1, 2, 0)

        if pos is None:
            continue

        if args.is3d or args.isMat or args.isPose or args.isShow:
            # 3D vertices
            vertices = prn.get_vertices(pos)
            if args.isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        if args.isImage:
            cv2.imwrite(os.path.join(save_folder, name + '.jpg'), image)

        # if args.is3d:
        #     # corresponding colors
        #     colors = prn.get_colors(image, vertices)
        #
        #     if args.isTexture:
        #         if args.texture_size != 256:
        #             pos_interpolated = cv2.resize(pos, (args.texture_size, args.texture_size), preserve_range=True)
        #         else:
        #             pos_interpolated = pos.copy()
        #         texture = cv2.remap(image, pos_interpolated[:, :, :2].astype(np.float32), None,
        #                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        #         if args.isMask:
        #             vertices_vis = get_visibility(vertices, prn.triangles, h, w)
        #             uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
        #             uv_mask = cv2.resize(uv_mask, (args.texture_size, args.texture_size), preserve_range=True)
        #             texture = texture * uv_mask[:, :, np.newaxis]
        #         write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture,
        #                                prn.uv_coords / prn.resolution_op)  # save 3d face with texture(can open with meshlab)
        #     else:
        #         write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles,
        #                               colors)  # save 3d face(can open with meshlab)

        # if args.isDepth:
        #     depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
        #     depth = get_depth_image(vertices, prn.triangles, h, w)
        #     cv2.imwrite(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
        #     sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth': depth})

        if args.isKpt or args.isShow:
            # get landmarks
            print(pos.shape)
            kpt = prn.get_landmarks(pos)
            print("size:", kpt.shape)
            np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)

        if args.isPose or args.isShow:
            # estimate pose
            camera_matrix, pose = estimate_pose(vertices)
            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)
            np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix)

            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)

        if args.isShow:
            # ---------- Plot
            image_pose = plot_pose_box(image, camera_matrix, kpt)
            cv2.imshow('sparse alignment', plot_kpt(image, kpt))
            cv2.imshow('dense alignment', plot_vertices(image, vertices))
            cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
            cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='test.data/AFLW2000-3D_crop/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--model', default='results/6channels.pth', type=str,
                        help='model path')
    parser.add_argument('--is3d', default=True, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=False, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=True, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=False, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')
    main(parser.parse_args())
