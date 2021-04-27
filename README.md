# PRNet PyTorch 1.1.0

![Github](https://img.shields.io/badge/PyTorch-v1.1.0-green.svg?style=for-the-badge&logo=data:image/png)
![Github](https://img.shields.io/badge/python-3.6-green.svg?style=for-the-badge&logo=python)
![Github](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge&logo=fire)

<p align="center"> 
<img src="docs/image/prnet.gif">
</p>

This is an unofficial pytorch implementation of **PRNet** since there is not a complete generating and training code
of [`300WLP`](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) dataset.

- Author: Samuel Ko, mjanddy, Zihao Jian, Minshan Xie.

### Our Contributions

① Based on the open source code, we proposed two potential improvement and have modified the network architecture and training codes. To use our modified network, please check out **out-and-in** branch or **multi-frames** branch.

② We have added testing code for this repository. Use test.py to estimate the Normalized Mean Error on 300WLP, or use benchmark.py to test on AFLW dataset.

③ We have added data pre-processing code. If you want to use your own facial images, please use utils/generate_posmap_300WLP.py or training and utils/cropImage.py for testing.

-------

### Noitce

Since replacing the default `PIL.Imgae` by `cv2.imread` in image reader, you need
do a little revise on your `tensorboard` package in `your_python_path/site-packages/torch/utils/tensorboard/summary.py`

What you should do is add `tensor = tensor[:, :, ::-1]` before `image = Image.fromarray(tensor)` in function `make_image(...)`.
```shell
...
def make_image(tensor, rescale=1, rois=None):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    height, width, channel = tensor.shape
    scaled_height = int(height * rescale)
    scaled_width = int(width * rescale)

    tensor = tensor[:, :, ::-1]
    image = Image.fromarray(tensor)
    ...
...
``` 

----
## ① Pre-Requirements 

Before we start generat uv position map and train it. The first step is generate BFM.mat according to [Basel Face Model](https://faces.dmi.unibas.ch/bfm/).
For simplicity, The corresponding `BFM.mat` has been provided [here](https://drive.google.com/open?id=1Bl21HtvjHNFguEy_i1W5g0QOL8ybPzxw).

After download it successfully, you need to move `BFM.mat` to `utils/`.

Besides, the essential python packages were listed in `requirements.txt`.

## ② Generate uv_pos_map

[YadiraF/face3d](https://github.com/YadiraF/face3d) have provide scripts for generating uv_pos_map, here i wrap it for 
Batch processing.

You can use `utils/generate_posmap_300WLP.py` as:

``` shell
python3 generate_posmap_300WLP.py --input_dir ./dataset/300WLP/IBUG/ --save_dir ./300WLP_IBUG/
```

Then `300WLP_IBUG` dataset is the proper structure for training PRNet:

```
- 300WLP_IBUG
 - 0/
  - IBUG_image_xxx.npy
  - original.jpg (original RGB)
  - uv_posmap.jpg (corresponding UV Position Map)
 - 1/
 - **...**
 - 100/ 
```

Except from download from [`300WLP`](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm),
I provide processed *original--uv_posmap* pair of IBUG [here](https://drive.google.com/open?id=16zZdkRUNdj7pGmBpZIwQMA00qGHLLi94).

*Please note that the provided dataset only contains about 1,700 samples. To train a robust and generic model, you are strongly recommended to download a full dataset and generate the ground truth of UV maps.

## ③ Training
After finish the above two step, you can train your own PRNet as:

``` shell

python3 train.py --train_dir ./300WLP_IBUG
```

The following image is used to judge the effectiveness of PRNet to unknown data.

(Original, UV_MAP_gt, UV_MAP_predicted)
![Test Data](docs/image/test_img.png)

## ④ Testing

You can use following command to test your model on 300WLP dataset.

``` shell
python3 test.py --test_dir ./300WLP_IBUG_Test
```

To test on AFLW dataset, please use this command.

``` shell
python3 benchmark.py -model results/latest.pth
```

## ⑤ Inference

You can use following instruction to do your prnet inference. The detail about parameters you can find in `inference.py`.
```shell
python3 inference.py -i input_dir(default is TestImages) -o output_dir(default is TestImages/results) --model model_path(default is results/latest.pth) --gpu 0 (-1 denotes cpu)
```
![Test Data](docs/image/inference_img.png)


--------
### Citation

If you use this code, please consider citing:

```
@inProceedings{feng2018prn,
  title     = {Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network},
  author    = {Yao Feng and Fan Wu and Xiaohu Shao and Yanfeng Wang and Xi Zhou},
  booktitle = {ECCV},
  year      = {2018}
}
```
