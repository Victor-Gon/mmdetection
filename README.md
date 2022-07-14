# Object Detection Using Depth Completion and Camera-LiDAR Fusion for Autonomous Driving

## Contents
1. [Dependency](#dependency)
0. [Kitti Dataset Setup](#kitti-dataset-setup)
0. [LDCNet](#ldcnet)
0. [MMDetection](#mmdetection)
0. [Citation](#citation)

## Dependency

## Kitti Dataset Setup

### Data

First, you will have to register at [KITTI Website](http://www.cvlibs.net/datasets/kitti/) and then you will be able to download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset and [KITTI Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) Dataset from the website.

The overall data directory is structured as follows:
```
├── kitti_depth
|   ├── depth
|   |   ├──data_depth_annotated
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_velodyne
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_selection
|   |   |  ├── test_depth_completion_anonymous
|   |   |  |── test_depth_prediction_anonymous
|   |   |  ├── val_selection_cropped
```

```
├── kitti_raw
|   ├── 2011_09_26
|   ├── 2011_09_28
|   ├── 2011_09_29
|   ├── 2011_09_30
|   ├── 2011_10_03
```

## LDCNet
### Method

Our proposed LiDAR Depth Completion network (LDCNet). The network outputs a dense depth map combining a camera image and the sparse LiDAR projections. In the encoder, 3D position maps are concatenated to the feature maps to encode geometric information. The decoder upsamples the feature maps using deconvolution. The numbers below the maps indicate the number of channels. K and S indicate kernel size and stride in the convolution, respectively.

<div align=center><img src="https://github.com/carranza96/mmdetection/blob/javi/images/LDCNet-1.png" width = "100%" height = "100%" /></div>

### Train

Train default 352x1216 LDCNet model.

```bash
python ldcnet/train.py --epochs 20 --batch-size 16 --depth-path kitti_dataset/kitti_depth --raw-path kitti_dataset/kitti_raw --device 0
```

### Evaluation

Evaluate RMSE and execution time for a trained model.

```bash
python ldcnet/val.py --depth-path kitti_dataset/kitti_depth --raw-path kitti_dataset/kitti_raw --model-path results/train1/model_Best.pth --device 0
```

## MMDetection

### Train

You will need a configuration archive like [this](https://github.com/carranza96/mmdetection/tree/javi/configs/waymo_open/lidar/faster_rcnn_r50_fpn_fp16_4x2_1x_1280x1920_ldcnet.py) where you may edit base model, dataset and training schedule.

Now you will have to configure  [two_stage.py](https://github.com/carranza96/mmdetection/tree/javi/mmdet/models/detectors/two_stage.py) or  [single_stage.py](https://github.com/carranza96/mmdetection/tree/javi/mmdet/models/detectors/single_stage.py) depending on which base model you have chosen. You may change `model_type` to the fusion model you want to use (or without fusion model to work with sparse lidar) and `model_path` to indicate where the .pth of the model is. Then you may select `fusion_type`, and for normalization, `mean` and `std` for the different input channels (Dense lidar mean and std are calculated in ldcnet/val.py).

Train model.

```bash
scripts/train_script_multigpu_ldcnet.sh
```

## Citation