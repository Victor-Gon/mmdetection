# Multi-modal object detection for autonomous driving using transfer learning for LiDAR depth completion

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

We recommend to paste the directory at ..... as default configuration
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

## MMDetection

## Citation