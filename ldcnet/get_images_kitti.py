import data
import argparse
from model import ENet, LDCNet
import os
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from CoordConv import AddCoordsNp
import cv2

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("/home/javgal/mmdetection_clean/mmdetection/ldcnet/dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                    (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    # K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    # K[1, 2] = K[1, 2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    K[0, 2] = K[0, 2] - 13;
    K[1, 2] = K[1, 2] - 11.5;
    return K

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()

    depth = depth_png.astype(np.float) / 256.
    # depth = depth_png.astype(np.float)
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral

vmin=0
vmax=95
cmap= "viridis"

def depth_colorize(depth):
    cmap = plt.cm.jet
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach1280()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--input-height', type=int, default=352, help='input image height')
    parser.add_argument('--input-width', type=int, default=1216, help='input image width')
    parser.add_argument('--output-height', type=int, default=1280, help='output image height')
    parser.add_argument('--output-width', type=int, default=1920, help='output image width')
    # parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    # parser.add_argument('--save-directory', help='the checkpoint file to resume from')
    parser.add_argument('--model', type=str, default=LDCNet , help='model type(LDCNet or ENet)')
    parser.add_argument('--depth-path', required=True, help='path to kitti dataset depth')
    parser.add_argument('--raw-path', required=True, help='path to kitti dataset raw')
    parser.add_argument('--device', type=int, help='graphic id number, stay empty for cpu')
    parser.add_argument('--workers', type=int, default=4 , help='workers')
    parser.add_argument('--model-path', required=True, help='path to trained model')

    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    h, w = args.input_height, args.input_width
    h2, w2 = args.output_height, args.output_width
    device_type = args.device
    model_type = args.model

    to_tensor = transforms.ToTensor()
    to_float_tensor = lambda x: to_tensor(x).float()
    transform = transforms.Compose([to_float_tensor])

    kitti_depth_route = args.depth_path
    kitti_raw_route = args.raw_path

    test_dataset = data.KittiDataset(h, w, kitti_depth_route, kitti_raw_route, "test",transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda:" + str(device_type)) if isinstance(device_type,int) else "cpu"

    # model_path = "/home/javgal/kitti_depth_clean/results/ENet_Simple_1216x352/ENet_Simple_Best.pth"
    model_path = args.model_path

    model = None

    if(model_type == LDCNet):
        model = LDCNet(h, w).to(device)
    elif(model_type == ENet):
        model = ENet(h, w).to(device)

    model.load_state_dict(torch.load(model_path))
    
    model.eval().to(device)

    if not os.path.exists("ldcnet/results/images_depth"):
        os.makedirs("ldcnet/results/images_depth")
    if not os.path.exists("ldcnet/results/images_colored"):
        os.makedirs("ldcnet/results/images_colored")

    for batch_features in test_loader:
        rgb = batch_features["rgb"]
        d = batch_features["d"]
        rgb_name = batch_features["rgb_name"][0]
        gt = batch_features['gt'].clone().detach().float()
        gt = gt.view(-1, 1, h, w).to(device)

        features = np.zeros((d.shape[0], 4, h, w))

        features[:,0:3,:,:] = rgb
        features[:,3,:,:] = np.reshape(d, (rgb.shape[0], h, w))

        args = {"position": batch_features["position"].clone().detach().view(-1, 2, h, w).to(device), "K": batch_features["K"].clone().detach().view(-1, 3, 3).to(device)}

        with torch.no_grad():
            batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)

            batch_features = batch_features.float()

            if(model_type == LDCNet):
                out = model(batch_features, args)
            elif(model_type == ENet):
                _ , _ , out = model(batch_features, args)

            out_image = out.cpu().detach().numpy()[0,0,:,:]
            print(rgb_name)

            image_to_write = (cv2.resize(out_image,dsize=(w2,h2), interpolation=cv2.INTER_AREA)*256).astype(np.uint16)
            res_name = (rgb_name.split("/")[-1]).split(".")[0] + "_LDCNet_u16.png"

            cv2.imwrite("ldcnet/results/images_depth/" + res_name, image_to_write)

            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            plt.imshow(depth_colorize(image_to_write/ 256.), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.savefig("ldcnet/results/images_colored/" + res_name, bbox_inches='tight')

            plt.show()
            plt.close()


if __name__ == '__main__':
    main()