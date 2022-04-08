import data
import argparse
from model import ENet, LDCNet
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import math
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--height', type=int, default=352, help='input image height')
    parser.add_argument('--width', type=int, default=1216, help='input image width')
    # parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    # parser.add_argument('--save-directory', help='the checkpoint file to resume from')
    parser.add_argument('--model', type=str, default=LDCNet , help='model type(LDCNet or ENet)')
    parser.add_argument('--batch-size', type=int, default=1 , help='batch size')
    parser.add_argument('--depth-path', required=True, help='path to kitti dataset depth')
    parser.add_argument('--raw-path', required=True, help='path to kitti dataset raw')
    parser.add_argument('--device', type=int, help='graphic id number, stay empty for cpu')
    parser.add_argument('--workers', type=int, default=4 , help='workers')
    parser.add_argument('--model-path', required=True, help='path to trained model')

    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    h, w = args.height, args.width
    model_type = args.model
    # kitti_depth_route = "/home/javgal/kitti_depth_clean/kitti_depth"
    # kitti_raw_route = '/home/javgal/kitti_depth_clean/kitti_raw'
    kitti_depth_route = args.depth_path
    kitti_raw_route = args.raw_path
    device_type = args.device
    
    temp_start = time.gmtime()
    temp = "(" + str(temp_start[2]) + "," + str(temp_start[1]) + "," + str(temp_start[0]) + "), " + str(temp_start[3]) + ":" + str(temp_start[4]) + ":" + str(temp_start[5])
    
    to_tensor = transforms.ToTensor()
    to_float_tensor = lambda x: to_tensor(x).float()
    transform = transforms.Compose([to_float_tensor])

    val_dataset = data.KittiDataset(h, w, kitti_depth_route, kitti_raw_route, "val",transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

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
    criterion = nn.MSELoss()

    i = 0
    a = len(val_loader)
    total_loss = 0
    gpu_total_time = 0
    for batch_features in val_loader:
        rgb = batch_features["rgb"]
        d = batch_features["d"]
        gt = torch.tensor(batch_features['gt']).float()
        gt = gt.view(-1, 1, h, w).to(device)

        features = np.zeros((d.shape[0], 4, h, w))

        features[:,0:3,:,:] = rgb
        features[:,3,:,:] = np.reshape(d, (rgb.shape[0], h, w))

        args = {"position": torch.tensor(batch_features["position"]).view(-1, 2, h, w).to(device), "K": torch.tensor(batch_features["K"]).view(-1, 3, 3).to(device)}

        num_images = len(val_loader)

        with torch.no_grad():
            batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)

            batch_features = batch_features.float()
            start = time.time()

            if(model_type == LDCNet):
                out = model(batch_features, args)
            elif(model_type == ENet):
                _ , _ , out = model(batch_features, args)

            gpu_time = time.time() - start
            gpu_total_time = gpu_total_time + (gpu_time / a)
            
            # RMSE
            valid_mask = gt > 0.1

            outputs_mm = 1e3 * out[valid_mask]
            gt_mm = 1e3 * gt[valid_mask]
            abs_diff = (outputs_mm - gt_mm).abs()
            mse = float((torch.pow(abs_diff, 2)).mean())
            loss = math.sqrt(mse)

            if(i % 10 == 0):
                print(i, " / ", a)
                print("RMSE: ", loss)

            total_loss = total_loss + (float(loss) / a)
            i = i + 1

    print("Mean RMSE: ", total_loss)
    print("Mean Execution Time: ", gpu_total_time)


if __name__ == '__main__':
    main()