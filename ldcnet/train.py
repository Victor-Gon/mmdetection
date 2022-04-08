import data
import argparse
from model import ENet, LDCNet
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import torch
import torch.nn as nn
import math
import time
import datetime

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

def adjust_learning_rate(lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    #lr = lr_init * (0.5**(epoch // 5))
    #'''
    lr = lr_init
    if (epoch >= 10):
        lr = lr_init * 0.5
    if (epoch >= 15):
        lr = lr_init * 0.1
    if (epoch >= 25):
        lr = lr_init * 0.01
    #'''

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Print iterations progress
def printProgress(iteration, total):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(50 * iteration // total)
    bar = '█' * filledLength + '-' * (50 - filledLength)
    print(f'\r{""} |{bar}| {percent}% {""}', end = '\r')

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
    parser.add_argument('--epochs', default=12, help='number of epochs')

    
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
    epochs = args.epochs
    
    temp_start = time.gmtime()
    temp = "(" + str(temp_start[2]) + "," + str(temp_start[1]) + "," + str(temp_start[0]) + "), " + str(temp_start[3]) + ":" + str(temp_start[4]) + ":" + str(temp_start[5])
    
    to_tensor = transforms.ToTensor()
    to_float_tensor = lambda x: to_tensor(x).float()
    transform = transforms.Compose([to_float_tensor])

    train_dataset = data.KittiDataset(h, w, kitti_depth_route, kitti_raw_route, "train",transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_dataset = data.KittiDataset(h, w, kitti_depth_route, kitti_raw_route, "val",transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + str(device_type)) if isinstance(device_type,int) else "cpu"

    absolute_loss = 9999999999999
    a = len(train_loader)

    model = None

    print("Device used: " + ("cuda:" + str(device_type) if isinstance(device_type,int) else "cpu"))


    if(model_type == LDCNet):
        model = LDCNet(h, w).to(device)
    elif(model_type == ENet):
        model = ENet(h, w).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6, betas=(0.9, 0.99))

    depth_criterion = MaskedMSELoss()

    if not os.path.exists("ldcnet/results"):
        os.makedirs("ldcnet/results")
    os.mkdir("ldcnet/results/" + temp)

    with open("ldcnet/results/" + temp + "/" + temp + ".txt", 'w') as file:
        file.write('Train ' + temp + "\n")
        file.write("Model: " + str(model_type) +  "\nEpochs: " +  str(epochs) +  "\n\n")

    for epoch in range(epochs):
        i = 0
        t_loss = 0
        v_loss = 0
        relative_time = 0
        absolute_time = 0
        
        lr = adjust_learning_rate(1e-3, optimizer, epoch)

        with open("ldcnet/results/" + temp + "/" + temp + ".txt", 'a') as file:
            for batch_features in train_loader:
                start = time.time()
                rgb = batch_features["rgb"]
                d = batch_features["d"]
                gt = batch_features["gt"]
                gt = gt.view(-1, 1, h, w).to(device)

                features = np.zeros((d.shape[0], 4, h, w))

                features[:,0:3,:,:] = rgb
                features[:,3,:,:] = np.reshape(d, (rgb.shape[0], h, w))

                args = {"position": batch_features["position"].clone().detach().view(-1, 2, h, w).to(device), "K":  batch_features["K"].clone().detach().view(-1, 3, 3).to(device)}

                batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)
                optimizer.zero_grad()

                batch_features = batch_features.float()

                if(model_type == LDCNet):
                    out = model(batch_features, args)

                    depth_loss = depth_criterion(out, gt)
                    st1_loss = 0
                    st2_loss = 0
                elif(model_type == ENet):
                    st1_pred, st2_pred, out = model(batch_features, args)

                    depth_loss = depth_criterion(out, gt)
                    st1_loss = depth_criterion(st1_pred, gt)
                    st2_loss = depth_criterion(st2_pred, gt)

                # RMSE
                w_st1, w_st2 = 0, 0

                if(epoch <= 1):
                    w_st1, w_st2 = 0.2, 0.2
                elif(epoch <= 3):
                    w_st1, w_st2 = 0.05, 0.05
                else:
                    w_st1, w_st2 = 0, 0
                
                train_loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss

                train_loss.backward()
                optimizer.step()

                relative_time = time.time() - start
                absolute_time = absolute_time + relative_time

                finish = (epochs - epoch - 1) * a * (absolute_time / (i+1)) + (absolute_time / (i+1)) * a - absolute_time

                printProgress(i, a)

                if(i%50 == 0):
                    text_to_write = "Epoch: [" +  str(epoch + 1) +  "] [" +  str(i) +  " / " +  str(a) +  "]  eta: " +  str(datetime.timedelta(seconds=int(finish))) + ", Loss: " + str(float(train_loss)) 
                    print(115 * " ", end="\r")
                    print(text_to_write + "\n", end = '\r')
                    file.write(text_to_write + "\n")

                t_loss = t_loss + float(train_loss)
                i = i + 1
            
            t_loss = t_loss / a

            absolute_time_val = 0

            for batch_features in val_loader:
                with torch.no_grad():
                    rgb = batch_features["rgb"]
                    d = batch_features["d"]
                    gt = batch_features['gt']
                    gt = gt.view(-1, 1, h, w).to(device)

                    features = np.zeros((d.shape[0], 4, h, w))

                    features[:,0:3,:,:] = rgb
                    features[:,3,:,:] = np.reshape(d,(rgb.shape[0], h, w))

                    args = {"position": batch_features["position"].clone().detach().view(-1, 2, h, w).to(device), "K":  batch_features["K"].clone().detach().view(-1, 3, 3).to(device)}

                    batch_features = torch.tensor(features).view(-1, 4, h, w).to(device)

                    batch_features = batch_features.float()
                    start = time.time()

                    if(model_type == LDCNet):
                        out = model(batch_features, args)
                    elif(model_type == ENet):
                        _ , _ , out = model(batch_features, args)

                    relative_time = time.time() - start
                    absolute_time_val = absolute_time_val + relative_time
                    
                    # RMSE

                    valid_mask = gt > 0.1

                    # convert from meters to mm
                    output_mm = 1e3 * out[valid_mask]
                    target_mm = 1e3 * gt[valid_mask]

                    abs_diff = (output_mm - target_mm).abs()

                    mse = float((torch.pow(abs_diff, 2)).mean())
                    val_loss = math.sqrt(mse)

                    v_loss = v_loss + float(val_loss)
            
            model.train()
            v_loss = v_loss / len(val_loader)
            ex_time = absolute_time_val / len(val_loader)

            if (v_loss < absolute_loss):
                torch.save(model.state_dict(),"ldcnet/results/{}/model_Best.pth".format(temp))
                absolute_loss = v_loss

            
            print(115 * " ", end="\r")
            print("epoch : {}/{}, validation loss = {:.6f} , training loss = {:.6f}, Execution time = {:.6f}".format(epoch + 1, epochs, v_loss, t_loss, ex_time))
            to_text = "epoch : {}/{}, validation loss = {:.6f} , training loss = {:.6f} \nExecution time = {:.6f}\n\n".format(epoch + 1, epochs, v_loss, t_loss, ex_time)
            file.write(to_text)

            torch.save(model.state_dict(),"ldcnet/results/{}/model_epoch_{}.pth".format(temp, epoch))

    print("Best result: ", absolute_loss)


if __name__ == '__main__':
    main()