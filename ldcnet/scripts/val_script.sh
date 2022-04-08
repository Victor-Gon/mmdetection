DEPTH_PATH=/home/javgal/kitti_depth_clean/kitti_depth
RAW_PATH=/home/javgal/kitti_depth_clean/kitti_raw
MODEL_PATH=/home/javgal/mmdetection_clean/mmdetection/ldcnet/results/ldcnet_testing/model_Best.pth

python ldcnet/val.py --depth-path ${DEPTH_PATH} --raw-path ${RAW_PATH} --model-path ${MODEL_PATH} --device 0

