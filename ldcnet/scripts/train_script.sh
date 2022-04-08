DEPTH_PATH=/home/javgal/kitti_depth_clean/kitti_depth
RAW_PATH=/home/javgal/kitti_depth_clean/kitti_raw

python ldcnet/train.py --depth-path ${DEPTH_PATH} --raw-path ${RAW_PATH} --device 0

