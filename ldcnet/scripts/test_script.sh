DEPTH_PATH=/home/javgal/kitti_depth_clean/kitti_depth/depth_selection/test_depth_completion_anonymous/velodyne_raw
IMAGE_PATH=/home/javgal/kitti_depth_clean/kitti_depth/depth_selection/test_depth_completion_anonymous/image
MODEL_PATH=/home/javgal/mmdetection_clean/mmdetection/ldcnet/results/ldcnet_testing/model_Best.pth

python ldcnet/get_images_kitti.py --depth-path ${DEPTH_PATH} --raw-path ${IMAGE_PATH} --model-path ${MODEL_PATH} --device 0
