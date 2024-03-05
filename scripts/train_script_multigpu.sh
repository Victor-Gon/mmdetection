MODEL=faster-rcnn_r50_fpn_1x_waymo

# ./tools/dist_train.sh configs/waymo_open/${MODEL}.py 2 \
#     --work-dir=saved_models/${MODEL}

./tools/dist_train.sh configs/waymo_open/${MODEL}.py 3 \
    --work-dir=saved_models/${MODEL}_amp --amp

