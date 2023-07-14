# Config file
CONFIG_FILE=configs/mamoas/faster_rcnn.py
# Work dir (path where checkpoints and predictions will be saved)
WORK_DIR=results/faster_rcnn

### TRAINING
# Single-GPU training
python tools/train.py ${CONFIG_FILE} ${WORK_DIR}

# Multi-GPU training
GPU_NUM=2
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --work-dir=${WORK_DIR}



#### TEST
python tools/test.py ${CONFIG_FILE} ${WORK_DIR}/epoch_12.pth