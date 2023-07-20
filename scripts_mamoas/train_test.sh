MODEL=faster_rcnn
# Config file
CONFIG_FILE=configs/mamoas/${MODEL}.py
# Work dir (path where checkpoints and predictions will be saved)
WORK_DIR=results/${MODEL}

data_root="data/mamoas/"
ann_train="annotations/example_30/train.json"
ann_val="annotations/example_30/val.json"



### TRAINING
# Single-GPU training
python tools/train.py ${CONFIG_FILE} --work-dir=${WORK_DIR} \
--cfg-options train_dataloader.dataset.ann_file=${ann_train} \
val_dataloader.dataset.ann_file=${ann_val} \
val_evaluator.ann_file="${data_root}${ann_val}"


#### TEST
# Faster R-CNN
python tools/test.py ${CONFIG_FILE} ${WORK_DIR}/epoch_24.pth --work-dir=${WORK_DIR} --out=${WORK_DIR}/preds.pkl --show-dir=${WORK_DIR} \
--cfg-options test_dataloader.dataset.ann_file="annotations/${ann_val}" test_evaluator.ann_file="${data_root}${ann_val}" #model.test_cfg.rcnn.score_thr=0.5

# YOLO
python tools/test.py ${CONFIG_FILE} ${WORK_DIR}/epoch_273.pth --work-dir=${WORK_DIR} --out=${WORK_DIR}/preds.pkl --show-dir=${WORK_DIR} #--cfg-options model.test_cfg.rcnn.score_thr=0.5



# python tools/analysis_tools/confusion_matrix.py ${CONFIG_FILE}  ${WORK_DIR}/preds.pkl  ${WORK_DIR} --show --score-thr=0.5 --tp-iou-thr=0.5

# python tools/analysis_tools/optimize_anchors.py ${CONFIG_FILE} --algorithm differential_evolution --input-shape 200 200 --device cuda --output-dir ${WORK_DIR}

# Multi-GPU training
# GPU_NUM=2
# bash ./tools/dist_train.sh \
#     ${CONFIG_FILE} \
#     ${GPU_NUM} \
#     --work-dir=${WORK_DIR}

python tools/analysis_tools/analyze_results.py configs/mamoas/faster_rcnn.py results_loo_30/faster_rcnn/preds.pkl  results_loo_30/faster_rcnn/analyze --show-score-thr=0.5

python tools/analysis_tools/confusion_matrix.py configs/mamoas/faster_rcnn.py results_loo_30/faster_rcnn/preds.pkl  results_loo_30/faster_rcnn/ --score-thr=0.5 --tp-iou-thr=0.5 --color-theme=Blues

python tools/analysis_tools/eval_metric.py configs/mamoas/faster_rcnn.py results_loo_30/faster_rcnn/preds.pkl 