import json
import random
from copy import deepcopy
from collections import defaultdict, Counter

def split_coco_train_val_sup_unsup(
    coco_json_path,
    percent_val: float = 0.2,
    percent_labeled: float = 0.1,
    seed: int = 42
):
    """
    Splits a COCO annotation file into:
      — VAL set (with full annotations)
      — TRAIN–SUP set (with annotations, size = (1 - percent_val) * percent_labeled)
      — TRAIN–UNSUP set (no annotations)

    Args:
        coco_json_path (str): Path to the original COCO JSON.
        percent_val (float): Fraction of annotated images to reserve for validation.
        percent_labeled (float): Fraction *of the remaining* train images to label.
        seed (int): Random seed for reproducibility.

    Returns:
        val_dict (dict): COCO-format JSON for validation.
        sup_dict (dict): COCO-format JSON for supervised training.
        unsup_dict (dict): COCO-format JSON for unsupervised training.
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        full = json.load(f)

    images      = full['images']
    annotations = full['annotations']
    categories  = full['categories']

    # group annotations by image_id
    ann_by_img = defaultdict(list)
    for ann in annotations:
        ann_by_img[ann['image_id']].append(ann)

    # only keep images that actually have annotations
    valid_imgs = [img for img in images if len(ann_by_img[img['id']]) > 0]

    random.seed(seed)
    ids = [img['id'] for img in valid_imgs]
    random.shuffle(ids)

    # 1) split out VAL
    n_val      = int(len(ids) * percent_val)
    val_ids    = set(ids[:n_val])
    train_ids0 = ids[n_val:]  # the rest are for training

    # 2) within TRAIN, pick SUP
    n_sup      = int(len(train_ids0) * percent_labeled)
    sup_ids    = set(random.sample(train_ids0, n_sup))
    unsup_ids  = set(train_ids0) - sup_ids

    # helper: filter down to images + annotations
    def make_subset(img_ids, keep_anns: bool):
        imgs = [img for img in valid_imgs if img['id'] in img_ids]
        if keep_anns:
            anns = [ann for ann in annotations if ann['image_id'] in img_ids]
        else:
            anns = []
        return {
            "images": deepcopy(imgs),
            "annotations": deepcopy(anns),
            "categories": deepcopy(categories)
        }

    val_dict   = make_subset(val_ids,   keep_anns=True)
    sup_dict   = make_subset(sup_ids,   keep_anns=True)
    unsup_dict = make_subset(unsup_ids, keep_anns=False)

    # print basic stats
    def count_classes(anns):
        ctr = Counter([a['category_id'] for a in anns])
        return ctr

    print(f"\n🔖 VAL set:   {len(val_dict['images'])} images, {len(val_dict['annotations'])} boxes")
    sup_ctr = count_classes(sup_dict['annotations'])
    print(f"🎯 SUP set:   {len(sup_dict['images'])} images, {sum(sup_ctr.values())} boxes")
    unsup_ctr = count_classes(unsup_dict['annotations'])
    print(f"🤫 UNSUP set: {len(unsup_dict['images'])} images, {sum(unsup_ctr.values())} boxes (should be 0)\n")

    return val_dict, sup_dict, unsup_dict