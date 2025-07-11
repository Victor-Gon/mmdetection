import os
import json
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utilities.dataset_utils import Coco7ChannelClaheDataset, CocoRGBDataset
from utilities_mmdet.utilities.dataset.dataset_utils import split_coco_train_val_sup_unsup


# ——— CONFIG ———————————————————————————————————————————————
SUPERVISED_PERCENTAGE   = 10   # % of *train* images to label
VAL_PERCENTAGE          = 20   # % of *all* images as val
TRAIN_IMG_DIR = "/mnt/hd/waymococo_f0/train2020"
TRAIN_ANN               = "/mnt/hd/waymococo_f0/annotations/instances_train2020.json"
OUTPUT_DIR              = "utilities_mmdet/annotations"
# file‐paths for the splits
VAL_SPLIT_JSON          = os.path.join(OUTPUT_DIR, "instances_val2020_split.json")
SUP_JSON                = os.path.join(OUTPUT_DIR, f"supervised_{SUPERVISED_PERCENTAGE}.json")
UNSUP_JSON              = os.path.join(OUTPUT_DIR, f"unsupervised_{100-SUPERVISED_PERCENTAGE}.json")


# STATS_JSON = os.path.join(OUTPUT_DIR, "dataset_stats_rgb.json")
STATS_JSON_CLAHE = os.path.join(OUTPUT_DIR, "dataset_stats_7channel_clahe.json")

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_mean_std(dataset, batch_size=32, num_workers=4):
    # A collate_fn that simply returns the list of samples,
    # so you can iterate one image at a time internally.
    def no_stack_collate(batch):
        return batch  # a list of (img_tensor, target) tuples

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=no_stack_collate,
        pin_memory=True,
    )

    # Figure out how many channels from the first sample
    first_img, _ = dataset[0]
    n_channels = first_img.shape[0]

    mean = np.zeros(n_channels, dtype=np.float64)
    var  = np.zeros(n_channels, dtype=np.float64)
    n_imgs = 0

    print("\nComputing mean and std over training set…")
    for batch in tqdm(loader):
        # batch is now a list of (img, _) pairs
        for img, _ in batch:
            img_np = img.numpy()          # (C, H, W)
            C, H, W = img_np.shape
            pixels = img_np.reshape(C, -1)  # (C, H*W)

            mean += pixels.mean(axis=1)
            var  += pixels.var(axis=1)
            n_imgs += 1

    mean /= n_imgs
    std   = np.sqrt(var / n_imgs)
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    print("Splitting annotations...")
    val_ann, sup_ann, unsup_ann = split_coco_train_val_sup_unsup(
        "/mnt/hd/waymococo_f0/annotations/instances_train2020.json",
        percent_val=VAL_PERCENTAGE/100,
        percent_labeled=SUPERVISED_PERCENTAGE/100,
        seed=42
    )

    # then save them:
    with open(VAL_SPLIT_JSON, "w") as f:
        json.dump(val_ann,   f, indent=2)
    with open(SUP_JSON, "w") as f:
        json.dump(sup_ann,   f, indent=2)
    with open(UNSUP_JSON, "w") as f:
        json.dump(unsup_ann, f, indent=2)

    # print("Building dataset for stats...")
    # # full_dataset = CocoRGBDataset(
    # full_dataset = Coco7ChannelClaheDataset(
    #     img_dir=TRAIN_IMG_DIR,
    #     annotations_dict={img['id']: {
    #         "file_name": img["file_name"],
    #         "boxes": [],
    #         "labels": [],
    #         "time_of_day": img.get("time_of_day", "na"),
    #         "weather": img.get("weather", "na")
    #     } for img in labeled_dict["images"] + unlabeled_dict["images"]},
    #     skip_additional_channels=False
    # )

    # mean, std = compute_mean_std(full_dataset)

    # print(f"\n→ Mean: {mean}")
    # print(f"→ Std:  {std}")

    # print(f"\nSaving to {STATS_JSON_CLAHE}...")
    # with open(STATS_JSON_CLAHE, 'w') as f:
    #     json.dump({"mean": mean, "std": std}, f, indent=2)

    # print("\nDone.")