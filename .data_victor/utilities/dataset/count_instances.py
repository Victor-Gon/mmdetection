import json
from collections import defaultdict

# Path to your COCO annotations file
# COCO_JSON = "/mnt/hd/waymococo_f0/semi_anns/instances_train2020.1@1-unlabeled.json"
COCO_JSON = "/mnt/hd/waymococo_f0/semi_anns/instances_train2020.1@1.json"

with open(COCO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

# Group annotations by image_id
ann_by_img = defaultdict(list)
for ann in annotations:
    ann_by_img[ann["image_id"]].append(ann)

# Filter only images that have at least one annotation
annotated_images = [img for img in images if len(ann_by_img[img["id"]]) > 0]

print(f"📊 Total images: {len(images)}")
print(f"🖊  Images with annotations: {len(annotated_images)}")
print(f"🚫 Images without annotations: {len(images) - len(annotated_images)}")