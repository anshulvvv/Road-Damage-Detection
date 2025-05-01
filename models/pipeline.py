# %%
!mkdir -p ~/datasets && cd ~/datasets
!wget -c --trust-server-names \
  https://figshare.com/ndownloader/articles/21431547/versions/1 \
  -O RDD2022.zip
!unzip -q RDD2022.zip && rm RDD2022.zip


# %%
!mkdir -p RDD2022
!unzip -q RDD2022_released_through_CRDDC2022.zip -d RDD2022


# %%
import os
import zipfile
from pathlib import Path

os.chdir('/home/ansul/RDD2022/RDD2022')

for z in Path('.').glob('*.zip'):
    target = z.stem               # e.g. 'India' from 'India.zip'
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(z, 'r') as zip_ref:
        zip_ref.extractall(target)
    print(f"Extracted {z.name} → {target}/")

# %%
!find /home/ansul/RDD2022/RDD2022 -type f | wc -l


# %%
!df -i /home/ansul/RDD2022/RDD2022

# %%
import random
from pathlib import Path

# ─── ADJUST THESE ────────────────────────────────────────────────────────────
SRC_ROOT = Path("/home/ansul/RDD2022/RDD2022")
DST_ROOT = Path("/home/ansul/final_dataset")
SEED     = 42
# ─────────────────────────────────────────────────────────────────────────────

# how many images per split, by “region”
SPLITS = {
    "China":          {"train": 3494, "val": 884,  "test": 500},
    "Czech":          {"train": 2269, "val": 560,  "test": 709},
    "India":          {"train": 6145, "val": 1561, "test": 1959},
    "Japan":          {"train": 8412, "val": 2094, "test": 2627},
    "Norway":         {"train": 6526, "val": 1635, "test": 2040},
    "United States": {"train": 3862, "val": 943,  "test": 1200},
}

# map each region → list of its source subfolders
CLASS_DIRS = {
    "China":          ["China_Drone", "China_MotorBike"],
    "Czech":          ["Czech"],
    "India":          ["India"],
    "Japan":          ["Japan"],
    "Norway":         ["Norway"],
    "United States": ["United_States"],
}

random.seed(SEED)

# 1) make output dirs
for split in ("train", "val", "test"):
    (DST_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    if split != "test":
        (DST_ROOT / split / "annotations").mkdir(exist_ok=True)

for region, counts in SPLITS.items():
    # 2) gather all train images + annotations for this region
    all_imgs = []
    all_xmls = {}
    for cls in CLASS_DIRS[region]:
        # note the extra subfolder layer: <cls>/<cls>/
        base = SRC_ROOT / cls / cls
        tr_img_dir = base / "train" / "images"
        tr_ann_dir = base / "train" / "annotations" / "xmls"

        # collect images
        for img in tr_img_dir.glob("*.*"):
            all_imgs.append(img)
        # collect xmls keyed by basename
        for xml in tr_ann_dir.glob("*.xml"):
            all_xmls[xml.stem] = xml

    total_needed = counts["train"] + counts["val"]
    if len(all_imgs) < total_needed:
        raise RuntimeError(f"{region}: only {len(all_imgs)} train images, need {total_needed}")

    # 3) shuffle & split
    random.shuffle(all_imgs)
    val_imgs   = all_imgs[: counts["val"]]
    train_imgs = all_imgs[counts["val"] : counts["val"] + counts["train"]]

    # 4) symlink train + val
    for subset, img_list in (("train", train_imgs), ("val", val_imgs)):
        img_out_dir  = DST_ROOT / subset / "images"
        ann_out_dir  = DST_ROOT / subset / "annotations"
        for img in img_list:
            dst_img = img_out_dir / img.name
            if not dst_img.exists():
                dst_img.symlink_to(img.resolve())
            # matching xml?
            xml = all_xmls.get(img.stem)
            if xml:
                dst_xml = ann_out_dir / xml.name
                if not dst_xml.exists():
                    dst_xml.symlink_to(xml.resolve())

    # 5) symlink test images
    test_out = DST_ROOT / "test" / "images"
    test_imgs = []
    for cls in CLASS_DIRS[region]:
        base = SRC_ROOT / cls / cls
        td = base / "test" / "images"
        if td.exists():
            test_imgs += list(td.glob("*.*"))

    if len(test_imgs) < counts["test"]:
        raise RuntimeError(f"{region}: only {len(test_imgs)} test images, need {counts['test']}")

    for img in test_imgs[: counts["test"]]:
        dst = test_out / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())

print("Done! Your split lives in", DST_ROOT)


# %%
import os
from pathlib import Path

def count_files(root: Path) -> int:
    """
    Walks through all subdirs of `root` and counts every entry in `files`.
    This will include regular files and symlinks to files.
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        total += len(filenames)
    return total

def breakdown(root: Path):
    splits = ["train", "val", "test"]
    grand_total = 0

    for split in splits:
        print(f"=== {split} ===")
        img_dir = root / split / "images"
        img_count = count_files(img_dir)
        print(f" images:      {img_count}")
        grand_total += img_count

        if split != "test":
            ann_dir = root / split / "annotations"
            ann_count = count_files(ann_dir)
            print(f" annotations: {ann_count}")
            grand_total += ann_count

    print(f"\nTotal files (including symlinks) in `{root}`: {grand_total}")


ROOT = Path("/home/ansul/final_dataset")  # adjust if your folder is elsewhere
breakdown(ROOT)


# %%
import os
import random
import shutil

# 1) Paths
BASE      = "/home/ansul/final_dataset"
train_img = os.path.join(BASE, "train", "images")
train_ann = os.path.join(BASE, "train", "annotations")
eval_img  = os.path.join(BASE, "eval",  "images")
eval_ann  = os.path.join(BASE, "eval",  "annotations")

# 2) Make eval dirs
os.makedirs(eval_img, exist_ok=True)
os.makedirs(eval_ann, exist_ok=True)

# 3) Specify exact sample size
num_to_sample = 7677
print(f"Sampling {num_to_sample} images from train/ → eval/")

# 4) Gather all train images
all_imgs = [
    f for f in os.listdir(train_img)
    if f.lower().endswith(('.jpg', '.png'))
]

# 5) Sample without replacement
random.seed(42)  # for reproducibility
selection = random.sample(all_imgs, min(num_to_sample, len(all_imgs)))

# 6) Copy images + annotations
for img_name in selection:
    # copy image
    shutil.copy(
        os.path.join(train_img, img_name),
        os.path.join(eval_img,  img_name)
    )

    # copy annotation (adjust extension if needed)
    stem, _   = os.path.splitext(img_name)
    ann_file  = stem + ".xml"
    src_ann   = os.path.join(train_ann, ann_file)
    dst_ann   = os.path.join(eval_ann,  ann_file)

    if os.path.exists(src_ann):
        shutil.copy(src_ann, dst_ann)
    else:
        print(f"[WARN] missing annotation for {img_name}")

print("Eval split ready at final_dataset/eval/")


# %% [markdown]
# Stage 1A  : Training All three models on training data

# %%
"""
Making the Directory structure for each model
for YOLO
  base
   - labels(in yolo format)
       -train
       -val
   - images
       -train
       -val  
for RTM DET and DETR
   -base
       -train
          -images
          -annotations.json
       -val   
          -images     
          -annotations.json
assume that the images are already downloaded and the labels are in xml format
the directory structure is as follows
base
   - train
      - images
      - annotations
   - test
      - images 
in each of the above subdirectories, there are images in the folder and a sub_folder called annotations
except for the test folder, the annotations folder contains the xml files for the images in the folder"
"""   

# %%
import sys
sys.path.append(r'/home/ansul/AI_Road_Health_Assessment/scripts')

# %%
import importlib
import directory_maker_train
importlib.reload(directory_maker_train)
from directory_maker_train import main as dmt

# %%
import importlib
import directory_struct_RDD
importlib.reload(directory_struct_RDD)
from directory_struct_RDD import main as dmt

# %%
dmt("/home/ansul/final_dataset")

# %%
%run directory_maker_train.py --base_dir "/home/ansul/dummy_data/India"

# %%
"""

Train YOLOv10

"""

# %%
%run yolo.py --data_config "/home/ansul/AI_Road_Health_Assessment/config/yolov10_config.yaml" --epochs 300 --batch_size 40 --new True

# %%
# Resume Abrupted Training
from ultralytics import YOLO
model = YOLO("runs/detect/train5/weights/last.pt")
model.train(data="/home/ansul/AI_Road_Health_Assessment/config/yolov10_config.yaml", epochs=300, batch=40, resume=True)

# model = YOLO("jameshalm/yolov10n.pt")    # or whatever .pt you began with
# model.train(
#     data="/home/ansul/AI_Road_Health_Assessment/config/yolov10_config.yaml",
#     epochs=300,               # final epoch you want
#     batch=40,
#     resume="/home/ansul/AI_Road_Health_Assessment/models/runs/detect/train5/weights/last.pt"               # this will load runs/train/exp/weights/last.pt
# )

# %%
"""

Train RTM DET

"""

# %%
from rtmdettrain import run_training_and_inference as rtm_train
ckpnt_rtm_1,config_path = rtm_train("/home/ansul/dummy_data/India/rtmdetr",32,20)

# %%
"""

Train DETR

"""

# %%
import accelerate

# %%
import importlib
import detr_train
importlib.reload(detr_train)

# %%
import pytorch_lightning

# %%
print(pytorch_lightning.__version__)

# %%
from detr_train import run_training as dtrain
dtrain("/home/ansul/dummy_data/India/rtmdetr" , 20 ,32 ,1e-4 , 1e-5 , 1e-4)

# %% [markdown]
# Stage 1B : Producing Pseudo Labels from DETR and RTM_DET Models on Train Set 

# %%
"""

RTM_DET Inference


"""

import os
import json
from mmdet.apis import init_detector, inference_detector
import torch
def pseudo_label_inference(model, test_dir, output_json):
    pseudo_labels = {}
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        result = inference_detector(model, img_path)
        boxes = []
        for class_idx, class_boxes in enumerate(result):
            for box in class_boxes:
                score = box[-1]
                if score >= 0.3 :  
                    boxes.append({
                        "label": class_idx,
                        "bbox": box[:4].tolist(),
                        "score": float(score)
                    })
        pseudo_labels[img_file] = boxes
    with open(output_json, "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    print("Pseudo labels saved to", output_json)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_detector(config_path, ckpnt_rtm_1, device=device)
pseudo_label_inference(model, "test_dir","test_dir/rtm_det_annotations.json" )

# %%
"""With NMS RTM_DETR"""
import os
import json
import numpy as np
import torch
import torchvision.ops as ops
from mmdet.apis import init_detector, inference_detector

def pseudo_label_inference(model, test_dir, output_json, score_thresh=0.3, nms_thresh=0.5):
    
    pseudo_labels = {}
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        result = inference_detector(model, img_path)
        boxes_filtered = []
        
        for class_idx, class_boxes in enumerate(result):
            if len(class_boxes) == 0:
                continue
            
            class_boxes = np.array(class_boxes)
            keep = class_boxes[:, 4] >= score_thresh
            class_boxes = class_boxes[keep]
            if len(class_boxes) == 0:
                continue
            
            boxes_tensor = torch.tensor(class_boxes[:, :4], dtype=torch.float32)
            scores_tensor = torch.tensor(class_boxes[:, 4], dtype=torch.float32)
            
            keep_indices = ops.nms(boxes_tensor, scores_tensor, nms_thresh)
            selected_boxes = class_boxes[keep_indices.numpy()]
            
            for box in selected_boxes:
                boxes_filtered.append({
                    "label": class_idx,
                    "bbox": box[:4].tolist(),
                    "score": float(box[4])
                })
        
        pseudo_labels[img_file] = boxes_filtered

    with open(output_json, "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    print("Pseudo labels saved to", output_json)



# %%
"""

DETR Inference


"""


import os
import json
import torch
from PIL import Image
from transformers import DetrFeatureExtractor,DetrForObjectDetection
from pytorch_lightning import LightningModule

class DetrLightning(LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_labels, id2label):
        super().__init__()
        self.save_hyperparameters()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.id2label = id2label 

def pseudo_label_inference_detr(checkpoint_path, test_dir, output_json, score_threshold=0.5):
    """Modified to load from checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pl_model = DetrLightning.load_from_checkpoint(
        checkpoint_path,
        lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, 
        num_labels=len(pl_model.hparams.id2label),  
        id2label=pl_model.hparams.id2label
    )
    model = pl_model.model.to(device).eval()
    
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    pseudo_labels = {}
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
        
        boxes = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            if score < score_threshold:
                continue
            boxes.append({
                "label": pl_model.id2label[int(label.item())],  
                "bbox": box.tolist(),
                "score": float(score.item())
            })
        pseudo_labels[img_file] = boxes
    
    with open(output_json, "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    print(f"Pseudo labels saved to {output_json} (Threshold: {score_threshold})")

pseudo_label_inference_detr(
    checkpoint_path="your_checkpoint.ckpt",
    test_dir="test_dir",
    output_json="test_dir/detr_annotations.json",
    score_threshold=0.5 
)



# %%
"""DETR NMS Inference"""
import os
import json
import torch
from PIL import Image
import torchvision.ops as ops
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from pytorch_lightning import LightningModule

class DetrLightning(LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_labels, id2label):
        super().__init__()
        self.save_hyperparameters()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.id2label = id2label 

def pseudo_label_inference_detr(checkpoint_path, test_dir, output_json, score_threshold=0.5, nms_threshold=0.5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pl_model = DetrLightning.load_from_checkpoint(
        checkpoint_path,
        lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
        num_labels= len(torch.load(checkpoint_path)["hyper_parameters"]["id2label"]),
        id2label= torch.load(checkpoint_path)["hyper_parameters"]["id2label"]
    )
    model = pl_model.model.to(device).eval() 
    
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    
    pseudo_labels = {}
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
        boxes_tensor = results["boxes"]
        scores_tensor = results["scores"]
        labels_tensor = results["labels"]
        valid = scores_tensor >= score_threshold
        boxes_tensor = boxes_tensor[valid]
        scores_tensor = scores_tensor[valid]
        labels_tensor = labels_tensor[valid]
        
        boxes_filtered = []
        if boxes_tensor.shape[0] > 0:
            keep = ops.batched_nms(boxes_tensor, scores_tensor, labels_tensor, nms_threshold)
            for idx in keep:
                boxes_filtered.append({
                    "label": pl_model.id2label[int(labels_tensor[idx].item())],
                    "bbox": boxes_tensor[idx].tolist(),
                    "score": float(scores_tensor[idx].item())
                })
        
        pseudo_labels[img_file] = boxes_filtered

    with open(output_json, "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    print(f"Pseudo labels saved to {output_json} (Threshold: {score_threshold}, NMS: {nms_threshold})")




# %% [markdown]
# <<<Stage 2 (Mutual Learning)>>>

# %%
# create a directory structure for the test images and ground truth images

# %%

# Activate the RTM environment and run RTM inference
!source /path/to/rtm-env/bin/activate
!python rtm_inference.py \
  --config_path path/to/your_config.py \
  --ckpnt path/to/your_checkpoint.pth \
  --test_dir path/to/test/images \
  --output_json rtm_pseudo_labels.json \
  --score_thresh 0.3 \
  --nms_thresh 0.5
!deactivate

# Activate the DETR environment and run DETR inference
!source /path/to/detr-env/bin/activate
!python detr_inference.py \
  --checkpoint_path "/home/ansul/AI_Road_Health_Assessment/models/detr_model.ckpt" \
  --test_dir path/to/test/images \
  --output_json detr_pseudo_labels.json \
  --score_threshold 0.5 \
  --nms_threshold 0.5
!deactivate


# %%
import os
from pathlib import Path
base_dir = Path("/home/ansul/dummy_data/India")
train_annotations = base_dir / "train" / "annotations" / "annotations_coco.json"
rtm_pseudo_labels = base_dir / "test" / "images" / "rtm_pseudo_labels.json"
if train_annotations.exists() and rtm_pseudo_labels.exists():
    print("hww")

# %%
import os
import json
import shutil
from pathlib import Path
import copy

def read_json_annotations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_annotations(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def merge_coco_annotations(train_annotations, test_annotations):
    
    merged = copy.deepcopy(train_annotations)
    
    max_img_id = max([img["id"] for img in merged["images"]]) if merged["images"] else 0
    max_ann_id = max([ann["id"] for ann in merged["annotations"]]) if merged["annotations"] else 0
    
    img_id_mapping = {}
    
    for img in test_annotations["images"]:
        img_copy = copy.deepcopy(img)
        old_img_id = img_copy["id"]
        new_img_id = max_img_id + 1
        img_id_mapping[old_img_id] = new_img_id
        
        img_copy["id"] = new_img_id
        merged["images"].append(img_copy)
        max_img_id += 1
    
    for ann in test_annotations["annotations"]:
        ann_copy = copy.deepcopy(ann)
        old_img_id = ann_copy["image_id"]
        
        if old_img_id in img_id_mapping:
            ann_copy["image_id"] = img_id_mapping[old_img_id]
            ann_copy["id"] = max_ann_id + 1
            merged["annotations"].append(ann_copy)
            max_ann_id += 1
    
    if "categories" in test_annotations:
        existing_cat_ids = {cat["id"] for cat in merged.get("categories", [])}
        for cat in test_annotations["categories"]:
            if cat["id"] not in existing_cat_ids:
                merged.setdefault("categories", []).append(cat)
    
    return merged

def create_symlinks(source_dir, target_dir):
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    
    for file_path in source_path.glob("*"):
        if not file_path.is_file():
            continue
        
        if file_path.suffix.lower() not in image_extensions:
            continue

        target_file = target_path / file_path.name
        if not target_file.exists():
            os.symlink(file_path.absolute(), target_file)
            count += 1
    
    return count

def setup_combined_dataset():
    
    base_dir = Path("/home/ansul/dummy_data/India")
    rtmdet_detr_dir = base_dir / "mutual"  
    
    train_images = base_dir / "train" / "images"
    test_images = base_dir / "test" / "images"
    train_annotations = base_dir / "train" / "annotations" / "annotations_coco.json"
    rtm_pseudo_labels = base_dir / "test" / "images" / "rtm_pseudo_labels.json"
    detr_pseudo_labels = base_dir / "test" / "images" / "detr_pseudo_labels.json"
    
    rtmdet_combined = rtmdet_detr_dir / "rtmdet_combined"
    rtmdet_combined.mkdir(exist_ok=True, parents=True)
    
    rtmdet_train_dir = rtmdet_combined / "train"
    rtmdet_train_dir.mkdir(exist_ok=True)
    
    rtmdet_train_images = rtmdet_train_dir / "images"
    rtmdet_train_images.mkdir(exist_ok=True)
    
    detr_combined = rtmdet_detr_dir / "detr_combined"
    detr_combined.mkdir(exist_ok=True, parents=True)
    
    detr_train_dir = detr_combined / "train"
    detr_train_dir.mkdir(exist_ok=True)
    
    detr_train_images = detr_train_dir / "images"
    detr_train_images.mkdir(exist_ok=True)
    
    print("Creating symlinks for train images in RTM-DET directory...")
    train_links = create_symlinks(train_images, rtmdet_train_images)
    print(f"Created {train_links} symlinks for train images")
    
    print("Creating symlinks for test images in RTM-DET directory...")
    test_links = create_symlinks(test_images, rtmdet_train_images)
    print(f"Created {test_links} symlinks for test images")
    
    print("Creating symlinks for train images in DETR directory...")
    create_symlinks(train_images, detr_train_images)
    
    print("Creating symlinks for test images in DETR directory...")
    create_symlinks(test_images, detr_train_images)
    
    if (base_dir / "rtmdetr" / "val").exists():
        print("Setting up validation directory...")
        
        rtmdet_val_dir = rtmdet_combined / "val"
        rtmdet_val_dir.mkdir(exist_ok=True)
        
        rtmdet_val_images = rtmdet_val_dir / "images"
        rtmdet_val_images.mkdir(exist_ok=True)
        
        val_images_src = base_dir / "rtmdetr" / "val" / "images"
        create_symlinks(val_images_src, rtmdet_val_images)
        
        val_ann_src = base_dir / "rtmdetr" / "val" / "images"/ "annotations_coco.json"
        if val_ann_src.exists():
            shutil.copy2(val_ann_src, rtmdet_val_images / "annotations_coco.json")
        
        detr_val_dir = detr_combined / "val"
        detr_val_dir.mkdir(exist_ok=True)
        
        detr_val_images = detr_val_dir / "images"
        detr_val_images.mkdir(exist_ok=True)
        
        create_symlinks(val_images_src, detr_val_images)
        
        if val_ann_src.exists():
            shutil.copy2(val_ann_src, detr_val_images / "annotations_coco.json")
    
    if train_annotations.exists() and detr_pseudo_labels.exists():
        print("Merging annotations for RTM-DET...")
        train_ann_data = read_json_annotations(train_annotations)
        rtm_pseudo_data = read_json_annotations(detr_pseudo_labels)
        
        rtm_merged = merge_coco_annotations(train_ann_data, rtm_pseudo_data)
        save_json_annotations(rtm_merged, rtmdet_train_images / "annotations_coco.json")
        print(f"Saved merged annotations to {rtmdet_train_images / 'annotations_coco.json'}")
    
    if train_annotations.exists() and rtm_pseudo_labels.exists():
        print("Merging annotations for DETR...")
        train_ann_data = read_json_annotations(train_annotations)
        detr_pseudo_data = read_json_annotations(rtm_pseudo_labels)
        
        detr_merged = merge_coco_annotations(train_ann_data, detr_pseudo_data)
        save_json_annotations(detr_merged, detr_train_images / "annotations_coco.json")
        print(f"Saved merged annotations to {detr_train_images/ 'annotations_coco.json'}")
    
    return rtmdet_combined, detr_combined

def update_with_new_pseudo_labels(base_dir, combined_dir, new_pseudo_labels_path):
    
    base_path = Path(base_dir)
    combined_path = Path(combined_dir)
    
    train_ann_path = base_path / "train" / "annotations"/ "annotations_coco.json"
    train_annotations = read_json_annotations(train_ann_path)
    new_pseudo_labels = read_json_annotations(new_pseudo_labels_path)
    merged_annotations = merge_coco_annotations(train_annotations, new_pseudo_labels)
    updated_ann_path = combined_path / "train" / "images" / "annotations_coco.json"
    save_json_annotations(merged_annotations, updated_ann_path)
    
    print(f"Updated annotations saved to {updated_ann_path}")
    
    return updated_ann_path
    


# %%
import subprocess

venv1_python = "/home/ansul/rtm-env/bin/python" 
venv2_python = ".local/bin"


# %%
def rtm_infer(thresh,nmsthresh,ckpnt):
    command = [
        "/home/ansul/rtm-env/bin/python",
        "/home/ansul/AI_Road_Health_Assessment/models/rtm_inference.py",
        "--config_path", "/home/ansul/AI_Road_Health_Assessment/models/mmyolo/configs/rtmdet/custom.py",
        "--ckpnt", ckpnt,
        "--test_dir", "/home/ansul/dummy_data/India/test/images",
        "--output_json", "/home/ansul/dummy_data/India/test/images/rtm_pseudo_labels.json",
        "--score_thresh", str(thresh),
        "--nms_thresh", str(nmsthresh),
    ]
    # for arg in command:
    #     print(arg, type(arg))
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print("Inference by RTM done.")

# %%

def detr_infer(thresh,nmsthresh,ckpnt):
    command = [
        "python3",  
        "/home/ansul/AI_Road_Health_Assessment/models/detr_inference.py",  
        "--checkpoint_path", ckpnt,
        "--test_dir", "/home/ansul/dummy_data/India/test/images",  
        "--output_json", "/home/ansul/dummy_data/India/test/images/detr_pseudo_labels.json",
        "--score_threshold", str(thresh),
        "--nms_threshold", str(nmsthresh),
    ]
    # for arg in command:
    #     print(arg, type(arg))
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print("Inference by DETR done.")


# %%
import subprocess
import json
def rtm_train_wrapper(root,ckpnt_dir, epochs, batch_size,new):
    command = [
        "/home/ansul/rtm-env/bin/python",
        "/home/ansul/AI_Road_Health_Assessment/models/rtmdettrain.py",
        "--data_root" , str(root),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--new", str(new),
        "--resume_checkpoint", ckpnt_dir
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    with open("output_results.json", "r") as f:
        results = json.load(f)
    print("RTM Training completed.")
    return results["checkpoint_path"]


# %%
path_str = str(rtmdet_dir)
path_str += "/"
ckpnt_rtm_1 = rtm_train_wrapper(path_str, "/home/ansul/AI_Road_Health_Assessment/models/work_dirs/custom/epoch_20.pth",2, 8 , False)

# %%
import torch

# %%
import os

symlink_path = "/home/ansul/dummy_data/India/mutual/rtmdet_combined/train/images/annotations_coco.json"
target = os.readlink(symlink_path)
print(f"The symlink points to: {target}")


# %%
checkpoint = torch.load("/home/ansul/AI_Road_Health_Assessment/models/detr_model.ckpt", map_location="cpu")
print(checkpoint.keys())


# %%
rtmdet_dir, detr_dir = setup_combined_dataset()
print(f"\nRTM-DET combined dataset created at: {rtmdet_dir}")
print(f"DETR combined dataset created at: {detr_dir}")

# %%
# def mutual_learning(iter):
#     """Run the mutual learning process for multiple iterations"""
#     print(f"Starting mutual learning with {iter} iterations")
#     """
#     -base
#         - test   
#             - images 
#             - rtm_pseudo_labels.json
#             - detr_pseudo_labels.json
#     """
#     base_dir = Path("/home/ansul/dummy_data/India")
#     rtmdet_dir, detr_dir = setup_combined_dataset()
#     detr_infer(0.2,0.2,"/home/ansul/AI_Road_Health_Assessment/models/detr_model_first.ckpt")
#     rtm_infer(0.3,0.5,"/home/ansul/AI_Road_Health_Assessment/models/work_dirs/custom/epoch_20.pth")
#     print(f"\nRTM-DET combined dataset created at: {rtmdet_dir}")
#     print(f"DETR combined dataset created at: {detr_dir}")

#     for i in range(iter):
#         print(f"\nIteration {i+1}")
#         #Train models on train+test data
#         # Train RTM-DET first
#         ckpnt_rtm_1, config_path = rtm_train(str(rtmdet_dir), 8, 2)
#         dtrain(str(detr_dir), 2, 32, 1e-4, 1e-5, 1e-4)

#         detr_infer(0.2,0.2,"/home/ansul/AI_Road_Health_Assessment/models/detr_model_first.ckpt")
#         rtm_infer(0.3,0.5,ckpnt_rtm_1)

#         update_with_new_pseudo_labels(base_dir, str(rtmdet_dir), "/home/ansul/dummy_data/India/test/images/detr_pseudo_labels.json")
#         update_with_new_pseudo_labels(base_dir, str(detr_dir), "/home/ansul/dummy_data/India/test/images/rtm_pseudo_labels.json")

        
        

# %%
rtmdet_dir, detr_dir = setup_combined_dataset()
detr_infer(0.2, 0.2, "/home/ansul/AI_Road_Health_Assessment/models/detr_model_first.ckpt")
rtm_infer(0.3, 0.5, "/home/ansul/AI_Road_Health_Assessment/models/work_dirs/custom/epoch_20.pth")

# %%
import shutil
from pathlib import Path
import subprocess
import importlib
import detr_train
importlib.reload(detr_train)
from detr_train import run_training as dtrain

def mutual_learning(iterations):
    base_dir = Path("/home/ansul/dummy_data/India")
    mutual_folder = base_dir / "mutual"
    ckpnt_rtm_1 = "/home/ansul/AI_Road_Health_Assessment/models/work_dirs/custom/epoch_20.pth"
    try:
        print(f"Starting mutual learning with {iterations} iterations")
        rtmdet_dir, detr_dir = setup_combined_dataset()
        detr_infer(0.2, 0.2, "/home/ansul/AI_Road_Health_Assessment/models/detr_model_first.ckpt")
        rtm_infer(0.3, 0.5, "/home/ansul/AI_Road_Health_Assessment/models/work_dirs/custom/epoch_20.pth")
        print(f"\nRTM-DET combined dataset created at: {rtmdet_dir}")
        print(f"DETR combined dataset created at: {detr_dir}")

        for i in range(iterations):
            print(f"\nIteration {i+1}")
            path_str = str(rtmdet_dir)
            path_str += "/"
            ckpnt_rtm_1 = rtm_train_wrapper(path_str,ckpnt_rtm_1,2, 8 , False)
            dtrain(str(detr_dir), 2, 32, 1e-4, 1e-5, 1e-4 , "/home/ansul/AI_Road_Health_Assessment/models/detr_model_first.ckpt")
            detr_infer(0.2, 0.2, "/home/ansul/AI_Road_Health_Assessment/models/detr_model_first.ckpt")
            rtm_infer(0.3, 0.5, ckpnt_rtm_1)
            update_with_new_pseudo_labels(base_dir, str(rtmdet_dir), "/home/ansul/dummy_data/India/test/images/detr_pseudo_labels.json")
            update_with_new_pseudo_labels(base_dir, str(detr_dir), "/home/ansul/dummy_data/India/test/images/rtm_pseudo_labels.json")
    except Exception as e:
        print(f"An error occurred: {e}")
        if mutual_folder.exists():
            try:
                shutil.rmtree(mutual_folder)
                print(f"Deleted mutual folder at {mutual_folder} due to error.")
            except Exception as delete_error:
                print(f"Error deleting mutual folder: {delete_error}")
        raise

    print("Mutual learning process completed successfully.")


# %%
import importlib
import detr_train
importlib.reload(detr_train)
from detr_train import run_training as dtrain

# %%
dtrain("/home/ansul/dummy_data/India/mutual/detr_combined", 2, 32, 1e-4, 1e-5, 1e-4 , "/home/ansul/AI_Road_Health_Assessment/models/detr_model_first.ckpt")

# %%
base_dir = Path("/home/ansul/dummy_data/India")
update_with_new_pseudo_labels(base_dir, str(rtmdet_dir), "/home/ansul/dummy_data/India/test/images/detr_pseudo_labels.json")

# %%
mutual_learning(2)

# %% [markdown]
# <<<Stage 3 (Knowledge Distillation)>>>

# %%
# import json
# import torch
# import torchvision.ops as ops

# def bbox_coco_to_xyxy(bbox):
#     x, y, w, h = bbox
#     return [x, y, x + w, y + h]

# def bbox_xyxy_to_coco(xyxy):
#     x1, y1, x2, y2 = xyxy
#     return [x1, y1, x2 - x1, y2 - y1]

# def merge_teacher_coco_annotations(coco1_json, coco2_json, output_json, iou_threshold=0.5):
#     with open(coco1_json, "r") as f:
#         coco1 = json.load(f)
#     with open(coco2_json, "r") as f:
#         coco2 = json.load(f)
        
#     merged_images = coco1["images"]
#     merged_categories = coco1["categories"]
    
#     annotations_by_image = {}
#     for ann in coco1["annotations"] + coco2["annotations"]:
#         image_id = ann["image_id"]
#         annotations_by_image.setdefault(image_id, []).append(ann)
    
#     merged_annotations = []
#     ann_id = 1
    
#     for image in merged_images:
#         image_id = image["id"]
#         anns = annotations_by_image.get(image_id, [])
#         if not anns:
#             continue
        
#         boxes = []
#         scores = []
#         category_ids = []
#         original_anns = []
#         for ann in anns:
#             xyxy = bbox_coco_to_xyxy(ann["bbox"])
#             score = float(ann.get("score", 1.0))
#             boxes.append(xyxy)
#             scores.append(score)
#             category_ids.append(ann["category_id"])
#             original_anns.append(ann)

#         if not boxes:
#             continue

#         boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
#         scores_tensor = torch.tensor(scores, dtype=torch.float32)
#         labels_tensor = torch.tensor(category_ids, dtype=torch.int64)
        
#         keep_indices = ops.batched_nms(boxes_tensor, scores_tensor, labels_tensor, iou_threshold)
        
#         for idx in keep_indices:
#             idx = int(idx.item())
#             xyxy = boxes_tensor[idx].tolist()
#             coco_bbox = bbox_xyxy_to_coco(xyxy)
#             score = float(scores_tensor[idx].item())
#             category_id = int(labels_tensor[idx].item())
#             area = coco_bbox[2] * coco_bbox[3]
            
#             new_ann = {
#                 "id": ann_id,
#                 "image_id": image_id,
#                 "category_id": category_id,
#                 "bbox": coco_bbox,
#                 "area": area,
#                 "iscrowd": 0,
#                 "score": score
#             }
#             merged_annotations.append(new_ann)
#             ann_id += 1

#     merged_coco = {
#         "images": merged_images,
#         "annotations": merged_annotations,
#         "categories": merged_categories
#     }

#     with open(output_json, "w") as f:
#         json.dump(merged_coco, f, indent=2)
#     print(f"[INFO] Merged COCO annotations saved to: {output_json}")



# %%
import json
import torch
import torchvision.ops as ops

def bbox_coco_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def bbox_xyxy_to_coco(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]

def merge_teacher_coco_annotations(coco1_json, coco2_json, output_json, iou_threshold=0.5):
    with open(coco1_json, "r") as f:
        coco1 = json.load(f)
    with open(coco2_json, "r") as f:
        coco2 = json.load(f)
        
    merged_images = coco1["images"]
    merged_categories = coco1["categories"]
    
    annotations_by_image = {}
    for ann in coco1["annotations"] + coco2["annotations"]:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)
    
    merged_annotations = []
    ann_id = 1
    
    for image in merged_images:
        image_id = image["id"]
        anns = annotations_by_image.get(image_id, [])
        if not anns:
            continue
        
        boxes = []
        scores = []
        category_ids = []
        original_anns = []
        for ann in anns:
            xyxy = bbox_coco_to_xyxy(ann["bbox"])
            score = float(ann.get("score", 1.0))
            boxes.append(xyxy)
            scores.append(score)
            category_ids.append(ann["category_id"])
            original_anns.append(ann)

        if not boxes:
            continue

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        labels_tensor = torch.tensor(category_ids, dtype=torch.int64)
        
        keep_indices = ops.batched_nms(boxes_tensor, scores_tensor, labels_tensor, iou_threshold)
        
        for idx in keep_indices:
            idx = int(idx.item())
            xyxy = boxes_tensor[idx].tolist()
            coco_bbox = bbox_xyxy_to_coco(xyxy)
            score = float(scores_tensor[idx].item())
            category_id = int(labels_tensor[idx].item())
            area = coco_bbox[2] * coco_bbox[3]
            
            new_ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "score": score
            }
            merged_annotations.append(new_ann)
            ann_id += 1

    merged_coco = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories
    }

    with open(output_json, "w") as f:
        json.dump(merged_coco, f, indent=2)
    print(f"[INFO] Merged COCO annotations saved to: {output_json}")



# %%
# import os
# import json
# from PIL import Image

# def convert_json_to_yolo(json_path, images_dir, output_dir, class_names):

#     os.makedirs(output_dir, exist_ok=True)

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     for image_name, boxes in data.items():
#         # Derive the image file path
#         image_path = os.path.join(images_dir, image_name)
#         if not os.path.exists(image_path):
#             # Skip if the image file doesn't exist
#             continue
        
#         # Open image to get dimensions
#         with Image.open(image_path) as img:
#             w, h = img.size
        
#         # Build lines for the YOLO annotation file
#         lines = []
#         for box_info in boxes:
#             x1, y1, x2, y2 = box_info["bbox"]
#             label_str = box_info["label"]  # "U00", "D10", etc.

#             # Convert string label to numeric index
#             label_idx = class_names.index(label_str)

#             # Convert xyxy -> YOLO center/width/height
#             cx = (x1 + x2) / 2.0
#             cy = (y1 + y2) / 2.0
#             bw = x2 - x1
#             bh = y2 - y1

#             # Normalize
#             cx /= w
#             cy /= h
#             bw /= w
#             bh /= h

#             # YOLO format: class x_center y_center width height
#             lines.append(f"{label_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

#         # Write to .txt file
#         txt_filename = os.path.splitext(image_name)[0] + ".txt"
#         txt_path = os.path.join(output_dir, txt_filename)

#         with open(txt_path, "w") as f_txt:
#             if lines:
#                 f_txt.write("\n".join(lines))
#             # If no boxes, it writes an empty file (you can omit if you prefer)

#     print(f"[INFO] Converted {json_path} to YOLO format in '{output_dir}'.")


# %%
import os
import json
from PIL import Image

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(coco_json_path, "r") as f:
        data = json.load(f)
    
    images_dict = {img["id"]: img for img in data["images"]}
    cat_map = {}
    for cat in data["categories"]:
        if cat["name"] in class_names:
            cat_map[cat["id"]] = class_names.index(cat["name"])
    
    annotations_by_image = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)
    
    for image_id, image in images_dict.items():
        image_name = image["file_name"]
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            continue
        
        with Image.open(image_path) as img:
            w, h = img.size
        
        lines = []
        anns = annotations_by_image.get(image_id, [])
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            bw /= w
            bh /= h
            
            if ann["category_id"] not in cat_map:
                continue
            label_idx = cat_map[ann["category_id"]]
            
            lines.append(f"{label_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        txt_filename = os.path.splitext(image_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, "w") as f_txt:
            if lines:
                f_txt.write("\n".join(lines))
    
    print(f"[INFO] Converted {coco_json_path} to YOLO format in '{output_dir}'.")



# %%
# import subprocess

# def knowledge_distillation_yolov10(
#     rtm_json="test_dir/rtm_pseudo_labels.json",
#     detr_json="test_dir/detr_pseudo_labels.json",
#     merged_json="test_dir/merged_pseudo_labels.json",
#     images_dir="test_dir/images",
#     yolo_output_dir="test_dir/yolo_annotations",
#     data_config="distillation_data.yaml",
#     checkpoint_path=None,
#     epochs=20,
#     batch_size=32
# ):
    
#     # Step 1: Merge teacher boxes
#     merge_teacher_labels(rtm_json, detr_json, merged_json, iou_threshold=0.5)

#     # Step 2: Convert merged labels to YOLO format
#     #   Your classes are: D00, D10, D20, D40
#     class_names = ["D00", "D10", "D20", "D40"]
#     convert_json_to_yolo(merged_json, images_dir, yolo_output_dir, class_names)

    
#     print(f"[INFO] Resuming YOLOv10 training from {checkpoint_path} with merged pseudo labels...")
#     subprocess.run([
#             "python3", "yolo.py",
#             "--data_config", data_config,
#             "--epochs", str(epochs),
#             "--batch_size", str(batch_size),
#             "--new", "False",
#             "--ckpnt_path", checkpoint_path
#     ])



# %%
# knowledge_distillation_yolov10(
#         rtm_json="test_dir/rtm_pseudo_labels.json",
#         detr_json="test_dir/detr_pseudo_labels.json",
#         merged_json="test_dir/merged_pseudo_labels.json",
#         images_dir="test_dir/images",
#         yolo_output_dir="test_dir/yolo_annotations",
#         data_config="distillation_data.yaml",
#         checkpoint_path=None,
#         epochs=20,
#         batch_size=32
# )

# %%
"""
there exist a folder base_dir/yolo let this be yolo_prev
-yolo_final
     -images
        -train ( symlink from base_dir/test/images)
        -val (symlink from yolo_prev/images/val)
     -labels
        -train (where the merged annotations have to be stored)
        -val (symlink from yolo_prev/labels/val)

"""

# %%
import os
import subprocess
from pathlib import Path

def knowledge_distillation_yolov10(
    rtm_json="test_dir/rtm_pseudo_labels.json",
    detr_json="test_dir/detr_pseudo_labels.json",
    merged_json="test_dir/merged_pseudo_labels.json",
    images_dir="test_dir/images",
    yolo_output_dir="test_dir/yolo_annotations",
    data_config="distillation_data.yaml",
    checkpoint_path=None,
    epochs=20,
    batch_size=32,
    base_dir="base_dir",            
    yolo_prev="base_dir/yolo",      
    yolo_final="yolo_final"         
):
    merge_teacher_coco_annotations(rtm_json, detr_json, merged_json, iou_threshold=0.5)

    class_names = ["D00", "D10", "D20", "D40"]
    convert_coco_to_yolo(merged_json, images_dir, yolo_output_dir, class_names)

    create_yolo_final_structure(base_dir, yolo_prev, yolo_final)

    copy_converted_labels(yolo_output_dir, os.path.join(yolo_final, "labels", "train"))

    print(f"[INFO] Resuming YOLOv10 training from {checkpoint_path} with merged pseudo labels...")
    subprocess.run([
        "python3", "yolo.py",
        "--data_config", data_config,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--new", "False",
        "--ckpnt_path", checkpoint_path
    ])


def create_yolo_final_structure(base_dir, yolo_prev, yolo_final):
    """
    Creates the following folder structure:
    yolo_final/
      images/
        train/ (symlink to base_dir/test/images)
        val/   (symlink to yolo_prev/images/val)
      labels/
        train/ (actual folder)
        val/   (symlink to yolo_prev/labels/val)
    """
    base_dir = Path(base_dir).resolve()
    yolo_prev = Path(yolo_prev).resolve()
    yolo_final = Path(yolo_final).resolve()

    yolo_final.mkdir(parents=True, exist_ok=True)

    images_folder = yolo_final / "images"
    labels_folder = yolo_final / "labels"
    images_folder.mkdir(exist_ok=True)
    labels_folder.mkdir(exist_ok=True)

    train_images_src = base_dir / "test" / "images"
    train_images_dst = images_folder / "train"
    train_images_dst.mkdir(exist_ok=True)

    for img_file in train_images_src.iterdir():
        if img_file.is_file():
            symlink_target = train_images_dst / img_file.name
            if not symlink_target.exists():
                symlink_target.symlink_to(img_file.resolve())

    val_images_src = yolo_prev / "images" / "val"
    val_images_dst = images_folder / "val"
    if val_images_dst.exists():
        val_images_dst.unlink()  
    val_images_dst.symlink_to(val_images_src)

    train_labels_dst = labels_folder / "train"
    train_labels_dst.mkdir(exist_ok=True)

    val_labels_src = yolo_prev / "labels" / "val"
    val_labels_dst = labels_folder / "val"
    if val_labels_dst.exists():
        val_labels_dst.unlink()
    val_labels_dst.symlink_to(val_labels_src)


def copy_converted_labels(source_dir, target_dir):
    
    from shutil import copy2
    src = Path(source_dir).resolve()
    dst = Path(target_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    for txt_file in src.glob("*.txt"):
        copy2(txt_file, dst / txt_file.name)


# %%
knowledge_distillation_yolov10(
    rtm_json="/home/ansul/dummy_data/India/test/images/rtm_pseudo_labels.json",
    detr_json="/home/ansul/dummy_data/India/test/images/detr_pseudo_labels.json",
    merged_json="/home/ansul/dummy_data/India/test/images/merged_pseudo_labels.json",
    images_dir="/home/ansul/dummy_data/India/test/images",
    yolo_output_dir="/home/ansul/dummy_data/India/test/yolo_annotations",
    data_config="/home/ansul/AI_Road_Health_Assessment/config/yolov10_config2.yaml",
    checkpoint_path="/home/ansul/AI_Road_Health_Assessment/models/runs/detect/train/weights/best.pt",
    epochs=2,
    batch_size=16,
    base_dir="/home/ansul/dummy_data/India",            # base directory that has test/images
    yolo_prev="/home/ansul/dummy_data/India/yolo",      # existing YOLO setup
    yolo_final="/home/ansul/dummy_data/India/yolo_final"         # new directory structure we want to create
)


