import os
import torch
import cv2
import json
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from mmdet.apis import init_detector, inference_detector

from transformers import DetrFeatureExtractor, DetrForObjectDetection

class MutualLearningFramework:
    def __init__(
        self,
        rtmdet_config_path,
        rtmdet_checkpoint_path,
        detr_checkpoint_path,
        labeled_data_dir,
        unlabeled_data_dir,
        rtmdet_pseudo_labels_path,
        detr_pseudo_labels_path,
        output_dir="./mutual_learning_outputs",
        batch_size=32,
        num_iterations=5,
        learning_rate=1e-4
    ):
        self.rtmdet_config_path = rtmdet_config_path
        self.rtmdet_checkpoint_path = rtmdet_checkpoint_path
        self.detr_checkpoint_path = detr_checkpoint_path
        self.labeled_data_dir = labeled_data_dir
        self.unlabeled_data_dir = unlabeled_data_dir
        self.rtmdet_pseudo_labels_path = rtmdet_pseudo_labels_path
        self.detr_pseudo_labels_path = detr_pseudo_labels_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)
        self._init_models()
        self._load_pseudo_labels()
        self._merge_pseudo_labels()

    def _init_models(self):
        print("Initializing models...")
        self.rtmdet_model = init_detector(
            self.rtmdet_config_path,
            self.rtmdet_checkpoint_path,
            device=self.device
        )
        self.detr_lightning = DetrLightning.load_from_checkpoint(
            self.detr_checkpoint_path,
            map_location=self.device
        )
        self.detr_model = self.detr_lightning.model.to(self.device)
        self.detr_feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.id2label = self.detr_lightning.id2label
        print("Models initialized successfully")

    def _load_pseudo_labels(self):
        print("Loading pseudo labels...")
        with open(self.rtmdet_pseudo_labels_path, 'r') as f:
            self.rtmdet_pseudo_labels = json.load(f)
        with open(self.detr_pseudo_labels_path, 'r') as f:
            self.detr_pseudo_labels = json.load(f)
        print(f"Loaded {len(self.rtmdet_pseudo_labels)} RTMDet pseudo labels")
        print(f"Loaded {len(self.detr_pseudo_labels)} DETR pseudo labels")

    def _merge_pseudo_labels(self):
        print("Merging pseudo labels with NMS...")
        self.merged_pseudo_labels = {}
        all_images = set(list(self.rtmdet_pseudo_labels.keys()) + list(self.detr_pseudo_labels.keys()))
        for img_file in all_images:
            rtmdet_boxes = self.rtmdet_pseudo_labels.get(img_file, [])
            detr_boxes = self.detr_pseudo_labels.get(img_file, [])
            all_boxes = rtmdet_boxes + detr_boxes
            all_boxes.sort(key=lambda x: x["score"], reverse=True)
            kept_boxes = []
            for box in all_boxes:
                keep = True
                current_box = np.array(box["bbox"])
                for kept_box in kept_boxes:
                    kept_box_array = np.array(kept_box["bbox"])
                    x1 = max(current_box[0], kept_box_array[0])
                    y1 = max(current_box[1], kept_box_array[1])
                    x2 = min(current_box[2], kept_box_array[2])
                    y2 = min(current_box[3], kept_box_array[3])
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    intersection = w * h
                    area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                    area2 = (kept_box_array[2] - kept_box_array[0]) * (kept_box_array[3] - kept_box_array[1])
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    if iou > 0.5 and box["label"] == kept_box["label"]:
                        keep = False
                        break
                if keep:
                    kept_boxes.append(box)
            self.merged_pseudo_labels[img_file] = kept_boxes
        merged_labels_path = os.path.join(self.output_dir, "merged_pseudo_labels.json")
        with open(merged_labels_path, 'w') as f:
            json.dump(self.merged_pseudo_labels, f, indent=2)
        print(f"Merged pseudo labels saved to {merged_labels_path}")

class PseudoLabeledDataset(Dataset):
    def __init__(self, img_folder, pseudo_labels_json, feature_extractor):
        self.img_folder = img_folder
        self.feature_extractor = feature_extractor
        with open(pseudo_labels_json, 'r') as f:
            self.pseudo_labels = json.load(f)
        self.img_files = list(self.pseudo_labels.keys())

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_file)
        image = Image.open(img_path).convert("RGB")
        boxes = self.pseudo_labels[img_file]
        annotations = []
        for box in boxes:
            if isinstance(box["label"], str):
                class_id = int(box["label"]) if box["label"].isdigit() else 0
            else:
                class_id = box["label"]
            annotations.append({
                "bbox": box["bbox"],
                "category_id": class_id,
                "score": box["score"]
            })
        target = {
            "image_id": idx,
            "annotations": annotations
        }
        encoding = self.feature_extractor(images=image, annotations=target, return_tensors="pt")
        return encoding["pixel_values"].squeeze(), encoding["labels"][0]

def _create_rtmdet_config_with_pseudo_labels(self, iteration):
    print("Creating RTMDet config with pseudo labels...")
    pseudo_ann_file = os.path.join(self.output_dir, f"rtmdet_pseudo_ann_iter_{iteration}.json")
    self._convert_pseudo_labels_to_coco(pseudo_ann_file)
    custom_config_path = os.path.join(self.output_dir, f"rtmdet_config_iter_{iteration}.py")
    custom_config = f"""
_base_ = ['{os.path.basename(self.rtmdet_config_path)}']

data = {{
    'train': {{
        'type': 'ConcatDataset',
        'datasets': [
            {{
                'type': 'CocoDataset',
                'ann_file': '{os.path.join(self.labeled_data_dir, "annotations", "annotations_coco.json")}',
                'img_prefix': '{os.path.join(self.labeled_data_dir, "train")}',
                'pipeline': _base_.train_pipeline
            }},
            {{
                'type': 'CocoDataset',
                'ann_file': '{pseudo_ann_file}',
                'img_prefix': '{self.unlabeled_data_dir}',
                'pipeline': _base_.train_pipeline
            }}
        ]
    }},
    'val': _base_.data.val,
    'test': _base_.data.test
}}

runner = {{'type': 'EpochBasedRunner', 'max_epochs': 5}}
work_dir = '{os.path.join(self.output_dir, f"rtmdet_work_dir_iter_{iteration}")}'
"""
    with open(custom_config_path, 'w') as f:
        f.write(custom_config)
    print(f"Custom RTMDet config created at {custom_config_path}")
    return custom_config_path, pseudo_ann_file

def _convert_pseudo_labels_to_coco(self, output_path):
    print("Converting pseudo labels to COCO format...")
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    for id, name in self.id2label.items():
        coco_data["categories"].append({
            "id": id,
            "name": name,
            "supercategory": "object"
        })
    image_id = 0
    annotation_id = 0
    for img_file, boxes in self.merged_pseudo_labels.items():
        img_path = os.path.join(self.unlabeled_data_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })
        for box in boxes:
            bbox = box["bbox"]
            coco_bbox = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]
            ]
            if isinstance(box["label"], str):
                category_id = None
                for cat in coco_data["categories"]:
                    if cat["name"] == box["label"]:
                        category_id = cat["id"]
                        break
                if category_id is None:
                    category_id = coco_data["categories"][0]["id"]
            else:
                category_id = box["label"]
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1
        image_id += 1
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
    print(f"Converted {len(coco_data['images'])} images with {len(coco_data['annotations'])} annotations")

def _train_rtmdet(self, iteration):
    print(f"Training RTMDet (Iteration {iteration})...")
    custom_config_path, pseudo_ann_file = self._create_rtmdet_config_with_pseudo_labels(iteration)
    train_cmd = f"python mmyolo/tools/train.py {custom_config_path} --resume-from {self.rtmdet_checkpoint_path}"
    print(f"Running command: {train_cmd}")
    os.system(train_cmd)
    self.rtmdet_checkpoint_path = os.path.join(
        self.output_dir, f"rtmdet_work_dir_iter_{iteration}", "latest.pth")
    self.rtmdet_model = init_detector(
        custom_config_path,
        self.rtmdet_checkpoint_path,
        device=self.device
    )
    print(f"RTMDet training completed. Checkpoint saved to {self.rtmdet_checkpoint_path}")
    return self.rtmdet_checkpoint_path

def _train_detr(self, iteration):
    print(f"Training DETR (Iteration {iteration})...")
    pseudo_labels_path = os.path.join(self.output_dir, f"detr_pseudo_labels_iter_{iteration}.json")
    with open(pseudo_labels_path, 'w') as f:
        json.dump(self.merged_pseudo_labels, f, indent=2)
    train_folder = os.path.join(self.labeled_data_dir, "train")
    val_folder = os.path.join(self.labeled_data_dir, "val")
    train_dataset = DetrCocoDataset(
        img_folder=train_folder, 
        feature_extractor=self.detr_feature_extractor, 
        train=True
    )
    val_dataset = DetrCocoDataset(
        img_folder=val_folder, 
        feature_extractor=self.detr_feature_extractor, 
        train=False
    )
    pseudo_dataset = PseudoLabeledDataset(
        img_folder=self.unlabeled_data_dir,
        pseudo_labels_json=pseudo_labels_path,
        feature_extractor=self.detr_feature_extractor
    )
    combined_dataset = ConcatDataset([train_dataset, pseudo_dataset])
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        pixel_values = self.detr_feature_extractor.pad(pixel_values, return_tensors="pt")
        return {
            'pixel_values': pixel_values['pixel_values'],
            'pixel_mask': pixel_values['pixel_mask'],
            'labels': targets
        }
    train_loader = DataLoader(
        combined_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    detr_lightning = DetrLightning(
        lr=self.learning_rate,
        lr_backbone=self.learning_rate / 10,
        weight_decay=1e-4,
        num_labels=len(self.id2label),
        id2label=self.id2label
    )
    detr_lightning.model.load_state_dict(self.detr_model.state_dict())
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=5,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
        default_root_dir=os.path.join(self.output_dir, f"detr_logs_iter_{iteration}")
    )
    trainer.fit(detr_lightning, train_loader, val_loader)
    checkpoint_path = os.path.join(self.output_dir, f"detr_model_iter_{iteration}.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    self.detr_model = detr_lightning.model
    self.detr_lightning = detr_lightning
    self.detr_checkpoint_path = checkpoint_path
    print(f"DETR training completed. Checkpoint saved to {self.detr_checkpoint_path}")
    return self.detr_checkpoint_path

def run_mutual_learning(self):
    print(f"Starting mutual learning with {self.num_iterations} iterations")
    for iteration in range(1, self.num_iterations + 1):
        print(f"\n=== Mutual Learning Iteration {iteration}/{self.num_iterations} ===\n")
        rtmdet_checkpoint = self._train_rtmdet(iteration)
        rtmdet_pseudo_path = os.path.join(self.output_dir, f'rtmdet_pseudo_iter_{iteration}.json')
        pseudo_label_inference(self.rtmdet_model, self.unlabeled_data_dir, rtmdet_pseudo_path)
        with open(rtmdet_pseudo_path, 'r') as f:
            self.rtmdet_pseudo_labels = json.load(f)
        detr_checkpoint = self._train_detr(iteration)
        detr_pseudo_path = os.path.join(self.output_dir, f'detr_pseudo_iter_{iteration}.json')
        pseudo_label_inference_detr(
            self.detr_checkpoint_path,
            self.unlabeled_data_dir,
            detr_pseudo_path,
            score_threshold=0.5
        )
        with open(detr_pseudo_path, 'r') as f:
            self.detr_pseudo_labels = json.load(f)
        self._merge_pseudo_labels()
        print(f"\n=== Completed Iteration {iteration}/{self.num_iterations} ===")
        print(f"RTMDet checkpoint: {rtmdet_checkpoint}")
        print(f"DETR checkpoint: {detr_checkpoint}")
    print("\nMutual learning completed successfully!")
    print(f"Final RTMDet checkpoint: {self.rtmdet_checkpoint_path}")
    print(f"Final DETR checkpoint: {self.detr_checkpoint_path}")
    return self.rtmdet_checkpoint_path, self.detr_checkpoint_path
