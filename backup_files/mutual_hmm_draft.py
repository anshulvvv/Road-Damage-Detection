import os
import argparse
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
import torchvision
from torchvision.ops import nms
from PIL import Image
import pytorch_lightning as pl
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from mmdet.apis import init_detector, inference_detector

class MutualLearningFramework:
    def __init__(self, 
                 labeled_data_dir,
                 unlabeled_data_dir,
                 detr_inference_func,
                 rtmdet_inference_func,
                 detr_checkpoint=None,
                 rtmdet_checkpoint=None,
                 rtmdet_config=None,
                 num_classes=80,
                 batch_size=4):
        self.labeled_data_dir = labeled_data_dir
        self.unlabeled_data_dir = unlabeled_data_dir
        self.detr_inference_func = detr_inference_func
        self.rtmdet_inference_func = rtmdet_inference_func
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.setup_models(detr_checkpoint, rtmdet_checkpoint, rtmdet_config)
        self.setup_data_loaders()
    
    def setup_models(self, detr_checkpoint, rtmdet_checkpoint, rtmdet_config):
        self.detr_teacher = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", 
            num_labels=self.num_classes
        ).to(self.device)
        if detr_checkpoint and os.path.exists(detr_checkpoint):
            self.detr_teacher.load_state_dict(torch.load(detr_checkpoint, map_location=self.device))
        for param in self.detr_teacher.parameters():
            param.requires_grad = False
        self.detr_teacher.eval()
        self.detr_student = DetrLightning(
            lr=1e-4,
            lr_backbone=1e-5,
            weight_decay=1e-4,
            num_labels=self.num_classes,
            id2label={i: f"class_{i}" for i in range(self.num_classes)}
        )
        self.rtmdet_config = rtmdet_config
        self.rtmdet_checkpoint = rtmdet_checkpoint
    
    def setup_data_loaders(self):
        labeled_ann_file = os.path.join(self.labeled_data_dir, "annotations", "annotations_coco.json")
        self.labeled_dataset = DetrCocoDataset(
            img_folder=self.labeled_data_dir,
            feature_extractor=self.feature_extractor,
            ann_file=labeled_ann_file
        )
        self.labeled_loader = DataLoader(
            self.labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.feature_extractor)
        )
        self.unlabeled_dataset = UnlabeledDataset(
            img_folder=self.unlabeled_data_dir,
            feature_extractor=self.feature_extractor
        )
        self.unlabeled_loader = DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.feature_extractor, has_labels=False)
        )
    
    def train(self, num_epochs=10, save_dir='./mutual_learning'):
        os.makedirs(save_dir, exist_ok=True)
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=1,
            gradient_clip_val=0.1
        )
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            pseudo_labels = self.generate_pseudo_labels()
            trainer.fit(self.detr_student, self.labeled_loader)
            self.train_with_pseudo_labels(pseudo_labels, trainer)
            self.train_rtmdet_with_pseudo_labels(pseudo_labels, save_dir)
            if epoch > 0:
                self.update_teacher_models(save_dir, epoch)
            self.save_checkpoints(save_dir, epoch)
    
    def generate_pseudo_labels(self):
        print("Generating pseudo labels...")
        pseudo_labels = {}
        with torch.no_grad():
            for batch in self.unlabeled_loader:
                image_ids = batch['image_ids']
                detr_outputs = self.detr_teacher(
                    pixel_values=batch['pixel_values'].to(self.device),
                    pixel_mask=batch.get('pixel_mask', None)
                )
                for i, img_id in enumerate(image_ids):
                    scores = detr_outputs.logits[i].softmax(-1)
                    scores = scores[:, :-1]
                    best_scores, classes = scores.max(-1)
                    boxes = detr_outputs.pred_boxes[i]
                    filtered_boxes, filtered_scores, filtered_classes = self.apply_nms_and_threshold(
                        boxes, best_scores, classes
                    )
                    annotations = []
                    for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
                        x, y, w, h = box.cpu().tolist()
                        w = w - x
                        h = h - y
                        annotations.append({
                            'bbox': [x, y, w, h],
                            'category_id': int(cls.item()),
                            'score': float(score.item())
                        })
                    pseudo_labels[img_id] = annotations
        return pseudo_labels
    
    def apply_nms_and_threshold(self, boxes, scores, classes, iou_threshold=0.5, score_threshold=0.5):
        keep = scores > score_threshold
        if not torch.any(keep):
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        filtered_classes = classes[keep]
        unique_classes = torch.unique(filtered_classes)
        keep_boxes, keep_scores, keep_classes = [], [], []
        for cls in unique_classes:
            cls_mask = filtered_classes == cls
            if not torch.any(cls_mask):
                continue
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = filtered_scores[cls_mask]
            xc, yc, w, h = cls_boxes.unbind(-1)
            boxes_xyxy = torch.stack([
                xc - 0.5 * w, yc - 0.5 * h,
                xc + 0.5 * w, yc + 0.5 * h
            ], dim=-1)
            keep_idx = nms(boxes_xyxy, cls_scores, iou_threshold)
            keep_boxes.append(cls_boxes[keep_idx])
            keep_scores.append(cls_scores[keep_idx])
            keep_classes.append(filtered_classes[cls_mask][keep_idx])
        if keep_boxes:
            return torch.cat(keep_boxes), torch.cat(keep_scores), torch.cat(keep_classes)
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    def train_with_pseudo_labels(self, pseudo_labels, trainer):
        print("Training DETR student with pseudo labels...")
        pseudo_dataset = PseudoLabelDataset(
            self.unlabeled_dataset,
            pseudo_labels,
            self.feature_extractor
        )
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.feature_extractor)
        )
        trainer.fit(self.detr_student, pseudo_loader)
    
    def train_rtmdet_with_pseudo_labels(self, pseudo_labels, save_dir):
        print("Training RTM_DET student with pseudo labels...")
        pseudo_labels_file = os.path.join(save_dir, "pseudo_labels.json")
        self.save_pseudo_labels_coco(pseudo_labels, pseudo_labels_file)
    
    def save_pseudo_labels_coco(self, pseudo_labels, output_file):
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": f"class_{i}"} for i in range(self.num_classes)]
        }
        ann_id = 0
        for img_id, annotations in pseudo_labels.items():
            coco_data["images"].append({
                "id": img_id,
                "file_name": f"{img_id}.jpg"
            })
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
        with open(output_file, "w") as f:
            json.dump(coco_data, f)
    
    def update_teacher_models(self, save_dir, epoch):
        print("Updating teacher models...")
        self.detr_teacher.load_state_dict(self.detr_student.model.state_dict())
        for param in self.detr_teacher.parameters():
            param.requires_grad = False
        self.detr_teacher.eval()
        self.rtmdet_checkpoint = os.path.join(save_dir, f"rtmdet_student_epoch_{epoch}.pth")
    
    def save_checkpoints(self, save_dir, epoch):
        detr_save_path = os.path.join(save_dir, f"detr_student_epoch_{epoch}.pth")
        torch.save(self.detr_student.state_dict(), detr_save_path)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, feature_extractor):
        self.img_folder = img_folder
        self.feature_extractor = feature_extractor
        self.image_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(self.img_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        return pixel_values, None, image_id

class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pseudo_labels, feature_extractor):
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        pixel_values, _, image_id = self.dataset[idx]
        annotations = self.pseudo_labels.get(image_id, [])
        boxes = []
        class_labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            class_labels.append(ann['category_id'])
        if boxes:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(class_labels, dtype=torch.long)
            }
        else:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.long)
            }
        return pixel_values, target, image_id

def collate_fn(batch, feature_extractor, has_labels=True):
    pixel_values = [item[0] for item in batch]
    image_ids = [item[2] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    result = {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding.get('pixel_mask'),
        'image_ids': image_ids
    }
    if has_labels:
        labels = [item[1] for item in batch if item[1] is not None]
        if labels:
            result['labels'] = labels
    return result

class MutualLearningLoss(torch.nn.Module):
    def __init__(self, bbox_weight=1.0, class_weight=1.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.bbox_loss = torch.nn.SmoothL1Loss()
        self.class_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred_boxes, pred_classes, target_boxes, target_classes):
        bbox_loss = self.bbox_loss(pred_boxes, target_boxes)
        class_loss = self.class_loss(pred_classes, target_classes)
        total_loss = self.bbox_weight * bbox_loss + self.class_weight * class_loss
        return total_loss, {"bbox_loss": bbox_loss, "class_loss": class_loss}

def train_mutual_learning(
        labeled_data_dir,
        unlabeled_data_dir,
        detr_checkpoint=None,
        rtmdet_checkpoint=None,
        rtmdet_config=None,
        num_classes=80,
        batch_size=4,
        num_epochs=10,
        save_dir='./mutual_learning'
    ):
    def detr_inference_func(images):
        pass
    
    def rtmdet_inference_func(images):
        pass
    
    framework = MutualLearningFramework(
        labeled_data_dir=labeled_data_dir,
        unlabeled_data_dir=unlabeled_data_dir,
        detr_inference_func=detr_inference_func,
        rtmdet_inference_func=rtmdet_inference_func,
        detr_checkpoint=detr_checkpoint,
        rtmdet_checkpoint=rtmdet_checkpoint,
        rtmdet_config=rtmdet_config,
        num_classes=num_classes,
        batch_size=batch_size
    )
    framework.train(num_epochs=num_epochs, save_dir=save_dir)
    return os.path.join(save_dir, f"detr_student_epoch_{num_epochs-1}.pth"), os.path.join(save_dir, f"rtmdet_student_epoch_{num_epochs-1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RTM_DET and DETR with mutual learning")
    parser.add_argument("--labeled_data_dir", type=str, required=True, help="Directory containing labeled data")
    parser.add_argument("--unlabeled_data_dir", type=str, required=True, help="Directory containing unlabeled data")
    parser.add_argument("--detr_checkpoint", type=str, default=None, help="Path to DETR checkpoint")
    parser.add_argument("--rtmdet_checkpoint", type=str, default=None, help="Path to RTM_DET checkpoint")
    parser.add_argument("--rtmdet_config", type=str, required=True, help="Path to RTM_DET config")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of object classes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./mutual_learning", help="Directory to save checkpoints")
    args = parser.parse_args()
    train_mutual_learning(
        labeled_data_dir=args.labeled_data_dir,
        unlabeled_data_dir=args.unlabeled_data_dir,
        detr_checkpoint=args.detr_checkpoint,
        rtmdet_checkpoint=args.rtmdet_checkpoint,
        rtmdet_config=args.rtmdet_config,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
