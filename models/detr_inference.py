import os
import json
import torch
from PIL import Image
import torchvision.ops as ops
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from pytorch_lightning import LightningModule
import argparse

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

# def pseudo_label_inference_detr(checkpoint_path, test_dir, output_json, score_threshold, nms_threshold):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #device = torch.device("cpu")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     id2label = checkpoint["hyper_parameters"]["id2label"]
#     num_labels = len(id2label)
#     pl_model = DetrLightning.load_from_checkpoint(
#         checkpoint_path,
#         lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
#         num_labels=num_labels,
#         id2label=id2label
#     )
#     model = pl_model.model.to(device).eval()
#     feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
#     pseudo_labels = {}
#     image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     for img_file in image_files:
#         img_path = os.path.join(test_dir, img_file)
#         image = Image.open(img_path).convert("RGB")
#         inputs = feature_extractor(images=image, return_tensors="pt")
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = model(**inputs)
#         target_sizes = torch.tensor([image.size[::-1]]).to(device)
#         results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
#         boxes_tensor = results["boxes"]
#         scores_tensor = results["scores"]
#         labels_tensor = results["labels"]
#         valid = scores_tensor >= score_threshold
#         boxes_tensor = boxes_tensor[valid]
#         scores_tensor = scores_tensor[valid]
#         labels_tensor = labels_tensor[valid]
#         boxes_filtered = []
#         if boxes_tensor.shape[0] > 0:
#             keep = ops.batched_nms(boxes_tensor, scores_tensor, labels_tensor, nms_threshold)
#             for idx in keep:
#                 boxes_filtered.append({
#                     "label": pl_model.id2label[int(labels_tensor[idx].item())],
#                     "bbox": boxes_tensor[idx].tolist(),
#                     "score": float(scores_tensor[idx].item())
#                 })
#         pseudo_labels[img_file] = boxes_filtered
#     with open(output_json, "w") as f:
#         json.dump(pseudo_labels, f, indent=2)
#     print(f"Pseudo labels saved to {output_json} (Threshold: {score_threshold}, NMS: {nms_threshold})")


def pseudo_label_inference_detr(checkpoint_path, test_dir, output_json, score_threshold, nms_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    id2label = checkpoint["hyper_parameters"]["id2label"]
    num_labels = len(id2label)
    pl_model = DetrLightning.load_from_checkpoint(
        checkpoint_path,
        lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
        num_labels=num_labels,
        id2label=id2label
    )
    model = pl_model.model.to(device).eval()

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    images_list = []
    annotations_list = []
    annotation_id = 0 
    image_id = 0 

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size  
        images_list.append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([[height, width]], device=device)
        results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
        boxes_tensor = results["boxes"]
        scores_tensor = results["scores"]
        labels_tensor = results["labels"]
        valid = scores_tensor >= score_threshold
        boxes_tensor = boxes_tensor[valid]
        scores_tensor = scores_tensor[valid]
        labels_tensor = labels_tensor[valid]
        
        if boxes_tensor.shape[0] > 0:
            keep = ops.batched_nms(boxes_tensor, scores_tensor, labels_tensor, nms_threshold)
            for idx in keep:
                bbox = boxes_tensor[idx].tolist()
                x_min, y_min, x_max, y_max = bbox
                w = x_max - x_min
                h = y_max - y_min
                coco_bbox = [x_min, y_min, w, h]
                
                annotations_list.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(labels_tensor[idx].item()),  
                    "bbox": coco_bbox,
                    "score": float(scores_tensor[idx].item()), 
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    categories_list = [{"id": int(k), "name": v} for k, v in id2label.items()]

    coco_output = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories_list
    }

    with open(output_json, "w") as f_out:
        json.dump(coco_output, f_out, indent=2)
        
    print(f"Pseudo labels saved to {output_json} (score_threshold: {score_threshold}, nms_threshold: {nms_threshold})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    pseudo_label_inference_detr(args.checkpoint_path, args.test_dir, args.output_json,
                                args.score_threshold, args.nms_threshold)
