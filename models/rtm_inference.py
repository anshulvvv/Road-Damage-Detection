#!/usr/bin/env python
import os
import json
import numpy as np
import torch
import torchvision.ops as ops
from mmdet.apis import init_detector, inference_detector
import argparse
from PIL import Image

# def pseudo_label_inference(model, test_dir, output_json, score_thresh, nms_thresh):
#     pseudo_labels = {}
#     # List image files with valid extensions
#     image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
#     for img_file in image_files:
#         img_path = os.path.join(test_dir, img_file)
#         # Get the detection result; result is a DetDataSample object
#         result = inference_detector(model, img_path)
#         boxes_filtered = []
        
#         # Access detection results from the 'pred_instances' attribute
#         pred_instances = result.pred_instances
#         if pred_instances is None or len(pred_instances) == 0:
#             pseudo_labels[img_file] = []
#             continue

#         # Extract bounding boxes, scores, and labels; convert to NumPy arrays if needed.
#         bboxes = pred_instances.bboxes.cpu().numpy() if isinstance(pred_instances.bboxes, torch.Tensor) else np.array(pred_instances.bboxes)
#         scores = pred_instances.scores.cpu().numpy() if isinstance(pred_instances.scores, torch.Tensor) else np.array(pred_instances.scores)
#         labels = pred_instances.labels.cpu().numpy() if isinstance(pred_instances.labels, torch.Tensor) else np.array(pred_instances.labels)
        
#         # Process detections by unique class label
#         unique_labels = np.unique(labels)
#         for cls in unique_labels:
#             # Get indices for the current class
#             cls_indices = np.where(labels == cls)[0]
#             cls_boxes = bboxes[cls_indices]
#             cls_scores = scores[cls_indices]
            
#             # Filter detections by score threshold
#             keep = cls_scores >= score_thresh
#             if not np.any(keep):
#                 continue
#             cls_boxes = cls_boxes[keep]
#             cls_scores = cls_scores[keep]
            
#             # Apply NMS using PyTorch's NMS function
#             boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32)
#             scores_tensor = torch.tensor(cls_scores, dtype=torch.float32)
#             keep_indices = ops.nms(boxes_tensor, scores_tensor, nms_thresh)
#             selected_boxes = cls_boxes[keep_indices.numpy()]
#             selected_scores = cls_scores[keep_indices.numpy()]
            
#             # Append each detection to the filtered list
#             for box, score in zip(selected_boxes, selected_scores):
#                 boxes_filtered.append({
#                     "label": int(cls),
#                     "bbox": box.tolist(),
#                     "score": float(score)
#                 })

#         pseudo_labels[img_file] = boxes_filtered
#         print(f"Processed {img_file}: Found {len(boxes_filtered)} detections.")

#     # Write the pseudo labels to a JSON file
#     with open(output_json, "w") as f:
#         json.dump(pseudo_labels, f, indent=2)
#     print("Pseudo labels saved to", output_json)

def pseudo_label_inference(model, test_dir, output_json, score_thresh, nms_thresh):
    images_list = []       
    annotations_list = [] 
    categories_set = set() 
    annotation_id = 0      
    image_id = 0           

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_file}: {e}")
            continue
        width, height = image.size 

        images_list.append({
            "id": int(image_id),
            "file_name": img_file,
            "width": int(width),
            "height": int(height)
        })

        result = inference_detector(model, img_path)

        pred_instances = result.pred_instances
        if pred_instances is None or len(pred_instances) == 0:
            image_id += 1
            continue

        bboxes = pred_instances.bboxes.cpu().numpy() if isinstance(pred_instances.bboxes, torch.Tensor) else np.array(pred_instances.bboxes)
        scores = pred_instances.scores.cpu().numpy() if isinstance(pred_instances.scores, torch.Tensor) else np.array(pred_instances.scores)
        labels = pred_instances.labels.cpu().numpy() if isinstance(pred_instances.labels, torch.Tensor) else np.array(pred_instances.labels)

        unique_labels = np.unique(labels)
        for cls in unique_labels:
            categories_set.add(int(cls))

            cls_indices = np.where(labels == cls)[0]
            cls_boxes = bboxes[cls_indices]
            cls_scores = scores[cls_indices]

            valid_mask = cls_scores >= score_thresh
            if not np.any(valid_mask):
                continue
            cls_boxes = cls_boxes[valid_mask]
            cls_scores = cls_scores[valid_mask]

            boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(cls_scores, dtype=torch.float32)
            keep_indices = ops.nms(boxes_tensor, scores_tensor, nms_thresh)
            selected_boxes = cls_boxes[keep_indices.numpy()]
            selected_scores = cls_scores[keep_indices.numpy()]

            for box, score in zip(selected_boxes, selected_scores):
                x_min, y_min, x_max, y_max = box
                w = x_max - x_min
                h = y_max - y_min
                coco_bbox = [float(x_min), float(y_min), float(w), float(h)]
                annotations_list.append({
                    "id": int(annotation_id),
                    "image_id": int(image_id),
                    "category_id": int(cls), 
                    "bbox": coco_bbox,
                    "score": float(score),
                    "area": float(w * h),
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    categories_list = [{"id": int(cat_id), "name": f"Category {int(cat_id)}"} for cat_id in sorted(categories_set)]

    coco_output = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories_list
    }

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)
    print("COCO-style pseudo labels saved to", output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpnt", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_detector(args.config_path, args.ckpnt, device=device)
    pseudo_label_inference(model, args.test_dir, args.output_json, args.score_thresh, args.nms_thresh)
