import os
import torch
import cv2
import json
import argparse
import numpy as np

from mmdet.apis import init_detector, inference_detector


DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_DATA_ROOT = "C:/Users/ANSHUL M/Downloads/RDD2022_India/India/"

def create_custom_config(data_root, batch_size, epochs):
    
    HOME = os.getcwd()
    config_dir = os.path.join(HOME, "configs", "rtmdet")
    os.makedirs(config_dir, exist_ok=True)
    custom_config_path = os.path.join(config_dir, "custom.py")
    
    CUSTOM_CONFIG = f"""
_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']
        custom-config goes here
                        """
    with open(custom_config_path, "w") as f:
        f.write(CUSTOM_CONFIG)
    print(f"Custom config created at: {custom_config_path}")
    return custom_config_path

def train_model(custom_config_path , new ,resume_checkpoint=None):
    
    if(new) : 
        train_cmd = f"python mmyolo/tools/train.py {custom_config_path}"
        print("Starting training with command:", train_cmd)
        os.system(train_cmd)
    else :
        train_cmd = f"python mmyolo/tools/train.py {custom_config_path} --resume-from {resume_checkpoint}"
        print("Resuming training with command:", train_cmd)
        os.system(train_cmd)
    checkpoint_path = os.path.join("work_dirs", "custom", "latest.pth")
    return checkpoint_path

def run_training_and_inference(
                               data_root=DEFAULT_DATA_ROOT,
                               batch_size=DEFAULT_BATCH_SIZE,
                               epochs=DEFAULT_EPOCHS,
                               new = True,
                               resume_checkpoint= None
                               ):
    
    config_path = create_custom_config(data_root, batch_size, epochs)
    checkpoint_path = train_model(config_path,new,resume_checkpoint)
    return checkpoint_path,config_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train RTMDet and generate pseudo labels on test set")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Root directory for training data")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--new", type=bool, default=True, help="Whether to start training from scratch")
    parser.add_argument("--resume_checkpoint", type=str, default= None, help="Path to checkpoint to resume training from")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training_and_inference(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        new = args.new,
        resume_checkpoint=args.resume_checkpoint
    )