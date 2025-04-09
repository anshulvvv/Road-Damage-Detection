import os
import argparse
import torch


from torch.nn import Sequential
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from numpy.core.multiarray import scalar
torch.serialization.add_safe_globals([Conv])
torch.serialization.add_safe_globals([scalar])
torch.serialization.add_safe_globals({"ultralytics.nn.tasks.DetectionModel": DetectionModel})
from ultralytics import YOLO

def run_training(data_config, epochs, batch_size, new, checkpoint_path ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model_name = "jameslahm/yolov10n"
    if(new) :
        model = YOLO(model_name)
        print(f"Starting training with data config: {data_config}, epochs: {epochs}, batch: {batch_size}")
        model.train(data=data_config, epochs=epochs, batch=batch_size)
    else :
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        print(f"Resuming training with data config: {data_config}, epochs: {epochs}, batch: {batch_size}")
        model.train(data=data_config, epochs=epochs, batch=batch_size)

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv10 model from scratch")
    parser.add_argument("--data_config", type=str, required=True,
                        help="Path to YOLOv10 data configuration YAML file")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--new", type=bool,required=True,
                        help="Train a new model (default: True)")
    parser.add_argument("--ckpnt_path", type=str, default= None,
                        help="Path to the checkpoint file to resume training from")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.new and args.ckpnt_path == "NULL":
        args.error("When not training a new model (i.e., --new not set), you must provide --ckpnt_path.")
    run_training(args.data_config, args.epochs, args.batch_size, args.new , args.ckpnt_path)
