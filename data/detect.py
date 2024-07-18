import argparse, sys
from ultralytics import YOLO

# Allow to pass source file as an argument
parser = argparse.ArgumentParser()
parser.add_argument("--source", help="Path to the image or video file")
parser.add_argument("--weights", help="Path to the model file")
args = parser.parse_args()

if args.weights is not None and args.source is not None:
    model = YOLO(args.weights)
    print("Model loaded successfully")

    print(f"Processing {args.source}")
    result = model(args.source, save=True)