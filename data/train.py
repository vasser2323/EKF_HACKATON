import os
import shutil
from ultralytics import YOLO

if __name__ == "__main__":
    # load a pretrained model (recommended for training)
    model = YOLO("weights/yolov8l.pt")
    # model = YOLO("weights/last.pt")

    # Train the model
    # results = model.train(
    #     data="data.yaml",
    #     optimizer="RAdam",
    #     dropout=0.001,
    #     agnostic_nms=True,
    #     epochs=50,
    #     imgsz=640,
    #     batch=16,
    #     workers=8,
    #     amp=True,
    #     # ----------
    #     show=True,
    #     save_json=True,
    #     plots=True,
    #     save=True,
    #     # ----------
    #     hsv_v=0.4,
    #     erasing=0.3,
    # )

    results = model.train(
        data="data.yaml",
        # optimizer="RAdam",
        # dropout=0.001,
        # agnostic_nms=True,
        epochs=100,
        imgsz=1280,
        batch=4,
        workers=8,
        save_period=3,
        fliplr=0,
        mosaic=0,
        # ----------
        # show=True,
        # save_json=True,
        # plots=True,
        # save=True,
        # ----------
        # hsv_v=0.4,
        # erasing=0.3,
    )
