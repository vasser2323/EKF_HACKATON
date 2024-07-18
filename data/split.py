import os
import random
import shutil
from icecream import ic

images_folder = os.path.join("dt", "images", "train")
annotations_folder = os.path.join("dt", "labels", "train")
file_names = [
    os.path.splitext(os.path.basename(file_path))[0]
    for file_path in os.listdir(images_folder)
]
ic(file_names)

val_images = random.sample(file_names, 8)
ic(val_images)

images_val_folder = os.path.join(images_folder, "test")
if not os.path.exists(images_val_folder):
    os.makedirs(images_val_folder)

annotations_val_folder = os.path.join(annotations_folder, "test")
if not os.path.exists(annotations_val_folder):
    os.makedirs(annotations_val_folder)

for image_name in val_images:

    source_path = os.path.join(images_folder, f"{image_name}.jpg")
    dest_path = os.path.join(images_val_folder, f"{image_name}.jpg")
    shutil.move(source_path, dest_path)

    source_path = os.path.join(annotations_folder, f"{image_name}.txt")
    dest_path = os.path.join(annotations_val_folder, f"{image_name}.txt")

    shutil.move(source_path, dest_path)
