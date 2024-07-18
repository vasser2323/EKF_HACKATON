# FIXME
import os
import cv2
import shutil
import src.Utility.utilities

from icecream import ic
import xml.etree.ElementTree as ET

unique_classes = dict()
unique_classes_counter = 0

for num in [330, 331, 333, 334, 482]:
    tree = ET.parse(os.path.join("data", "annotations_xml", f"annotations_{num}.xml"))
    root = tree.getroot()

    output_folder = os.path.join("data", "dataset", "labels", "train")
    output_folder = "converted"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image in root.findall(".//image"):
        image_file_name = os.path.splitext(os.path.basename(image.get("name")))[0]
        output_file_path = os.path.join(output_folder, f"{image_file_name}.txt")

        # image_file_path = os.path.join(
        #     "dataset", "images", "train", f"{image_file_name}.jpg"
        # )
        image_file_path = os.path.join("images", f"{image_file_name}.jpg")
        image_cv2 = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

        if image_cv2 is None:
            continue

        image_height, image_width, channels = image_cv2.shape

        ic(image_file_path, image_width, image_height)

        with open(output_file_path, "w") as f:

            # f.write(f"Image: {image.get('name')}\n")

            for box in image.findall("box"):

                label = src.Utility.utilities.replace_unsafe_characters(
                    box.get("label")
                ).upper()

                xtl = float(box.get("xtl")) / image_width
                ytl = float(box.get("ytl")) / image_height

                xbr = float(box.get("xbr")) / image_width
                ybr = float(box.get("ybr")) / image_height

                width = round(xbr - xtl, 5)
                height = round(ybr - ytl, 5)

                center_x = round((xtl + xbr) / 2, 5)
                center_y = round((ytl + ybr) / 2, 5)

                if label not in unique_classes.keys():
                    unique_classes[label] = unique_classes_counter
                    unique_classes_counter += 1

                f.write(
                    f"{unique_classes[label]} {center_x} {center_y} {width} {height}\n"
                )

with open("classes.txt", "w") as f:
    for class_name, class_index in unique_classes.items():
        f.write(f"{class_name}\n")

with open("data_temp.yaml", "w") as f:
    f.write("names:\n")
    for class_name, class_index in unique_classes.items():
        f.write(f"  {class_index}: {class_name}\n")
