import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import sys

# Paths
IMAGE_DIR = r"data\colorferet\converted_images\images"
XML_DIR = r"data\colorferet\converted_images\ground_truths\xml"
OUTPUT_DIR = r"data\colorferet\converted_images\cropped_faces"

def get_face_region(xml_path):
    """ Extracts facial landmarks and computes a bounding box. """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        face = root.find(".//Face")  # Find <Face> element

        if face is None:
            print(f"Error: <Face> tag missing in {xml_path}")
            return None

        def get_coordinates(tag):
            element = face.find(tag)
            if element is not None and "x" in element.attrib and "y" in element.attrib:
                return int(element.attrib["x"]), int(element.attrib["y"])
            else:
                print(f"Warning: Missing {tag} in {xml_path}")
                return None

        left_eye = get_coordinates("LeftEye")
        right_eye = get_coordinates("RightEye")
        nose = get_coordinates("Nose")
        mouth = get_coordinates("Mouth")

        if None in (left_eye, right_eye, nose, mouth):
            return None

        x_min = min(left_eye[0], right_eye[0], nose[0], mouth[0]) - 80
        y_min = min(left_eye[1], right_eye[1], nose[1], mouth[1]) - 80
        x_max = max(left_eye[0], right_eye[0], nose[0], mouth[0]) + 80
        y_max = max(left_eye[1], right_eye[1], nose[1], mouth[1]) + 80

        return (x_min, y_min, x_max, y_max)
    
    except ET.ParseError:
        print(f"Error: Unable to parse {xml_path}")
        return None

def crop_face(image_path, xml_path):
    """ Crops the face using metadata and returns a 50x50 array. """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    face_region = get_face_region(xml_path)

    if face_region is not None:
        x_min, y_min, x_max, y_max = face_region
        print(f"Face Region: {face_region}")
        
        h, w, _ = img.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        cropped_face = img[y_min:y_max, x_min:x_max]

        # Resize to 50x50
        cropped_face = cv2.resize(cropped_face, (50, 50))

        return cropped_face
    else:
        print(f"Skipping {xml_path} due to missing facial landmarks.")

def process_dataset(save=False):
    """ Processes images and returns a list of (cropped_face, label) for training. """
    dataset = []
    for folder in os.listdir(IMAGE_DIR):
        img_folder = os.path.join(IMAGE_DIR, folder)
        xml_folder = os.path.join(XML_DIR, folder)

        if not os.path.isdir(img_folder):
            continue

        for img_file in os.listdir(img_folder):
            if img_file.endswith(".jpg"):  
                img_path = os.path.join(img_folder, img_file)
                xml_path = os.path.join(xml_folder, img_file.replace(".jpg", ".xml"))

                if os.path.exists(xml_path):
                    face = crop_face(img_path, xml_path)
                    if face is not None:
                        dataset.append((face, folder))  

                        if save:
                            person_folder = os.path.join(OUTPUT_DIR, folder)
                            os.makedirs(person_folder, exist_ok=True)
                            output_path = os.path.join(person_folder, img_file)
                            cv2.imwrite(output_path, face)

    return dataset

# Load cropped images and save them in organized folders
data = process_dataset(save=True)

print(f"✅ تمت معالجة {len(data)} صورة.", f" ,إجمالي حجم البيانات {sys.getsizeof(data)} بايت")
