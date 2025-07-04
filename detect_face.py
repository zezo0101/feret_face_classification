import cv2
import xml.etree.ElementTree as ET
import os
import torch
from datetime import datetime
from facenet_pytorch import MTCNN

# تحميل كاشف الوجه MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def generate_xml(img_path, img, boxes, xml_save_path, person_id=""):
    """Generate and save XML from detection results."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    root = ET.Element("Recordings")
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    recording = ET.SubElement(root, "Recording", id=f"cfrR{img_name}")

    ET.SubElement(recording, "URL", root="Disc1", relative=img_path)

    current_time = datetime.now()
    ET.SubElement(recording, "CaptureDate").text = current_time.strftime("%m/%d/%Y")
    ET.SubElement(recording, "CaptureTime").text = current_time.strftime("%H:%M:%S")

    ET.SubElement(recording, "Format", value="jpg", scanning="Progressive", compression="None")

    subject = ET.SubElement(recording, "Subject", id=f"cfrS{person_id}" if person_id else "Unknown")
    application = ET.SubElement(subject, "Application")
    face_element = ET.SubElement(application, "Face")

    for box in boxes:
        x, y, x2, y2 = map(int, box)
        w, h = x2 - x, y2 - y

        ET.SubElement(face_element, "Pose", name="fa", yaw="0", pitch="0", roll="0")
        ET.SubElement(face_element, "Wearing", glasses="Unknown")
        ET.SubElement(face_element, "Hair", beard="Unknown", mustache="Unknown", source="Retrospectively")
        ET.SubElement(face_element, "Expression", name="Neutral")

        ET.SubElement(face_element, "LeftEye", x=str(x + int(w * 0.7)), y=str(y + int(h * 0.35)))
        ET.SubElement(face_element, "RightEye", x=str(x + int(w * 0.3)), y=str(y + int(h * 0.35)))
        ET.SubElement(face_element, "Nose", x=str(x + int(w * 0.5)), y=str(y + int(h * 0.5)))
        ET.SubElement(face_element, "Mouth", x=str(x + int(w * 0.5)), y=str(y + int(h * 0.75)))

    xml_tree = ET.ElementTree(root)
    xml_tree.write(xml_save_path, encoding="utf-8", xml_declaration=True)
    print(f"✔ XML saved to: {xml_save_path}")

def detect(image_folder, xml_output_folder=None, single_image=True):
    if single_image:
        # Single image detection
        img_path = image_folder  # here image_folder is actually the image path
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not load image: {img_path}")
            return

        boxes, _ = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if boxes is None or len(boxes) == 0:
            print(f"❌ No face detected in {img_path}")
            return

        # Save XML next to image
        img_dir = os.path.dirname(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_save_path = os.path.join(img_dir, f"{img_name}_xml.xml")
        generate_xml(img_path, img, boxes, xml_save_path)

    else:
        # Multi-image detection
        if not os.path.exists(xml_output_folder):
            os.makedirs(xml_output_folder, exist_ok=True)

        for person in os.listdir(image_folder):
            person_path = os.path.join(image_folder, person)
            if os.path.isdir(person_path):
                person_xml_folder = os.path.join(xml_output_folder, person)
                os.makedirs(person_xml_folder, exist_ok=True)

                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"❌ Couldn't load {img_path}")
                        continue

                    boxes, _ = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if boxes is None or len(boxes) == 0:
                        print(f"❌ No face detected in {img_name}")
                        continue

                    xml_file_path = os.path.join(person_xml_folder, f"{os.path.splitext(img_name)[0]}.xml")
                    generate_xml(img_path, img, boxes, xml_file_path, person_id=person)
