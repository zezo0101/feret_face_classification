import torch
import torchvision.transforms as transforms
from PIL import Image
from model import FaceRecognitionCNN
import os
import xml.etree.ElementTree as ET
import cv2
from detect_face import detect
import tkinter as tk
from tkinter import filedialog, Label, Frame
from tkinterdnd2 import DND_FILES, TkinterDnD


def get_face_region(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        face = root.find(".//Face")
        if face is None:
            return None

        def get_coordinates(tag):
            element = face.find(tag)
            if element is not None:
                return int(element.attrib["x"]), int(element.attrib["y"])
            return None

        coords = [get_coordinates(tag) for tag in ["LeftEye", "RightEye", "Nose", "Mouth"]]
        if None in coords:
            return None

        x_min = min(pt[0] for pt in coords) - 80
        y_min = min(pt[1] for pt in coords) - 80
        x_max = max(pt[0] for pt in coords) + 80
        y_max = max(pt[1] for pt in coords) + 80
        return (x_min, y_min, x_max, y_max)
    except:
        return None

def crop_face(image_path, xml_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    face_region = get_face_region(xml_path)
    if face_region is None:
        return None
    x_min, y_min, x_max, y_max = map(int, face_region)
    h, w, _ = img.shape
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    cropped_face = img[y_min:y_max, x_min:x_max]
    cropped_face = cv2.resize(cropped_face, (50, 50))
    return cropped_face

# Load model
model = FaceRecognitionCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_folder_path = "data/colorferet/converted_images/test"
person_names = sorted(os.listdir(test_folder_path))

def process_image(file_path):
    try:
        img_dir = os.path.dirname(file_path)
        img_name = os.path.splitext(os.path.basename(file_path))[0]
        xml_path = os.path.join(img_dir, f"{img_name}_xml.xml")

        detect(file_path)

        face = crop_face(file_path, xml_path)
        if face is None:
            result_label.config(text="❌ No face detected.")
            return

        rgb_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_person = person_names[predicted_class] if 0 <= predicted_class < len(person_names) else "Unknown"
            print(f"✅ Predicted: {predicted_person}")
            result_label.config(text=f"✅ Predicted: {predicted_person}")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path)

def drop(event):
    file_path = event.data.strip("{}")  # handle drag paths with spaces
    if os.path.isfile(file_path):
        process_image(file_path)

# GUI setup
root = TkinterDnD.Tk()
root.title("🧠 Face Recognition")
root.geometry("600x500")
root.configure(bg="#e6f0ff")

frame = Frame(root, bg="#cce0ff", width=500, height=250, relief="ridge", borderwidth=3)
frame.pack(pady=30)
frame.pack_propagate(False)

drop_label = Label(frame, text="📂 Drag & Drop Image Here", font=("Arial", 16), bg="#cce0ff", fg="#003366")
drop_label.pack(expand=True)
frame.drop_target_register(DND_FILES)
frame.dnd_bind('<<Drop>>', drop)

btn = tk.Button(root, text="📁 Select Image", command=select_image, font=("Arial", 12), bg="#3399ff", fg="white", padx=20, pady=10)
btn.pack()

result_label = Label(root, text="", font=("Arial", 16), bg="#e6f0ff", fg="#003366")
result_label.pack(pady=20)

root.mainloop()
