# Face Recognition System

This repository provides a **complete end-to-end pipeline** for face recognition—from data preprocessing, through model training, to an easy-to-use GUI for prediction and evaluation.

---

## Repository Structure

```
.
├── cropping.py       # Crop faces using ground-truth XML annotations
├── train_test.py     # Train/test split of cropped faces
├── choose_152.py     # (Optional) keep only the top 152 identities
├── model.py          # CNN architecture definition
├── train.py          # Training loop
└── predict.py        # Tkinter GUI for face classification
```

---

## Requirements

### Software

| Tool | Version / Notes |
|------|-----------------|
| **Python** | 3.7+ |
| **Packages** | Install all required packages:<br/>  
```bash
pip install opencv-python numpy scikit-learn torch torchvision pillow tkinterdnd2
``` |

### Hardware

- **CUDA-enabled GPU (recommended)** – used automatically if available  
- **Minimum 8 GB RAM** for training

---

## Usage Pipeline

### 1. Prepare the Dataset

Organize your dataset as follows:

```
data/
├── images/
│   ├── 001/               # One folder per person
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── 002/
└── ground_truths/
    ├── 001/               # XML annotations per image
    │   ├── image1.xml
    │   └── image2.xml
    └── 002/
```

### 2. Run the Pipeline

```bash
# Crop faces using XML annotations
python cropping.py

# Split dataset (80% train / 20% test)
python train_test.py

# (Optional) Keep top 152 classes only
python choose_152.py

# Train the CNN model
python train.py

# Launch the GUI for predictions
python predict.py
```

---

## Key Features

### cropping.py
- Uses OpenCV for image loading and processing  
- Parses XML annotations using `xml.etree.ElementTree`  
- Crops and saves detected face regions  

### train_test.py
- Uses stratified splitting with `sklearn.model_selection`  
- Maintains directory structure during the split  

### model.py
- PyTorch-based CNN architecture:
```python
class FaceRecognitionCNN(nn.Module):
    def __init__(self): ...
    def forward(self, x): ...
```

### train.py
- Automatically detects GPU/CPU
- Saves training checkpoints
- Uses typical training setup:
```python
transform = transforms.Compose([...])
train_loader = DataLoader(...)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(...)
```

### predict.py
- Tkinter GUI with drag-and-drop support
- Supports file selection, face detection, and model prediction

---

## Performance Notes

- Automatically selects computation device:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- For 152-class configuration:
  - **Recommended batch size**: 32–64  
  - **Suggested image size**: 50×50 pixels

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **XML parsing error** | Ensure your `.xml` files are properly formatted. |
| **CUDA out of memory** | Lower the batch size in `train.py`. |
| **Tkinter module error** | On Linux, install Tkinter with:<br/>  
```bash
sudo apt-get install python3-tk
```

---

## License

This project is licensed under the **MIT License**.

---

## Contact

For support or questions, please open an issue or contact:  
📧 **zeyadsheeref@gmail.com**
