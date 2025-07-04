# Face Recognition System

This repository contains a complete pipeline for face recognition, from data preprocessing to model training and prediction. The system processes face images, trains a deep learning model, and provides a GUI for face classification.

## Repository Structure

.
├── cropping.py # Script to crop faces using ground truth data
├── train_test.py # Script to split cropped faces into train/test sets
├── choose_152.py # Filters dataset to keep only the best 152 persons
├── model.py # Deep learning model architecture
├── train.py # Training script
└── predict.py # GUI for face classification


## Requirements

- Python 3.7+
- Essential Packages:
  ```bash
  pip install opencv-python numpy scikit-learn torch torchvision pillow tkinterdnd2

Hardware Requirements
CUDA-enabled GPU recommended (script automatically uses GPU if available)

Minimum 8GB RAM for training

Usage Pipeline
1. Data Preparation
Organize your dataset with this structure:

data/
├── images/
│   ├── 001/               # Each numbered folder represents one person
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── 002/
└── ground_truths/
│   ├── 001/               # XML annotations matching images in folders
│   │   ├── image1.xml
│   │   └── image2.xml
│   └── 002/

2. Run the Pipeline

# Crop faces using XML annotations
python cropping.py

# Split dataset (80% train, 20% test)
python train_test.py

# Optional: Filter to best 152 classes
python choose_152.py

# Train the model (automatically uses GPU if available)
python train.py

# Launch prediction GUI
python predict.py

Key Features
cropping.py
.Uses OpenCV for image processing

.Parses XML annotations with ElementTree

.Handles batch processing of face crops

train_test.py
.Implements stratified splitting with sklearn

.Maintains directory structure during split

model.py
PyTorch CNN implementation

Includes:
class FaceRecognitionCNN(nn.Module):
    # CNN architecture definition

train.py
Features:

.Automatic GPU/CPU detection

.Model checkpointing

Training workflow:
transform = transforms.Compose([...])
train_loader = DataLoader(...)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(...)

predict.py
Tkinter-based GUI with drag-and-drop support

Supports:

.Image file selection

.face detection

.Model inference

Performance Notes
The system automatically detects CUDA devices:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

For optimal performance with 152 classes:

.Recommended batch size: 32-64

.Suggested image size: 50x50

License
MIT License

Troubleshooting
Common Issues:

XML parsing errors: Verify your annotation files are valid XML

CUDA out of memory: Reduce batch size in train.py

Tkinter errors: Install required packages:

sudo apt-get install python3-tk  # For Linux
Contact
For support, please open an issue or contact zeyadsheeref@gmail.com
