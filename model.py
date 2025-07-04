import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚀 نموذج CNN المطابق للمواصفات
class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes=152):
        super(FaceRecognitionCNN, self).__init__()

        # ✅ الكتلة الأولى: Conv → BatchNorm → ReLU → MaxPool
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ✅ الكتلة الثانية: Conv → BatchNorm → ReLU → MaxPool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # ✅ الكتلة الثالثة: Conv → BatchNorm → ReLU (بدون MaxPool)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # ✅ حساب حجم الميزات بعد الطبقات الالتفافية
        self.fc1 = nn.Linear(64 * 12 * 12, 500)  # 12x12 بعد تقليل الأبعاد بـ MaxPool
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # الكتلة الأولى
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # الكتلة الثانية
        x = torch.relu(self.bn3(self.conv3(x)))  # الكتلة الثالثة (بدون MaxPool)

        x = torch.flatten(x, start_dim=1)  # تحويل المصفوفة إلى شكل مناسب لـ FC
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # الطبقة الأخيرة بدون ReLU (لأن CrossEntropy بيستخدم Softmax تلقائيًا)

        return x
