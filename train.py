import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FaceRecognitionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 تحميل النموذج
model = FaceRecognitionCNN(num_classes=152).to(device)

# ✅ التحويلات المطلوبة للصور
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ✅ تحميل البيانات
train_data = datasets.ImageFolder(root="data/colorferet/converted_images/train", transform=transform)
test_data = datasets.ImageFolder(root="data/colorferet/converted_images/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ✅ إعداد دالة الخسارة والمُحسّن
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ حفظ أفضل نموذج
best_accuracy = 0.0
save_path = "best_model.pth"

# 🚀 بدء التدريب
num_epochs = 20  # عدد الحقب

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ✅ تقييم النموذج بعد كل حقبة
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    # ✅ حفظ النموذج الأفضل فقط
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), save_path)
        print("🎯 أفضل نموذج تم حفظه ✅")

print("🚀 التدريب انتهى بنجاح! 🎉")

# --- Performance Evaluation ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Put model in evaluation mode
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.exp(outputs)  # convert log_softmax to probability
        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds, labels=list(range(152)))
unique_labels = np.unique(all_labels)  # actual label values in validation

# Plot the confusion matrix with colored pixels
plt.figure(figsize=(16, 14))  # Adjust figure size to ensure proper alignment
sns.heatmap(cm, cmap='Blues', cbar=True, square=True, annot=False,
            xticklabels=range(1, 153), yticklabels=range(1, 153),
            linewidths=0.05, linecolor='gray', cbar_kws={'shrink': 0.8})

# Title and axis labels
plt.title('Confusion Matrix for All 152 Classes')
plt.xlabel('Predicted')
plt.ylabel('True')

# Adjust axis ticks properly
plt.xticks(ticks=np.arange(0, 153, 8), labels=range(0, 153, 8), rotation=90)  # Rotate xticks for clarity
plt.yticks(ticks=np.arange(0, 153, 8), labels=range(0, 153, 8), rotation=0)  # Keep yticks horizontal

# Adjust plot layout to prevent axis cut-off
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

plt.show()


# --- ROC & AUC ---
y_true = label_binarize(all_labels, classes=list(range(152)))

auc_scores = []
for i in range(152):
    try:
        auc = roc_auc_score(y_true[:, i], all_probs[:, i])
        auc_scores.append(auc)
    except ValueError:
        auc_scores.append(np.nan)

mean_auc = np.nanmean(auc_scores)
print(f"Mean AUC: {mean_auc:.4f}")

# Optional: Plot ROC curve for 3 sample classes
for i in [0, 1, 2]:
    try:
        fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_scores[i]:.2f})")
    except ValueError:
        continue

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Sample Classes')
plt.legend()
plt.show()