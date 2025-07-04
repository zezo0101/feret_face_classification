import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# مسار مجلد الصور بعد القص
data_dir = "data/colorferet/converted_images/cropped_faces"

# تحديد مجلدات التخزين
train_dir = "data/colorferet/converted_images/train"
test_dir = "data/colorferet/converted_images/test"

# إنشاء المجلدات الرئيسية لو مش موجودة
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# تجميع الصور حسب كل شخص
person_dict = {}

# البحث في كل فولدر فرعي داخل cropped_faces
for person in os.listdir(data_dir):
    person_folder = os.path.join(data_dir, person)
    
    if os.path.isdir(person_folder):  # تأكيد أنه فولدر وليس ملف
        images = glob.glob(os.path.join(person_folder, "*.jpg"))
        
        if images:  # لو فيه صور داخل الفولدر
            person_dict[person] = images

# تقسيم الصور لكل شخص
for person, images in person_dict.items():
    if len(images) < 2:  # لو الشخص عنده صورة واحدة فقط، نحطها في التدريب
        train_folder = os.path.join(train_dir, person)
        os.makedirs(train_folder, exist_ok=True)
        for img in images:
            shutil.move(img, os.path.join(train_folder, os.path.basename(img)))
    else:
        train, test = train_test_split(images, test_size=0.1, random_state=42)

        # إنشاء فولدرات الشخص جوا مجلدات train و test
        train_folder = os.path.join(train_dir, person)
        test_folder = os.path.join(test_dir, person)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # نقل الصور
        for img in train:
            shutil.move(img, os.path.join(train_folder, os.path.basename(img)))
        for img in test:
            shutil.move(img, os.path.join(test_folder, os.path.basename(img)))

print(f"Total Persons: {len(person_dict)}")
print(f"Total Images: {sum(len(v) for v in person_dict.values())}")
print("✅ Done!")
