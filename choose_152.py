import os
import shutil

# المسارات إلى مجلدات train و test
train_dir = "data/colorferet/converted_images/train"
test_dir = "data/colorferet/converted_images/test"

def get_class_counts(directory):
    """ إرجاع قاموس يحتوي على عدد الصور لكل شخص """
    class_counts = {}
    
    for person in os.listdir(directory):
        person_path = os.path.join(directory, person)
        if os.path.isdir(person_path):
            class_counts[person] = len(os.listdir(person_path))  # عد الصور
    
    return class_counts

# حساب عدد الصور لكل شخص في train و test
train_counts = get_class_counts(train_dir)
test_counts = get_class_counts(test_dir)

# الأشخاص المشتركين فقط بين train و test
common_people = set(train_counts.keys()) & set(test_counts.keys())

# حساب العدد الإجمالي للصور لكل شخص مشترك
total_counts = {person: train_counts[person] + test_counts[person] for person in common_people}

# اختيار أفضل 152 شخصًا من حيث عدد الصور
top_152_classes = {person for person, _ in sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:152]}

print(f"عدد الفئات المختارة: {len(top_152_classes)}")

def delete_unwanted_classes(directory, allowed_classes):
    """ حذف المجلدات التي لا تنتمي إلى الفئات المسموح بها """
    for person in os.listdir(directory):
        person_path = os.path.join(directory, person)
        if os.path.isdir(person_path) and person not in allowed_classes:
            print(f"حذف {person_path}")
            shutil.rmtree(person_path)  # حذف المجلد بالكامل

# حذف الفئات غير المختارة
delete_unwanted_classes(train_dir, top_152_classes)
delete_unwanted_classes(test_dir, top_152_classes)

print("✅ تمت العملية بنجاح!")
