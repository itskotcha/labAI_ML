import os
import cv2
import numpy as np
import glob
import time
import kagglehub
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- ⚙️ 1. การตั้งค่าระบบ ---
IMAGE_SIZE = (64, 64)
TRAIN_LIMIT = 800  # จำนวนรูปต่อคลาสที่จะใช้เทรน
CLASS_NAMES = ['Cat', 'Dog']

print("="*60)
print(" 🚀 เริ่มโปรแกรม: จำแนก แมว/สุนัข (SVM, KNN, Decision Tree)")
print("="*60)
start_time_total = time.time()

# --- 📥 2. ดาวน์โหลดและจัดการข้อมูลจาก Kaggle ---
KAGGLE_DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"
print(f"\n[ 🌐 ] กำลังตรวจสอบ/ดาวน์โหลดชุดข้อมูล {KAGGLE_DATASET}...")

download_path = kagglehub.dataset_download(KAGGLE_DATASET)
extract_path = os.path.join(download_path, "petimages")

if not os.path.exists(extract_path):
    print(f"[ ⏳ ] กำลังแตกไฟล์ไปยัง: {extract_path}...")
    with zipfile.ZipFile(os.path.join(download_path, "archive.zip"), 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("[ ✅ ] แตกไฟล์สำเร็จ")

cat_folder = os.path.join(extract_path, 'Cat')
dog_folder = os.path.join(extract_path, 'Dog')

cat_files = glob.glob(os.path.join(cat_folder, '*.jpg'))
dog_files = glob.glob(os.path.join(dog_folder, '*.jpg'))

test_cat_files = cat_files[-3:]
test_dog_files = dog_files[-3:]

train_cat_files = cat_files[:TRAIN_LIMIT]
train_dog_files = dog_files[:TRAIN_LIMIT]

# --- 🔄 3. ฟังก์ชันโหลดและแปลงฟีเจอร์รูปภาพ ---
start_time_load = time.time()
def process_images(file_paths, label):
    data, valid_paths = [], []
    for path in file_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            try:
                img_resized = cv2.resize(img, IMAGE_SIZE)
                img_flatten = (img_resized / 255.0).flatten()
                data.append((img_flatten, label))
                valid_paths.append(path)
            except:
                continue
    return data, valid_paths

print(f"\n[ 🔄 ] กำลังโหลดและแปลงภาพ (ฝั่งละ {TRAIN_LIMIT} ภาพ)...")
cat_data, _ = process_images(train_cat_files, 0)
dog_data, _ = process_images(train_dog_files, 1)

all_data = cat_data + dog_data
X = np.array([item[0] for item in all_data])
y = np.array([item[1] for item in all_data])
loading_time = time.time() - start_time_load
print(f"[ ✅ ] โหลดข้อมูลสำเร็จ: {len(X)} ภาพ (เวลา: {loading_time:.2f} วินาที)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 🧠 4. ฝึกและประเมินผล 3 โมเดล ---
models = {
    # 📌 เพิ่ม probability=True เพื่อให้ SVM คำนวณเปอร์เซ็นต์ได้
    "SVM (Kernel: Linear)": SVC(kernel='linear', probability=True, random_state=42), 
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

trained_models = {}

for name, model in models.items():
    print(f"\n\n{'='*55}")
    print(f"[ 📈 ] กำลังฝึกโมเดล {name}...")
    start_time_train = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    training_time = time.time() - start_time_train
    trained_models[name] = model
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"[ ✅ ] ฝึกโมเดลเสร็จสิ้น (เวลา: {training_time:.2f} วินาที)")
    
    print("\n--- ตารางความสับสน (Confusion Matrix) ---")
    print("ทำนาย: [Cat (0)] [Dog (1)]")
    print(f"จริง Cat (0): {cm[0][0]:<4} (TN) | {cm[0][1]:<4} (FP)")
    print(f"จริง Dog (1): {cm[1][0]:<4} (FN) | {cm[1][1]:<4} (TP)")
    
    print("\n--- รายงานการจำแนกประเภท (Classification Report) ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    print(f"[ 🎯 ] ความแม่นยำรวม (Accuracy) ของ {name.split(' ')[0]}: {acc:.4f}")

# --- 📸 5. ทดสอบรูปภาพใหม่ 6 รูป (แมว 3 / สุนัข 3) แบบโชว์ % ---
print("\n\n" + "="*55)
print("[ 🔍 ] ทดสอบโมเดลกับภาพใหม่ (พร้อมเปอร์เซ็นต์ความมั่นใจ)")
print("="*55)

test_files = test_cat_files + test_dog_files
test_labels_true = [0, 0, 0, 1, 1, 1]

test_data, valid_test_paths = process_images(test_files, -1)

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
fig.suptitle("Model Predictions with Confidence (%): SVM | KNN | Decision Tree", fontsize=16)

for i, (features, dummy_lbl) in enumerate(test_data):
    true_label = CLASS_NAMES[test_labels_true[i]]
    img_name = os.path.basename(valid_test_paths[i])
    
    print(f"\nรูปที่ {i+1}: {img_name} (เฉลยจริง: {true_label})")
    
    preds_text = ""
    for name, model in trained_models.items():
        input_feature = features.reshape(1, -1)
        
        # 📌 ทำนายคลาส และ ดึงค่าความน่าจะเป็น (Probability)
        pred_val = model.predict(input_feature)[0]
        pred_label = CLASS_NAMES[pred_val]
        
        # ดึงเปอร์เซ็นต์ความมั่นใจของคลาสที่ทายออกมา
        proba = model.predict_proba(input_feature)[0]
        confidence = np.max(proba) * 100 
        
        result_mark = "✅" if pred_label == true_label else "❌"
        short_name = name.split()[0] 
        
        # โชว์ผล + % ใน Console
        print(f"    - {short_name:15} ทายว่า -> {pred_label:3} ({confidence:5.1f}%) {result_mark}")
        
        # เก็บข้อความไปแปะใต้รูปภาพ (โชว์ %)
        preds_text += f"{short_name}: {pred_label} ({confidence:.1f}%)\n"
        
    # พล็อตรูปภาพ
    ax = axes[i // 3, i % 3]
    display_img = cv2.imread(valid_test_paths[i])
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    
    ax.imshow(display_img)
    ax.set_title(f"True: {true_label}", color='blue')
    ax.set_xlabel(preds_text, fontsize=11, loc='left')
    ax.set_xticks([])
    ax.set_yticks([])

# --- ⏱️ สรุปเวลา ---
print("\n" + "="*55)
total_execution_time = time.time() - start_time_total
print(f"[ ⏱️ ] เวลาที่ใช้ในการทำงานรวมทั้งหมด: {total_execution_time:.2f} วินาที")
print("="*55)

print("\n[ ✅ ] โปรแกรมทำงานเสร็จสมบูรณ์! กำลังเปิดหน้าต่างรูปภาพ...")
plt.tight_layout(h_pad=3.5) # เผื่อที่ว่างแนวตั้งให้ข้อความไม่ทับรูป
plt.show()