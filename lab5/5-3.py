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
from sklearn.metrics import accuracy_score

# --- ⚙️ 1. การตั้งค่าระบบ ---
IMAGE_SIZE = (64, 64)
TRAIN_LIMIT = 800  # จำนวนรูปต่อคลาสที่จะใช้เทรน
CLASS_NAMES = ['Cat', 'Dog']

print("="*60)
print(" 🚀 เริ่มโปรแกรม: จำแนก แมว/สุนัข (แสดงผลแบบ 3x2 Grid)")
print("="*60)

# --- 📥 2. ดาวน์โหลดและจัดการข้อมูลจาก Kaggle ---
KAGGLE_DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"
print(f"[ 🌐 ] กำลังตรวจสอบ/ดาวน์โหลดชุดข้อมูล...")

download_path = kagglehub.dataset_download(KAGGLE_DATASET)
extract_path = os.path.join(download_path, "petimages")

if not os.path.exists(extract_path):
    with zipfile.ZipFile(os.path.join(download_path, "archive.zip"), 'r') as zip_ref:
        zip_ref.extractall(extract_path)

cat_folder = os.path.join(extract_path, 'Cat')
dog_folder = os.path.join(extract_path, 'Dog')

cat_files = glob.glob(os.path.join(cat_folder, '*.jpg'))
dog_files = glob.glob(os.path.join(dog_folder, '*.jpg'))

# ใช้รูปสุดท้ายเป็น Test Image สำหรับโชว์ในกราฟ (แมว 1 รูป, หมา 1 รูป)
test_cat_path = cat_files[-1]
test_dog_path = dog_files[-1]

# รูปที่เหลือใช้สำหรับฝึกโมเดล
train_cat_files = cat_files[:TRAIN_LIMIT]
train_dog_files = dog_files[:TRAIN_LIMIT]

# --- 🔄 3. ฟังก์ชันโหลดและแปลงฟีเจอร์รูปภาพ ---
def process_images(file_paths, label):
    data = []
    for path in file_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            try:
                img_resized = cv2.resize(img, IMAGE_SIZE)
                img_flatten = (img_resized / 255.0).flatten()
                data.append((img_flatten, label))
            except:
                continue
    return data

print(f"[ 🔄 ] กำลังโหลดและแปลงภาพ (ฝั่งละ {TRAIN_LIMIT} ภาพ)...")
cat_data = process_images(train_cat_files, 0)
dog_data = process_images(train_dog_files, 1)

all_data = cat_data + dog_data
X = np.array([item[0] for item in all_data])
y = np.array([item[1] for item in all_data])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 🧠 4. ฝึกและประเมินผล 3 โมเดล ---
models = {
    "SVM": SVC(kernel='linear', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

trained_models = {}
model_accuracies = {}

print("\n[ 📈 ] กำลังฝึกโมเดลทั้ง 3 แบบ (อาจใช้เวลาสักครู่)...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    trained_models[name] = model
    model_accuracies[name] = acc
    print(f"  ✅ {name:15} ฝึกเสร็จสิ้น! Accuracy: {acc*100:.2f}%")

# --- 📸 5. สร้างกราฟิกแสดงผลแบบ 3x2 Grid (ให้เหมือนรูปตัวอย่าง) ---
print("\n[ 🖼️ ] กำลังสร้างหน้าต่างแสดงผล...")

# กำหนดรูปที่จะใช้เทสต์และป้ายกำกับ
test_images_info = [
    {"path": test_cat_path, "true_label": "Cat"},
    {"path": test_dog_path, "true_label": "Dog"}
]

# สร้าง Figure 3 แถว 2 คอลัมน์
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
fig.subplots_adjust(wspace=0.05, hspace=0.1) # ปรับให้รูปชิดกันมากขึ้น

# ลูปวาดภาพตามแถว (โมเดล)
for row_idx, (model_name, model) in enumerate(trained_models.items()):
    acc = model_accuracies[model_name]
    
    # ลูปวาดภาพตามคอลัมน์ (แมว/หมา)
    for col_idx, img_info in enumerate(test_images_info):
        path = img_info["path"]
        true_label = img_info["true_label"]
        
        # โหลดภาพสำหรับ Predict
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img_gray, IMAGE_SIZE)
        img_flatten = (img_resized / 255.0).reshape(1, -1)
        
        # ทำนายผล
        pred_idx = model.predict(img_flatten)[0]
        pred_label = CLASS_NAMES[pred_idx]
        
        # โหลดภาพเพื่อแสดงผล (ปรับขนาดให้เท่ากันเพื่อความสวยงาม)
        disp_img = cv2.imread(path)
        disp_img = cv2.resize(disp_img, (500, 350))
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB) # แปลงเป็น RGB เพื่อใช้กับ matplotlib
        
        # 🎨 วาดข้อความทับลงไปบนรูปภาพ (รูปแบบ RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # บรรทัด 1: Prediction (สีเขียว)
        cv2.putText(disp_img, f"{model_name}: {pred_label}", (15, 40), font, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        # บรรทัด 2: Actual (สีฟ้า/Cyan)
        cv2.putText(disp_img, f"Actual: {true_label}", (15, 85), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        # บรรทัด 3: Accuracy (สีขาว)
        cv2.putText(disp_img, f"Accuracy: {acc*100:.2f}%", (15, 125), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # แสดงรูปลงในแกน (Axes)
        ax = axes[row_idx, col_idx]
        ax.imshow(disp_img)
        ax.axis('off') # ปิดสเกลแกน X, Y
        
        # ใส่ป้ายชื่อโมเดลตรงกลาง ระหว่าง 2 รูป
        if col_idx == 0:
            # ใช้ Text วางตำแหน่ง X นอกกรอบรูปภาพเล็กน้อย เพื่อให้อยู่กึ่งกลาง
            ax.text(1.05, 0.5, f"{model_name}", transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', color='white',
                    ha='center', va='center', bbox=dict(facecolor='black', edgecolor='none', pad=10, boxstyle='round,pad=0.3'))

print("[ 🎉 ] เสร็จสมบูรณ์! กำลังเปิดกราฟิก...")
plt.show()