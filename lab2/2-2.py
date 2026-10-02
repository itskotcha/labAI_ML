import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import kagglehub
import zipfile
import glob
import time

# --- การตั้งค่าระบบและการแสดงผล ---
font_name = 'Tahoma'
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# --- การตั้งค่าชุดข้อมูล ---
KAGGLE_DATASET = "jangedoo/utkface-new"
IMAGE_SIZE = 128
SAMPLE_LIMIT = 500  # จำกัด 500 ภาพเพื่อความรวดเร็วในการเทรน
SUBDIR = "UTKFace"  # ชื่อโฟลเดอร์หลังแตกไฟล์

# --- 1. ดาวน์โหลดและเตรียมข้อมูลจาก Kaggle Hub (Global) ---
print(f"[*] กำลังดาวน์โหลดชุดข้อมูล {KAGGLE_DATASET} จาก Kaggle Hub...")
start_time_total = time.time()

try:
    # ดาวน์โหลดไฟล์ .zip ของชุดข้อมูล
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[✓] ดาวน์โหลดสำเร็จ. พาธหลัก: {download_path}")
    
    # หารายชื่อไฟล์ในโฟลเดอร์ที่ดาวน์โหลดมา
    files_in_path = os.listdir(download_path)
    
    if len(files_in_path) > 0:
        first_file = files_in_path[0]
        # พาธของไฟล์ (อาจจะเป็น .zip หรือโฟลเดอร์ที่แตกแล้ว)
        file_path = os.path.join(download_path, first_file)
        image_dir = os.path.join(download_path, SUBDIR)
        
        # 1.1 แตกไฟล์ (Unzip) หากยังไม่ได้ทำ และไฟล์นั้นเป็น zip
        if not os.path.exists(image_dir) and first_file.endswith('.zip'):
            print(f"[*] กำลังแตกไฟล์ไปยัง: {image_dir}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print("[✓] แตกไฟล์สำเร็จ.")
        elif not os.path.exists(image_dir):
            # กรณีที่ Kagglehub แตกไฟล์มาให้เป็นโฟลเดอร์แล้ว
            image_dir = download_path
    else:
        image_dir = download_path

except Exception as e:
    print(f"[X] เกิดข้อผิดพลาดในการดาวน์โหลด/แตกไฟล์: {e}")
    exit()

# --- 2. โหลดและเตรียมข้อมูล ---
images = []
age_labels = []

# ใช้ glob เพื่อหาไฟล์ทั้งหมดในโฟลเดอร์ UTKFace
# หาไฟล์ .jpg ทั้งหมดรวมถึงในโฟลเดอร์ย่อย
file_paths = glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)

print(f"[*] กำลังโหลดภาพตัวอย่างสูงสุด ({SAMPLE_LIMIT}) ภาพจาก {len(file_paths)} ไฟล์ที่พบ...")

count = 0
for path in file_paths:
    if count >= SAMPLE_LIMIT:
        break
        
    filename = os.path.basename(path)
    # รูปแบบชื่อไฟล์ UTKFace: [age]_[gender]_[race]_[date].jpg
    try:
        age = int(filename.split("_")[0])
    except ValueError:
        continue
        
    img = cv2.imread(path)
    if img is None:
        continue
        
    # resize และ normalize
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    
    images.append(img.flatten())
    age_labels.append(age)
    count += 1

# แปลงเป็น NumPy Array
X = np.array(images)
y = np.array(age_labels)

print(f"[✓] โหลดข้อมูลสำเร็จ: {len(X)} ภาพ")

if len(X) == 0:
    print("[X] ไม่มีข้อมูลภาพที่ใช้งานได้ (อาจต้องตรวจสอบชื่อไฟล์ในโฟลเดอร์ UTKFace)")
    exit()

# --- 3. แบ่งชุดข้อมูล train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. เทรนโมเดล Linear Regression ---
print("[*] กำลังเทรนโมเดล Linear Regression...")
start_time_train = time.time()

model = LinearRegression()
model.fit(X_train, y_train)

train_time = time.time() - start_time_train
print(f"[✓] เทรนเสร็จแล้ว (เวลา: {train_time:.2f} วินาที)")

# --- 5. ทำนายและประเมินผลลัพธ์ ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- ผลการประเมินประสิทธิภาพโมเดล ---")
print(f"[MAE] ค่าความผิดพลาดเฉลี่ยสัมบูรณ์ (MAE): {mae:.2f} ปี")
print(f"[R2] ความแม่นยำของโมเดล (R-squared): {r2:.4f}")

total_time = time.time() - start_time_total
print("\n--- สรุปเวลาการทำงาน ---")
print(f"เวลาที่ใช้ในการฝึกโมเดล (Training Time): {train_time:.2f} วินาที")
print(f"เวลาที่ใช้ในการทำงานรวมทั้งหมด (Total Time): {total_time:.2f} วินาที (ไม่รวมเวลาแสดงกราฟ)")

# --- 6. แสดงภาพพร้อมผลทำนายโดยใช้ Matplotlib ---
num_display = min(10, len(X_test))
fig, axes = plt.subplots(2, num_display // 2, figsize=(15, 6))
fig.suptitle(f"ผลการทำนายอายุด้วย Linear Regression (MAE: {mae:.2f} ปี)", fontsize=16)

axes = axes.flatten()
for i in range(num_display):
    img = X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    img = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    actual = y_test[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    
    axes[i].imshow(img_rgb)
    axes[i].set_title(f"ทำนาย: {predicted:.1f}\nจริง: {actual} (ผิดพลาด: {error:.1f})", fontsize=9)
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()