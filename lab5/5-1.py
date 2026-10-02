# svm_cat_dog.py
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import kagglehub
import zipfile
import glob
import time
from matplotlib import font_manager as fm

# --- ⚙️ การตั้งค่าระบบและการแสดงผล ---
font_name = 'Tahoma'
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
IMAGE_SIZE = (64, 64)
SAMPLE_LIMIT = 2000 # จำกัดภาพที่ใช้ฝึก
CLASS_NAMES = ['Cat', 'Dog']

# --- เริ่มจับเวลาทั้งหมดของโปรแกรม ---
start_time_total = time.time()

# --- 1. การจัดการข้อมูล (ดาวน์โหลดและแตกไฟล์จาก Kaggle Hub) ---
KAGGLE_DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"
ZIP_FILE_NAME = "archive.zip"
EXTRACT_DIR_NAME = "petimages"

print(f"[ 🌐 ] กำลังดาวน์โหลดชุดข้อมูล {KAGGLE_DATASET}...")
try:
    # 1.1 ดาวน์โหลดและรับพาร์ท
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    
    # 1.2 แตกไฟล์ archive.zip
    zip_file_path = os.path.join(download_path, ZIP_FILE_NAME)
    extract_path = os.path.join(download_path, EXTRACT_DIR_NAME)
    
    if not os.path.exists(extract_path):
        print(f"[ ⏳ ] กำลังแตกไฟล์ไปยัง: {extract_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("[ ✅ ] แตกไฟล์สำเร็จ.")
        
    cat_folder = os.path.join(extract_path, 'Cat')
    dog_folder = os.path.join(extract_path, 'Dog')
except Exception as e:
    print(f"[ ❌ ] เกิดข้อผิดพลาดในการดาวน์โหลด/แตกไฟล์: {e}")
    exit()

# --- 2. โหลดภาพและเตรียมฟีเจอร์ ---
start_time_load = time.time()
def load_images_from_folder(folder, label, limit, image_size):
    data = []
    for i, path in enumerate(glob.glob(os.path.join(folder, '*.jpg'))):
        if len(data) >= limit:
            break
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None and img.size > 0:
            try:
                img = cv2.resize(img, image_size)
                img_normalized = img / 255.0
                data.append((img_normalized.flatten(), label))
            except cv2.error:
                continue
    return data

print(f"[ 🔄 ] กำลังโหลดภาพ: แมว/สุนัข สูงสุดฝั่งละ {SAMPLE_LIMIT // 2} ภาพ...")
cat_images = load_images_from_folder(cat_folder, label=0, limit=SAMPLE_LIMIT // 2, image_size=IMAGE_SIZE)
dog_images = load_images_from_folder(dog_folder, label=1, limit=SAMPLE_LIMIT // 2, image_size=IMAGE_SIZE)

# รวมข้อมูล
all_data = cat_images + dog_images
if not all_data:
    print("[ ❌ ] ไม่พบข้อมูลภาพที่ใช้งานได้")
    exit()

X = np.array([i[0] for i in all_data])
y = np.array([i[1] for i in all_data])
end_time_load = time.time()
loading_time = end_time_load - start_time_load

print(f"[ ✅ ] โหลดข้อมูลสำเร็จ: {len(X)} ภาพ (เวลา: {loading_time:.2f} วินาที)")

# --- 3. แบ่งข้อมูลและฝึกโมเดล SVM ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[ 📈 ] กำลังฝึกโมเดล SVM (Kernel: Linear)...")
start_time_train = time.time()
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)
end_time_train = time.time()
training_time = end_time_train - start_time_train
print(f"[ ✅ ] ฝึกโมเดลเสร็จสิ้น (เวลา: {training_time:.2f} วินาที)")

# --- 4. ทำนายและแสดงผลลัพธ์เชิงตัวเลข ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- 4.1 ตารางความสับสน (Confusion Matrix) ---")
print("ทำนาย: [Cat (0)] [Dog (1)]")
print(f"จริง Cat (0): {cm[0][0]} (TN) | {cm[0][1]} (FP)")
print(f"จริง Dog (1): {cm[1][0]} (FN) | {cm[1][1]} (TP)")

print("\n--- 4.2 รายงานการจำแนกประเภท (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

print(f"\n[ 🎯 ] ความแม่นยำรวม (Accuracy): {accuracy:.4f}")


# --- 5. แสดงกราฟแนวคิดของ SVM ---
# [โค้ดส่วนแสดงกราฟแนวคิดของ SVM อยู่ที่นี่ (เหมือนเดิม)]
plt.figure(figsize=(9, 7))
rng = np.random.RandomState(42)
X_mock_0 = rng.randn(30, 2) * 0.5 - [1, 1]
X_mock_1 = rng.randn(30, 2) * 0.5 + [1, 1]
X_mock = np.r_[X_mock_0, X_mock_1]
y_mock = np.array([0] * 30 + [1] * 30)
model_mock = SVC(kernel='linear', C=1000)
model_mock.fit(X_mock, y_mock)
plt.scatter(X_mock[:, 0], X_mock[:, 1], c=y_mock, s=80, cmap='coolwarm', alpha=0.9,
            edgecolors='k', linewidths=0.5)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model_mock.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=1, linestyles=['-'], linewidths=3)
ax.contour(XX, YY, Z, colors='blue', levels=[1], alpha=0.7, linestyles=['--'], linewidths=2)
ax.contour(XX, YY, Z, colors='red', levels=[-1], alpha=0.7, linestyles=['--'], linewidths=2)
ax.scatter(model_mock.support_vectors_[:, 0], model_mock.support_vectors_[:, 1], s=200,
           facecolors='none', edgecolors='k', marker='o')
plt.title("แนวคิด SVM: การจำแนก Cat / Dog (ข้อมูลจำลอง 2D)", fontsize=16)
plt.xlabel("ฟีเจอร์ 1 (รูปแบบของภาพ)", fontsize=12)
plt.ylabel("ฟีเจอร์ 2 (รูปแบบของภาพ)", fontsize=12)
plt.legend([plt.Line2D([0], [0], color='k', linestyle='-'),
            plt.Line2D([0], [0], color='blue', linestyle='--'),
            plt.Line2D([0], [0], color='red', linestyle='--'),
            plt.scatter([0], [0], s=200, facecolors='none', edgecolors='k')],
           ['Hyperplane (เส้นแบ่ง)', 'Positive Hyperplane (+1)', 'Negative Hyperplane (-1)', 'Support Vectors'],
           loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# --- 6. ทดสอบภาพใหม่ (sat5.jpg) และแสดงผล ---
TEST_IMAGE_PATH = 'sat5.jpg' # <<< กำหนดชื่อภาพที่ต้องการทำนาย

def predict_and_show_image(image_path, model, image_size=IMAGE_SIZE):
    print(f"\n--- 6.1 กำลังทำนายภาพ: {os.path.basename(image_path)} ---")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"[ ❌ ] ไม่พบไฟล์: {image_path} หรือไฟล์ไม่ถูกต้อง")
        return
        
    # 1. โหลดภาพต้นฉบับ (เพื่อแสดงผลเป็นสี)
    original_img = cv2.imread(image_path)
    if original_img is None:
        original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # ถ้าโหลดสีไม่ได้ ให้แปลงสีเทาเป็น BGR
        
    # 2. ปรับขนาดและเตรียมฟีเจอร์ (เหมือนกับขั้นตอนฝึกโมเดล)
    img_resized = cv2.resize(img, image_size)
    img_normalized = img_resized / 255.0
    img_flatten = img_normalized.flatten().reshape(1, -1)
    
    # 3. ทำนายผล
    prediction = model.predict(img_flatten)[0]
    label = "Dog" if prediction == 1 else "Cat"
    
    # 4. แสดงผลลัพธ์ใน Console
    print(f"[ ✅ ] ผลการทำนาย: {label}")
    
    # 5. แสดงภาพพร้อม Label (ใช้ OpenCV)
    # 5. 1. แสดงผลการทำนาย (บรรทัดแรก)
    cv2.putText(original_img, f"Prediction: {label}", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    # 5.2. แสดงค่า Accuracy (บรรทัดที่สอง)
    # y = 65 pixels (1x ฟอนต์ + ช่องว่าง)
    cv2.putText(original_img, f"Test Accuracy: {accuracy*100:.2f}%", (5, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # สีเหลือง/ฟ้า
                
    cv2.imshow("Prediction Result", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# เรียกใช้ฟังก์ชันทำนายภาพ 'sat5.jpg'
predict_and_show_image(TEST_IMAGE_PATH, model)

# --- สรุปเวลาการทำงาน ---
end_time_total = time.time()
total_execution_time = end_time_total - start_time_total

print("\n--- สรุปเวลาการทำงาน ---")
print(f"[ ⏱️ ] เวลาที่ใช้ในการโหลดและประมวลผลข้อมูล: {loading_time:.2f} วินาที")
print(f"[ ⏱️ ] เวลาที่ใช้ในการฝึกโมเดล SVM: {training_time:.2f} วินาที")
print(f"[ ⏱️ ] เวลาที่ใช้ในการทำงานรวมทั้งหมด: {total_execution_time:.2f} วินาที")