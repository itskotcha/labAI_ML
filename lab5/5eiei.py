import os
import glob
import time
import random
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 1. ตั้งค่าระบบและตัวแปรเริ่มต้น
# ==========================================
IMAGE_SIZE = (128, 128)
SAMPLES_PER_CLASS = 1000   # จำนวนภาพที่ใช้เทรนต่อคลาส
MODEL_RANDOM_STATE = 42    # ล็อก seed เฉพาะตอนเทรนโมเดล เพื่อให้ผลลัพธ์แม่นยำคงที่
CLASS_NAMES = ['With_Mask (สวมหน้ากาก)', 'Without_Mask (ไม่สวมหน้ากาก)']

print("=" * 65)
print(" 🚀 เริ่มต้นระบบตรวจจับการสวมหน้ากากอนามัยด้วย Support Vector Machines (SVM)")
print("=" * 65)
start_time_total = time.time()

# ==========================================
# 📥 2. ดาวน์โหลดและจัดเตรียม Dataset จาก Kaggle
# ==========================================
KAGGLE_DATASET = "omkargurav/face-mask-dataset"
print(f"\n[ 🌐 ] กำลังตรวจสอบ/ดาวน์โหลดชุดข้อมูล: {KAGGLE_DATASET}...")

dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)

valid_extensions = ('.jpg', '.jpeg', '.png')

# ค้นหาไฟล์ภาพ With Mask (สวมหน้ากาก) และ Without Mask (ไม่สวมหน้ากาก)
with_mask_files = sorted([
    f for f in glob.glob(os.path.join(dataset_dir, "**", "with_mask", "*.*"), recursive=True)
    if f.lower().endswith(valid_extensions)
])
without_mask_files = sorted([
    f for f in glob.glob(os.path.join(dataset_dir, "**", "without_mask", "*.*"), recursive=True)
    if f.lower().endswith(valid_extensions)
])

if len(with_mask_files) == 0 or len(without_mask_files) == 0:
    raise FileNotFoundError("ไม่พบไฟล์ภาพ กรุณาตรวจสอบว่าดาวน์โหลด Dataset ครบถ้วน")

print(f"[ 📦 ] ค้นพบภาพสวมหน้ากาก (With Mask): {len(with_mask_files)} ภาพ")
print(f"[ 📦 ] ค้นพบภาพไม่สวมหน้ากาก (Without Mask): {len(without_mask_files)} ภาพ")

# แยกส่วนสำหรับเทรน (คลาสละ 1,000 ภาพ)
train_mask_files = with_mask_files[:SAMPLES_PER_CLASS]
train_nomask_files = without_mask_files[:SAMPLES_PER_CLASS]

# ส่วนที่เหลือเก็บไว้ใน Pool สำหรับ "สุ่มทดสอบแสดงผลใหม่ทุกครั้งที่กดรัน"
pool_mask_files = with_mask_files[SAMPLES_PER_CLASS:]
pool_nomask_files = without_mask_files[SAMPLES_PER_CLASS:]

# ==========================================
# 🔄 3. ฟังก์ชันสกัดฟีเจอร์ด้วย HOG
# ==========================================
def extract_hog_features(file_paths, label):
    features, valid_paths = [], []
    for path in file_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            try:
                img_resized = cv2.resize(img, IMAGE_SIZE)
                feat = hog(
                    img_resized,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                )
                features.append((feat, label))
                valid_paths.append(path)
            except Exception:
                continue
    return features, valid_paths

print(f"\n[ 🔄 ] กำลังสกัดฟีเจอร์รูปทรงใบหน้าด้วย HOG (คลาสละ {SAMPLES_PER_CLASS} ภาพ)...")
start_time_load = time.time()

data_mask, _ = extract_hog_features(train_mask_files, 0)
data_nomask, _ = extract_hog_features(train_nomask_files, 1)

all_data = data_mask + data_nomask
X = np.array([item[0] for item in all_data])
y = np.array([item[1] for item in all_data])

print(f"[ ✅ ] สกัดฟีเจอร์เสร็จสิ้น: {len(X)} ภาพ (มิติเดิม: {X.shape[1]} มิติ) ใช้เวลา: {time.time() - start_time_load:.2f} วินาที")

# แบ่ง Train/Test 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=MODEL_RANDOM_STATE, stratify=y
)

# ปรับมาตรฐานข้อมูล (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ลดมิติข้อมูลด้วย PCA เพื่อเพิ่มความเร็วในการหาระนาบของ SVM
pca = PCA(n_components=120, random_state=MODEL_RANDOM_STATE)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(f"[ 📉 ] ลดมิติด้วย PCA เหลือ: {X_train.shape[1]} มิติ (อธิบายความแปรปรวนสะสมได้ {pca.explained_variance_ratio_.sum()*100:.1f}%)")

# ==========================================
# 🧠 4. ฝึกและทดสอบโมเดล Support Vector Machines (SVM)
# ==========================================
print(f"\n{'=' * 65}")
print("[ 📈 ] กำลังฝึกโมเดล Support Vector Classifier (SVM: Kernel RBF)...")
print(f"{'=' * 65}")

# RBF Kernel ช่วยแปลงข้อมูลให้อยู่ในมิติที่สูงขึ้นเพื่อหาสันปันเขต (Hyperplane) ที่ดีที่สุด
svm_model = SVC(kernel='rbf', C=3.0, gamma='scale', probability=True, random_state=MODEL_RANDOM_STATE)

start_time_train = time.time()
svm_model.fit(X_train, y_train)
train_time = time.time() - start_time_train

y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"[ ✅ ] ฝึกโมเดลสำเร็จ (ใช้เวลาฝึก: {train_time:.2f} วินาที)")
print(f"[ 🎯 ] ความแม่นยำรวมของ SVM (Accuracy): {acc * 100:.2f}%\n")

print("--- ตารางความสับสน (Confusion Matrix) ---")
print("ทำนายผล ->          [With Mask (0)]  [Without Mask (1)]")
print(f"ของจริง With Mask    : {cm[0][0]:<16} {cm[0][1]:<16}")
print(f"ของจริง Without Mask : {cm[1][0]:<16} {cm[1][1]:<16}\n")

print("--- รายงานการประเมินผล (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ==========================================
# 📸 5. สุ่มทดสอบภาพใหม่ 6 ภาพ (เปลี่ยนรูปใหม่ทุกครั้งที่กดรัน)
# ==========================================
print("=" * 65)
print("[ 🔍 ] ทดสอบโมเดล SVM กับภาพใหม่ที่สุ่มมาสด ๆ (รูปจะเปลี่ยนใหม่ทุกครั้งที่รัน)")
print("=" * 65)

random.seed(time.time())

test_mask_sample = random.sample(pool_mask_files, 3)
test_nomask_sample = random.sample(pool_nomask_files, 3)

test_demo_files = test_mask_sample + test_nomask_sample
demo_true_labels = [0, 0, 0, 1, 1, 1]

demo_features_list, valid_demo_paths, valid_labels = [], [], []
for path, label in zip(test_demo_files, demo_true_labels):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None and img.size > 0:
        try:
            img_resized = cv2.resize(img, IMAGE_SIZE)
            feat = hog(
                img_resized,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )
            demo_features_list.append(feat)
            valid_demo_paths.append(path)
            valid_labels.append(label)
        except Exception:
            continue

demo_features = np.array(demo_features_list)
demo_features = scaler.transform(demo_features)
demo_features = pca.transform(demo_features)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("SVM Predictions: Face Mask Detection (Random Test)", fontsize=14, fontweight='bold')

for i in range(len(demo_features)):
    feat_vector = demo_features[i].reshape(1, -1)
    pred_idx = svm_model.predict(feat_vector)[0]
    prob = svm_model.predict_proba(feat_vector)[0]
    confidence = np.max(prob) * 100

    actual_text = CLASS_NAMES[valid_labels[i]].split()[0]
    pred_text = CLASS_NAMES[pred_idx].split()[0]
    is_correct = (pred_idx == valid_labels[i])
    status_text = "[CORRECT]" if is_correct else "[INCORRECT]"

    img_bgr = cv2.imread(valid_demo_paths[i])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ax = axes[i // 3, i % 3]
    ax.imshow(img_rgb)
    ax.set_title(
        f"True: {actual_text}\nPred: {pred_text} ({confidence:.1f}%)\n{status_text}",
        fontsize=11,
        color="darkgreen" if is_correct else "red",
        fontweight='bold'
    )
    ax.axis('off')

plt.tight_layout()
print(f"[ ⏱️ ] เวลาประมวลผลรวมทั้งหมด: {time.time() - start_time_total:.2f} วินาที")
plt.show()