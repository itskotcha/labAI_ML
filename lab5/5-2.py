import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # ปิดคำเตือนเรื่อง SVC(probability=True) ที่ไม่กระทบผลลัพธ์

import cv2
import numpy as np
import glob
import time
import random
import kagglehub
import zipfile
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- ⚙️ 1. การตั้งค่าระบบ ---
IMAGE_SIZE = (128, 128)   # ขยายขนาดขึ้นเล็กน้อย ให้ HOG จับลายเส้น/ขอบได้ดีขึ้น
TRAIN_LIMIT = 1500        # เพิ่มจำนวนภาพต่อคลาส (ยิ่งเยอะยิ่งช่วย เพราะ raw pixel เดิมมันอ่อนมาก)
RANDOM_STATE = 42
CLASS_NAMES = ['Cat', 'Dog']

random.seed(RANDOM_STATE)

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

cat_files = sorted(glob.glob(os.path.join(cat_folder, '*.jpg')))
dog_files = sorted(glob.glob(os.path.join(dog_folder, '*.jpg')))

# สุ่มเลือกภาพทดสอบ 3+3 ภาพ "แยกออก" จากชุดเทรนจริง ๆ (ของเดิมสุ่มจากท้ายไฟล์เฉย ๆ ซึ่งก็โอเค
# แต่ทำให้ชื่อไฟล์ซ้ำกันระหว่าง Cat/Dog จนดูสับสนตอนอ่านผล เลยสุ่มแบบมี seed แทน)
test_cat_files = random.sample(cat_files[TRAIN_LIMIT:], 3)
test_dog_files = random.sample(dog_files[TRAIN_LIMIT:], 3)

train_cat_files = cat_files[:TRAIN_LIMIT]
train_dog_files = dog_files[:TRAIN_LIMIT]

# --- 🔄 3. ฟังก์ชันโหลดภาพ + สกัดฟีเจอร์แบบ HOG ---
# ⚠️ จุดที่ทำให้ผลเดิมแม่นยำแค่ ~50-57% (ใกล้เดาสุ่ม) คือการใช้ "ค่าพิกเซลดิบ" (flatten)
# เป็นฟีเจอร์ตรง ๆ ซึ่งไม่มีความหมายเชิงรูปทรง/ลาย ให้โมเดล classical ML เรียนรู้ได้ดีพอ
# แก้โดยเปลี่ยนไปใช้ HOG (Histogram of Oriented Gradients) ซึ่งจับ "ขอบ/ทิศทางลาย"
# ของภาพได้ดีกว่ามาก และเป็นฟีเจอร์มาตรฐานที่ SVM/KNN/Decision Tree ใช้ได้ผลจริง
start_time_load = time.time()
def process_images(file_paths, label):
    data, valid_paths = [], []
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
                data.append((feat, label))
                valid_paths.append(path)
            except:
                continue
    return data, valid_paths

print(f"\n[ 🔄 ] กำลังโหลดและแปลงภาพเป็นฟีเจอร์ HOG (ฝั่งละ {TRAIN_LIMIT} ภาพ)...")
cat_data, _ = process_images(train_cat_files, 0)
dog_data, _ = process_images(train_dog_files, 1)

all_data = cat_data + dog_data
X = np.array([item[0] for item in all_data])
y = np.array([item[1] for item in all_data])
loading_time = time.time() - start_time_load
print(f"[ ✅ ] โหลดข้อมูลสำเร็จ: {len(X)} ภาพ, ฟีเจอร์ {X.shape[1]} มิติ (เวลา: {loading_time:.2f} วินาที)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# --- 📏 มาตรฐานฟีเจอร์ (สำคัญมากสำหรับ SVM/KNN ที่อิงระยะห่าง/ผลคูณจุด) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 📉 ลดมิติด้วย PCA ---
# ⚠️ บั๊กที่ทำให้ KNN ทายเอียงไปทาง Dog เกือบหมด (recall Cat แค่ 3%) คือ "curse of dimensionality":
# HOG ให้ฟีเจอร์ถึง 8100 มิติ ในมิติสูงขนาดนี้ระยะห่างแบบ Euclidean ของทุกจุดจะเริ่มใกล้เคียงกันหมด
# (แทบวัดความต่างไม่ได้จริง) พอบวกกับ weights='distance' ความต่างเล็กจิ๋วที่เหลือถูกขยายจนเอนเอียง
# ผิดปกติไปทางคลาสเดียว แก้โดยลดมิติเหลือ ~150 องค์ประกอบหลักด้วย PCA ก่อนส่งเข้าโมเดล
# (ช่วย KNN มากที่สุด และทำให้ SVM/Decision Tree เทรนเร็วขึ้นด้วย)
from sklearn.decomposition import PCA
pca = PCA(n_components=150, random_state=RANDOM_STATE)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(f"[ 📉 ] ลดมิติด้วย PCA: {X.shape[1]} -> {X_train.shape[1]} มิติ "
      f"(รักษาความแปรปรวนไว้ {pca.explained_variance_ratio_.sum()*100:.1f}%)")

# --- 🧠 4. ฝึกและประเมินผล 3 โมเดล ---
models = {
    # เปลี่ยนจาก linear -> rbf (จับ pattern ไม่เป็นเส้นตรงได้ดีกว่า) และปรับ C
    "SVM (Kernel: RBF)": SVC(kernel='rbf', C=5, gamma='scale', probability=True, random_state=RANDOM_STATE),
    # เปลี่ยน weights กลับเป็น 'uniform' เพราะ 'distance' จะไปขยายความเอนเอียงในมิติสูง (ดูคอมเมนต์ PCA ด้านบน)
    "KNN (K=9)": KNeighborsClassifier(n_neighbors=9, weights='uniform'),
    # จำกัดความลึก ป้องกัน overfit ที่ทำให้มั่นใจ 100% ทั้งที่ทายผิด (ของเดิมไม่ได้จำกัด)
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE)
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
test_features = np.array([item[0] for item in test_data])
test_features = scaler.transform(test_features)  # ⚠️ ต้อง transform ด้วย scaler เดียวกับตอนเทรน
test_features = pca.transform(test_features)      # ⚠️ ต้อง transform ด้วย pca ตัวเดียวกับตอนเทรนด้วย

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
fig.suptitle("Model Predictions with Confidence (%): SVM | KNN | Decision Tree", fontsize=16)

for i in range(len(test_features)):
    true_label = CLASS_NAMES[test_labels_true[i]]
    img_name = os.path.basename(valid_test_paths[i])

    print(f"\nรูปที่ {i+1}: {img_name} (เฉลยจริง: {true_label})")

    preds_text = ""
    for name, model in trained_models.items():
        input_feature = test_features[i].reshape(1, -1)

        pred_val = model.predict(input_feature)[0]
        pred_label = CLASS_NAMES[pred_val]

        proba = model.predict_proba(input_feature)[0]
        confidence = np.max(proba) * 100

        result_mark = "✅" if pred_label == true_label else "❌"
        short_name = name.split()[0]

        print(f"    - {short_name:15} ทายว่า -> {pred_label:3} ({confidence:5.1f}%) {result_mark}")

        preds_text += f"{short_name}: {pred_label} ({confidence:.1f}%)\n"

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
plt.tight_layout(h_pad=3.5)
plt.show()


#อันนี้้้้