import os
import glob
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import kagglehub

# --- 1. ตั้งค่าระบบและกราฟ ---
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False

# ชื่อ Dataset อายุกระดูกจาก Kaggle
KAGGLE_DATASET = "kmader/rsna-bone-age"
IMAGE_SIZE = 64
SAMPLE_LIMIT = 500  # จำกัด 500 ภาพเพื่อให้เทรนได้รวดเร็ว (ปรับเพิ่มได้ถ้าคอมแรง)

print(f"[*] กำลังดาวน์โหลดชุดข้อมูล {KAGGLE_DATASET} จาก Kaggle Hub...")
print("    (หมายเหตุ: ข้อมูลจริงทางการแพทย์มีขนาดค่อนข้างใหญ่ อาจใช้เวลาดาวน์โหลดสักครู่ในครั้งแรก)")
start_time = time.time()

try:
    # --- 2. ดาวน์โหลดและค้นหาไฟล์ใน Dataset ---
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[✓] ดาวน์โหลดสำเร็จ! พาธหลัก: {download_path}")

    # หาไฟล์ CSV ที่เก็บ Label (อายุกระดูก)
    csv_files = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
    train_csv = [f for f in csv_files if 'train' in f.lower()]
    if not train_csv:
        train_csv = csv_files # ถ้าไม่มีคำว่า train ให้ใช้ไฟล์ CSV แรกที่เจอ
        
    df = pd.read_csv(train_csv[0])
    print(f"[*] พบไฟล์ข้อมูล: {os.path.basename(train_csv[0])}")

    # หาไฟล์รูปภาพ (.png) ทั้งหมดในโฟลเดอร์
    all_images = glob.glob(os.path.join(download_path, "**", "*.png"), recursive=True)
    
    # สร้าง Dictionary ค้นหารูปภาพจากชื่อไฟล์ (ID) เพื่อความรวดเร็ว
    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in all_images}

    X_images = []
    y_age = []
    display_images = []

    print(f"[*] กำลังเตรียมข้อมูลรูปภาพ (สูงสุด {SAMPLE_LIMIT} ภาพ)...")
    count = 0
    
    # --- 3. แปลงภาพเป็นฟีเจอร์ ---
    for _, row in df.iterrows():
        if count >= SAMPLE_LIMIT:
            break
            
        img_id = str(row['id'])  # รหัสภาพ
        bone_age = row['boneage'] # อายุกระดูก (เดือน)
        
        if img_id in image_dict:
            # โหลดภาพเป็นสีขาวดำ (Grayscale)
            img = cv2.imread(image_dict[img_id], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # ย่อขนาดภาพเป็น 64x64 พิกเซล
                img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                
                # Flatten แปลงภาพเป็นเวกเตอร์ 1 มิติ (4096 ฟีเจอร์) และ Normalize (0-1)
                X_images.append(img_resized.flatten() / 255.0)
                y_age.append(bone_age)
                display_images.append(img_resized) # เก็บไว้พล็อตกราฟ
                count += 1

    X = np.array(X_images)
    y = np.array(y_age)
    print(f"[✓] โหลดข้อมูลสำเร็จ: {len(X)} ภาพ (Feature ต่อภาพ: {X.shape[1]})")

    # --- 4. แบ่งข้อมูล Train / Test ---
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, display_images, test_size=0.2, random_state=42
    )

    # --- 5. เทรนโมเดล Linear Regression ---
    print("[*] กำลังเทรนโมเดล Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[✓] เทรนโมเดลเสร็จสิ้น")

    # --- 6. ทำนายและประเมินผลลัพธ์ ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- ผลการประเมินประสิทธิภาพโมเดล ---")
    print(f"ความผิดพลาดเฉลี่ยสัมบูรณ์ (MAE): {mae:.2f} เดือน")
    print(f"ความแม่นยำ (R-squared): {r2:.4f}")
    print(f"เวลาทำงานรวม: {time.time() - start_time:.2f} วินาที")

    # --- 7. แสดงผลลัพธ์บนกราฟ ---
    num_display = min(10, len(X_test))
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(f"ผลทำนายอายุกระดูก (Bone Age) จากฟิล์มเอกซเรย์\n(MAE: {mae:.2f} เดือน)", fontsize=16)

    axes = axes.flatten()
    for i in range(num_display):
        axes[i].imshow(img_test[i], cmap='gray')
        actual = y_test[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        
        # ถ้าทายคลาดเคลื่อนไม่เกิน 1 ปี (12 เดือน) ให้ตัวหนังสือสีเขียว
        title_color = 'green' if error <= 12 else 'red'
        axes[i].set_title(f"ทำนาย: {predicted:.1f} เดือน\nจริง: {actual} เดือน", fontsize=10, color=title_color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n[X] เกิดข้อผิดพลาด: {e}")