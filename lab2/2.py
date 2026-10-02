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

# --- 1. ตั้งค่าระบบและหน้าตาของกราฟ ---
plt.rcParams['font.family'] = 'Tahoma' 
plt.rcParams['axes.unicode_minus'] = False

# กำหนดชื่อ Dataset บน Kaggle และพารามิเตอร์ต่างๆ
KAGGLE_DATASET = "ted8080/house-prices-and-images-socal"
IMAGE_SIZE = 64
SAMPLE_LIMIT = 500  # จำกัดแค่ 500 ภาพเพื่อความรวดเร็วในการเทรน

print(f"[*] กำลังดาวน์โหลดชุดข้อมูล {KAGGLE_DATASET} จาก Kaggle Hub...")
start_time = time.time()

try:
    # --- 2. ดาวน์โหลดข้อมูลจาก Kaggle อัตโนมัติ ---
    download_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[✓] ดาวน์โหลดสำเร็จ. พาธหลัก: {download_path}")
    
    # ใช้ glob เพื่อหาไฟล์ CSV (ตารางราคา) และไฟล์ .jpg (รูปบ้าน) ทั้งหมดในโฟลเดอร์
    csv_files = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
    all_images = glob.glob(os.path.join(download_path, "**", "*.jpg"), recursive=True)
    
    if not csv_files or not all_images:
        print("[X] ไม่พบไฟล์ CSV หรือรูปภาพในโฟลเดอร์ กรุณาตรวจสอบชุดข้อมูล")
        exit()
        
    print(f"[*] พบไฟล์ CSV: {os.path.basename(csv_files[0])}")
    df = pd.read_csv(csv_files[0])
    
    # โค้ดส่วนนี้จะหาคอลัมน์ที่มีคำว่า 'id' และ 'price' อัตโนมัติ
    image_col = [col for col in df.columns if 'id' in col.lower() or 'image' in col.lower()][0]
    price_col = [col for col in df.columns if 'price' in col.lower()][0]
    
    # สร้าง Dictionary สำหรับค้นหาพาธรูปภาพอย่างรวดเร็ว (ประหยัดเวลาค้นหา)
    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in all_images}
    
    X_images = []
    y_prices = []
    display_images = []
    
    print(f"[*] กำลังเตรียมข้อมูลรูปภาพ (สูงสุด {SAMPLE_LIMIT} ภาพ)...")
    count = 0
    
    # --- 3. แปลงภาพเป็นฟีเจอร์และดึงป้ายกำกับ (Label) ---
    for _, row in df.iterrows():
        if count >= SAMPLE_LIMIT:
            break
            
        # ดึง ID ของรูปภาพเพื่อไปหาไฟล์จริง
        img_id = str(row[image_col]).replace(".jpg", "")
        
        if img_id in image_dict:
            img = cv2.imread(image_dict[img_id])
            if img is not None:
                # ลดขนาดภาพเพื่อลดภาระการคำนวณ
                img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                # Flatten แปลงภาพเป็นเวกเตอร์ 1 มิติ และ Normalize ให้อยู่ในช่วง 0-1
                X_images.append(img_resized.flatten() / 255.0) 
                y_prices.append(row[price_col])
                
                # เก็บภาพสี RGB ไว้สำหรับแสดงผลบนกราฟ
                display_images.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                count += 1

    X = np.array(X_images)
    y = np.array(y_prices)
    
    print(f"[✓] โหลดข้อมูลสำเร็จ: จำนวน {len(X)} ภาพ (Feature ต่อภาพ: {X.shape[1]})")

    # --- 4. แบ่งชุดข้อมูล Train / Test ---
    # เราส่ง display_images เข้าไปแบ่งด้วย เพื่อให้ภาพที่แสดงผลตรงกับข้อมูลที่ใช้ทดสอบ
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, display_images, test_size=0.2, random_state=42
    )

    # --- 5. เทรนโมเดล Linear Regression ---
    print("[*] กำลังเทรนโมเดล Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[✓] เทรนโมเดลเสร็จสิ้น")

    # --- 6. ประเมินผลลัพธ์ ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- ผลการประเมินประสิทธิภาพโมเดล ---")
    print(f"ความผิดพลาดเฉลี่ยสัมบูรณ์ (MAE): {mae:,.2f} ดอลลาร์")
    print(f"ความแม่นยำ (R-squared): {r2:.4f}")
    print(f"เวลาทำงานรวม: {time.time() - start_time:.2f} วินาที")

    # --- 7. แสดงผลรูปภาพพร้อมคำทำนาย ---
    num_display = min(10, len(X_test))
    fig, axes = plt.subplots(2, num_display // 2, figsize=(15, 6))
    fig.suptitle(f"ผลการทำนายราคาบ้านจากภาพถ่ายด้วย Linear Regression (MAE: ${mae:,.0f})", fontsize=16)

    axes = axes.flatten()
    for i in range(num_display):
        axes[i].imshow(img_test[i])
        actual = y_test[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        
        # ย่อตัวเลขราคา (เช่น 500,000 -> 500k) เพื่อให้กราฟดูสะอาดตา
        title_text = f"ทำนาย: ${predicted/1000:,.0f}k\nจริง: ${actual/1000:,.0f}k"
        
        # ถ้าทายผิดพลาดน้อยกว่า 50% ของราคาจริง ให้ตัวหนังสือสีเขียว (ถือว่าพอรับได้ในงานภาพ)
        axes[i].set_title(title_text, fontsize=10, color='green' if error < actual*0.5 else 'red')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n[X] เกิดข้อผิดพลาด: {e}")