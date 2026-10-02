import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- ตั้งค่าฟอนต์ภาษาไทย ---
plt.rcParams['font.family'] = 'Tahoma' 
plt.rcParams['axes.unicode_minus'] = False

print("[*] กำลังสร้าง Dataset รูปภาพไอคอนแบตเตอรี่จำลอง...")

# 1. สร้าง Dataset รูปภาพแบตเตอรี่จำลอง (Synthetic Image Dataset)
num_samples = 500  # สร้างภาพทั้งหมด 500 ภาพ
image_size = 64
X_images = []
y_percentages = []

for _ in range(num_samples):
    # สุ่มเปอร์เซ็นต์แบตเตอรี่ 0 - 100%
    battery_pct = np.random.randint(0, 101)
    
    # สร้างภาพพื้นหลังสีดำ (ขนาด 64x64)
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # วาดกรอบแบตเตอรี่สีขาว
    cv2.rectangle(img, (10, 20), (50, 44), 255, 2)
    cv2.rectangle(img, (50, 26), (54, 38), 255, -1) # ขั้วแบตเตอรี่
    
    # วาดปริมาณแบตเตอรี่ข้างในตามเปอร์เซ็นต์ที่สุ่มได้
    max_fill_width = 36 # ความกว้างสูงสุดของพื้นที่ชาร์จด้านใน
    fill_width = int((battery_pct / 100.0) * max_fill_width)
    
    if fill_width > 0:
        cv2.rectangle(img, (12, 22), (12 + fill_width, 42), 255, -1)
        
    X_images.append(img)
    y_percentages.append(battery_pct)

# 2. แปลงข้อมูล (Preprocessing)
# แปลงภาพ 2 มิติ (64x64) ให้เป็นเวกเตอร์ 1 มิติ (Flatten เป็น 4096 features) และ Normalize (0-1)
X = np.array([img.flatten() / 255.0 for img in X_images])
y = np.array(y_percentages)

print(f"[✓] สร้างข้อมูลสำเร็จ: จำนวน {len(X)} ภาพ (Feature ต่อภาพ: {X.shape[1]})")

# 3. แบ่งชุดข้อมูล Train/Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. เทรนโมเดล Linear Regression
print("[*] กำลังเทรนโมเดล Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)
print("[✓] เทรนโมเดลเสร็จสิ้น")

# 5. ทำนายและประเมินผลลัพธ์
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- ผลการประเมินประสิทธิภาพโมเดล ---")
print(f"ความผิดพลาดเฉลี่ยสัมบูรณ์ (MAE): {mae:.2f} %")
print(f"ความแม่นยำ (R-squared): {r2:.4f}")

# 6. แสดงผลลัพธ์รูปภาพพร้อมค่าที่ทำนายได้
num_display = 10
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle(f"ผลการทำนายเปอร์เซ็นต์แบตเตอรี่ด้วย Linear Regression\n(MAE: {mae:.2f}%)", fontsize=16)

axes = axes.flatten()
for i in range(num_display):
    # นำข้อมูลที่แบนราบ (Flatten) กลับมาเป็นรูปภาพ 2 มิติเพื่อแสดงผล
    img_display = X_test[i].reshape(image_size, image_size)
    
    # พล็อตภาพ
    axes[i].imshow(img_display, cmap='gray')
    
    actual = y_test[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    
    axes[i].set_title(f"ทำนาย: {predicted:.1f}%\nจริง: {actual}%", fontsize=10, color='green' if error <= 5 else 'red')
    axes[i].axis('off')

plt.tight_layout()
plt.show()