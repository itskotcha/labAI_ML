import numpy as np
import matplotlib.pyplot as plt  # เพิ่มไลบรารีนี้สำหรับแสดงรูปภาพ
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. โหลดและเตรียมชุดข้อมูลภาพจากลิงก์ OpenML (Dataset: Fashion-MNIST)
print("กำลังดาวน์โหลดรูปภาพจากอินเทอร์เน็ตผ่าน OpenML... (อาจใช้เวลา 1-2 นาที)")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')

X = X.astype('float32') # ข้อมูลพิกเซลภาพ
y = y.astype('int')     # ป้ายกำกับ (0-9)
print(f"โหลดข้อมูลสำเร็จ! ได้รับรูปภาพทั้งหมด: {X.shape[0]} รูป")

# รายชื่อหมวดหมู่เสื้อผ้าทั้ง 10 คลาส
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- ส่วนที่เพิ่มเข้ามา: แสดงภาพตัวอย่าง 10 ภาพแรก ---
print("\nแสดงภาพตัวอย่าง 10 ภาพแรก (กรุณาปิดหน้าต่างรูปภาพเพื่อรันโค้ดต่อ):")
fig, axes = plt.subplots(2, 5, figsize=(10, 4)) # สร้างกริด 2 แถว 5 คอลัมน์

for i, ax in enumerate(axes.flat):
    # ปรับรูปร่างข้อมูลจาก 1 มิติ (784) กลับเป็น 2 มิติ (28x28) เพื่อให้วาดรูปได้
    image_reshaped = X[i].reshape(28, 28)
    ax.imshow(image_reshaped, cmap='gray')
    
    # กำหนดหัวข้อรูปภาพเป็น Label (แสดงชื่อหมวดหมู่เสื้อผ้าให้ดูง่ายขึ้น)
    ax.set_title(f"Label: {class_names[y[i]]}") 
    ax.axis('off')

plt.tight_layout()
plt.show() 
# หมายเหตุ: โปรแกรมจะหยุดรอตรงนี้จนกว่าคุณจะกดปิด (X) หน้าต่างรูปภาพที่เด้งขึ้นมา
# --------------------------------------------------

# 2. แบ่งข้อมูลสำหรับฝึก (Training) 80% และทดสอบ (Testing) 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. สร้างและฝึกโมเดล Random Forest Classifier
print(f"\nกำลังฝึกโมเดล RandomForest ด้วยข้อมูล {X_train.shape[0]} รูป...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("ฝึกโมเดลเสร็จสิ้น!")

# 4. ทำนายและประเมินผล
print("\nกำลังทำนายผลลัพธ์บนชุดข้อมูลทดสอบ...")
y_pred = rf_model.predict(X_test)

# วัดความแม่นยำ (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nความแม่นยำรวมของโมเดล (Accuracy): {accuracy:.4f}")

# แสดงตาราง Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# แสดงรายงานผลการจำแนกประเภทเชิงลึก
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))