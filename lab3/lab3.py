import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# =======================================================
# *** ส่วนที่เพิ่ม: ตั้งค่า Matplotlib สำหรับภาษาไทย ***
# =======================================================

# 1. ระบุฟอนต์ภาษาไทย (ตัวอย่าง: ใช้ฟอนต์ SarabunPS หรือ Tahoma ที่มักมีอยู่ในระบบ)
#    ถ้าฟอนต์ที่ระบุไม่มีในระบบของคุณ โปรดเปลี่ยนเป็นฟอนต์ภาษาไทยอื่นที่ติดตั้งไว้
thai_font_name = 'Tahoma' # ลองเปลี่ยนเป็น 'SarabunPS' หรือ 'TH Sarabun New' ถ้า Tahoma ไม่มี

# 2. ตั้งค่าฟอนต์หลักสำหรับ Matplotlib
plt.rcParams['font.family'] = thai_font_name
# ป้องกันปัญหาเครื่องหมายลบแสดงผลเป็นสี่เหลี่ยม
plt.rcParams['axes.unicode_minus'] = False

# --- 1. โหลดชุดข้อมูลภาพตัวเลข (Digits Dataset) ---
digits = load_digits()
X = digits.data
y = digits.target

print(f"จำนวนรูปภาพทั้งหมด: {X.shape[0]}")
print(f"มิติของข้อมูล (รูปภาพ x คุณลักษณะ): {X.shape}")

# --- 2. แบ่งข้อมูลสำหรับฝึก (Training) และทดสอบ (Testing) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. สร้างและฝึก Decision Tree Classifier ---
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# --- 4. ทดสอบประสิทธิภาพของโมเดล ---
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nความแม่นยำ (Accuracy) บนชุดข้อมูลทดสอบ: {accuracy:.4f}")

# --- 5. การแสดงตัวอย่างรูปภาพและการทำนาย (Optional) ---
sample_index = 10
sample_image = X_test[sample_index].reshape(8, 8)
prediction = dt_classifier.predict([X_test[sample_index]])[0]
actual = y_test[sample_index]

plt.imshow(sample_image, cmap=plt.cm.gray_r, interpolation='nearest')
# ใช้ภาษาไทยใน title
plt.title(f"ภาพจริง: {actual}, ทำนาย: {prediction}", fontsize=14)
plt.show()

# --- 6. การแสดงผลโครงสร้าง Decision Tree (Visualization) ---
feature_names = [f'Pixel_{i}' for i in range(X.shape[1])]

plt.figure(figsize=(20, 10))
plot_tree(dt_classifier,
          feature_names=feature_names,
          class_names=[str(i) for i in digits.target_names],
          filled=True,
          rounded=True,
          max_depth=3)
# ใช้ภาษาไทยใน title
plt.title("การแสดงผลโครงสร้าง Decision Tree (ระดับความลึกสูงสุด 3)", fontsize=16)
plt.show()