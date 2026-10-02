import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# --- การตั้งค่าฟอนต์ภาษาไทย ---
font_name = 'Tahoma'
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# ส่วนที่ 1: Simple Linear Regression (พยากรณ์รายได้ ตามประสบการณ์ทำงาน)
# =====================================================================
print("="*50)
print("PART 1: Simple Linear Regression")
print("="*50)

# 1. เตรียมข้อมูล (Data Preparation)
data_simple = {
    'Years Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Income': [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000]
}
df_simple = pd.DataFrame(data_simple)

# 2. แยก X (คุณลักษณะ) และ y (เป้าหมาย)
X_simp = df_simple[['Years Experience']]
y_simp = df_simple['Income']

# 3. สร้างโมเดล Linear Regression และฝึก (Training)
model_simp = LinearRegression()
model_simp.fit(X_simp, y_simp)

# 4. วิเคราะห์และแสดงผลลัพธ์ของโมเดล
df_simple['Predicted'] = model_simp.predict(X_simp)
r2_simp = r2_score(y_simp, df_simple['Predicted'])

intercept_simp = model_simp.intercept_
coefficient_simp = model_simp.coef_[0]

print("--- ผลการวิเคราะห์ Linear Regression ---")
print(f"สมการโมเดล: Y (รายได้) = {coefficient_simp:,.2f} X (ประสบการณ์) + {intercept_simp:,.2f}")
print(f"ความแม่นยำ (R-squared): {r2_simp:.4f}")
print(f"ค่าสัมประสิทธิ์ (Coefficient): {coefficient_simp:,.2f} บาท/ปี")
print(f"ค่าคงที่ (Intercept): {intercept_simp:,.2f} บาท")

# 5. แสดงผลกราฟ
plt.figure(figsize=(10, 6))
plt.scatter(X_simp, y_simp, color='blue', label='รายได้จริง', s=80, alpha=0.7)
plt.plot(X_simp, df_simple['Predicted'], color='red', label=f'เส้นทำนาย (R²={r2_simp:.4f})', linewidth=2)

plt.xlabel('ประสบการณ์ทำงาน (ปี)', fontsize=12)
plt.ylabel('รายได้ (บาท)', fontsize=12)
plt.title('การวิเคราะห์ความสัมพันธ์ระหว่างประสบการณ์และรายได้', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show() # กราฟแรกจะเด้งขึ้นมา (ปิดกราฟเพื่อรันโค้ดส่วนต่อไป)

# 6. ทำนายรายได้จากประสบการณ์ทำงาน 12 ปี (Forecasting)
experience_new = np.array([[12]])
predicted_income = model_simp.predict(experience_new)

print("\n--- ผลการทำนาย ---")
print(f"รายได้ที่คาดว่าจะได้รับเมื่อมีประสบการณ์ {experience_new[0][0]} ปี: {predicted_income[0]:,.2f} บาท\n")


# =====================================================================
# ส่วนที่ 2: Multiple Linear Regression (คาดการณ์ราคาบ้าน จากพื้นที่และอายุบ้าน)
# =====================================================================
print("="*50)
print("PART 2: Multiple Linear Regression")
print("="*50)

# 1. สร้าง DataFrame (ข้อมูลบ้านจำลอง)
data_mul = {
    'Area': [50, 70, 80, 100, 60, 90],
    'Bedrooms': [1, 2, 3, 3, 2, 3],
    'Age': [10, 5, 3, 1, 15, 7],
    'Price': [1000000, 1500000, 1800000, 2200000, 1200000, 2000000]
}
df_mul = pd.DataFrame(data_mul)

# 2. สร้าง X (คุณลักษณะ) และ y (เป้าหมาย) 
# เราเลือกใช้ 'Area' และ 'Age' สำหรับกราฟ 3D
X_mul = df_mul[['Area', 'Age']]
y_mul = df_mul['Price']

# 3. สร้างและฝึกโมเดล Multiple Linear Regression
model_mul = LinearRegression()
model_mul.fit(X_mul, y_mul)

# 4. วิเคราะห์ผลลัพธ์ของโมเดล
intercept_mul = model_mul.intercept_
coef_area = model_mul.coef_[0]
coef_age = model_mul.coef_[1]

y_pred_mul = model_mul.predict(X_mul)
r2_mul = r2_score(y_mul, y_pred_mul)

print("--- ผลการวิเคราะห์ Multiple Linear Regression (Area, Age) ---")
print(f"สมการโมเดล: Price = ({coef_area:,.2f} * Area) + ({coef_age:,.2f} * Age) + ({intercept_mul:,.2f})")
print(f"ความแม่นยำ (R-squared): {r2_mul:.4f}")
print(f"อัตราส่วนราคาต่อพื้นที่ (Area Coeff.): {coef_area:,.2f} บาท/ตร.ม.")
print(f"ผลกระทบต่อราคาต่ออายุ (Age Coeff.): {coef_age:,.2f} บาท/ปี")
print(f"ราคาฐานเริ่มต้น (Intercept): {intercept_mul:,.2f} บาท")

# 5. สร้างกราฟ 3 มิติ (แสดงจุดข้อมูลจริงและพื้นผิวการถดถอย)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 5.1 จุดข้อมูลจริง
ax.scatter(df_mul['Area'], df_mul['Age'], y_mul, c='blue', s=80, label='ราคาจริง')

# 5.2 สร้างพื้นผิวของเส้นถดถอย (Regression Plane)
area_range = np.linspace(df_mul['Area'].min(), df_mul['Area'].max(), 10)
age_range = np.linspace(df_mul['Age'].min(), df_mul['Age'].max(), 10)
area_grid, age_grid = np.meshgrid(area_range, age_range)

# ทำนายราคาสำหรับทุกจุดใน Grid
input_grid = np.c_[area_grid.ravel(), age_grid.ravel()]
price_pred_flat = model_mul.predict(input_grid)
price_grid = price_pred_flat.reshape(area_grid.shape)

# พล็อตพื้นผิว
ax.plot_surface(area_grid, age_grid, price_grid, color='red', alpha=0.5, label='พื้นผิวการทำนาย')

# 5.3 ตั้งค่าแกนและชื่อกราฟ
ax.set_xlabel('พื้นที่ (ตร.ม.)', fontsize=12)
ax.set_ylabel('อายุบ้าน (ปี)', fontsize=12)
ax.set_zlabel('ราคา (บาท)', fontsize=12)
ax.set_title(f'3D Multiple Linear Regression (R²={r2_mul:.4f}): การทำนายราคาบ้าน', fontsize=14)

# สร้างจุด Dummy สำหรับ Legend ของพื้นผิว
dummy_line = plt.plot([1], [1], [1], linestyle='-', color='red', alpha=0.5, label='พื้นผิวทำนาย')[0]
ax.legend([ax.collections[0], dummy_line], ['ราคาจริง', 'พื้นผิวทำนาย'], loc='upper left')

plt.show() # กราฟที่สองจะเด้งขึ้นมา

# 6. ทำนายบ้านใหม่
new_house = np.array([[85, 4]])
predicted_price = model_mul.predict(new_house)

print("\n--- ผลการทำนาย ---")
print(f"ราคาที่คาดการณ์สำหรับบ้านใหม่ (พื้นที่ {new_house[0][0]} ตร.ม., อายุ {new_house[0][1]} ปี): {predicted_price[0]:,.2f} บาท")