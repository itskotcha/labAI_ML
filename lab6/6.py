# 1. นำเข้าไลบรารีที่จำเป็น (รวม 3D Plot)
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
from matplotlib import font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# --- การตั้งค่าฟอนต์ภาษาไทย ---
font_name = 'Tahoma'
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 2. โหลดและเตรียมข้อมูลภาพ
image_path = 'healthy_meal.jpg'

# หากยังไม่มีไฟล์ภาพ จะทำการดาวน์โหลดภาพจานอาหารเพื่อสุขภาพอัตโนมัติ
if not os.path.exists(image_path):
    print("[ 🌐 ] กำลังดาวน์โหลดภาพอาหารตัวอย่าง...")
    url = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=600&auto=format&fit=crop&q=80"
    urllib.request.urlretrieve(url, image_path)

original_image = Image.open(image_path)
resized_image = original_image.resize((100, 100))

if resized_image.mode != 'RGB':
    resized_image = resized_image.convert('RGB')

# แปลงภาพเป็น NumPy Array และ normalize (หารด้วย 255)
image_np = np.array(resized_image, dtype=np.float64) / 255.0

# 3. แปลงภาพ 2 มิติ (กว้าง x สูง x สี) ให้เป็นข้อมูล 2 มิติ (พิกเซล x สี)
data = image_np.reshape(-1, 3)  # (10000, 3)

# 4. กำหนดจำนวนกลุ่ม (Clusters) ที่ต้องการ
n_colors = 16

# 5. ฝึกโมเดล K-Means เพื่อหาค่าสีหลัก
kmeans = MiniBatchKMeans(n_clusters=n_colors, n_init='auto', random_state=42)
kmeans.fit(data)

# 6. สร้างภาพใหม่ด้วยสีจาก Clusters ที่ได้
new_colors_normalized = kmeans.cluster_centers_[kmeans.labels_]
new_image_array = (new_colors_normalized * 255).reshape(image_np.shape)
new_image = new_image_array.astype(np.uint8)

# 7. แสดงผลภาพ (2D Image Plots)
plt.figure(figsize=(15, 6))

# --- 7.1 ภาพต้นฉบับ ---
plt.subplot(1, 2, 1)
plt.title('ภาพต้นฉบับ (100x100)', fontsize=14)
plt.imshow(resized_image)
plt.axis('off')

# --- 7.2 ภาพลดสี ---
plt.subplot(1, 2, 2)
plt.title(f'ภาพลดสีเหลือ {n_colors} สี', fontsize=14)
plt.imshow(new_image)
plt.axis('off')

plt.suptitle('การลดจำนวนสีของภาพอาหารด้วย MiniBatchKMeans', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 8. สร้าง 3D SCATTER PLOT (ตามรูปแบบของอาจารย์)
plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.set_title('3D Scatter Plot: การกระจายตัวของสี (R, G, B) และ Centroids', fontsize=14)

# 8.1 เตรียมข้อมูลสำหรับ 3D Plot (สุ่ม 2,000 จุดเพื่อลดเวลาในการพล็อต)
sample_size = 2000
sample_indices = np.random.choice(len(data), size=sample_size, replace=False)

sample_data = data[sample_indices]
sample_labels = kmeans.labels_[sample_indices]
centers = kmeans.cluster_centers_

# 8.2 พล็อตจุดข้อมูลตัวอย่าง (พิกเซล) ตามค่าสีจริง
ax.scatter(sample_data[:, 0], sample_data[:, 1], sample_data[:, 2],
           c=sample_data, s=10, alpha=0.3)

# 8.3 พล็อต Centroids (จุดศูนย์กลางของกลุ่มสี 16 จุด)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
           s=250, c=centers, marker='o', edgecolors='black', linewidth=1.5, label='Centroids')

# 8.4 กำหนดป้ายกำกับแกนตามแบบอาจารย์
ax.set_xlabel('ค่าสีแดง (R)', fontsize=12)
ax.set_ylabel('ค่าสีเขียว (G)', fontsize=12)
ax.set_zlabel('ค่าสีน้ำเงิน (B)', fontsize=12)

# กำหนดขีดจำกัดแกน (0-1 เนื่องจากข้อมูลถูก Normalize)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# เพิ่ม Legend สำหรับ Centroids
ax.legend(['พิกเซลตัวอย่าง', 'Centroids'], loc='upper right')

# 9. แสดงจำนวนสีที่แท้จริง
orig_colors_count = len(np.unique(np.array(resized_image).reshape(-1, 3), axis=0))
new_colors_count = len(np.unique(new_image.reshape(-1, 3), axis=0))

print(f"\nจำนวนสีในภาพต้นฉบับ (100x100): {orig_colors_count} สี")
print(f"จำนวนสีในภาพใหม่ (เป้าหมาย {n_colors}): {new_colors_count} สี")

# แสดงทั้ง 2 หน้าต่างพร้อมกัน
plt.show()