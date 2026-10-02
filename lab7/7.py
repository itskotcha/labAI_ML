import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random
import time
import matplotlib.font_manager as fm

# ==========================================
# 1. จัดการฟอนต์ภาษาไทยสำหรับกราฟ
# ==========================================
def get_thai_font():
    try:
        thai_fonts = ['Tahoma', 'Arial Unicode MS', 'Cordia New']
        for font in thai_fonts:
            if fm.findfont(font): return font
    except: pass
    return 'sans-serif'

plt.rcParams['font.family'] = get_thai_font()
plt.rcParams['axes.unicode_minus'] = False 

start_time = time.time()

# ==========================================
# 2. จัดเตรียมข้อมูล (ดึงจากโฟลเดอร์ของคุณโดยตรง)
# ==========================================
print("\n[ ⬇️ ] กำลังตรวจสอบไฟล์ชุดข้อมูล...")

# ชี้พาทไปยังโฟลเดอร์ที่ซ้อนกันตามรูปภาพของคุณ
source_dir = "C:/Users/kotch/labAI_ML/lab7/datasets/flower_photos/flower_photos"
dataset_dir = "C:/Users/kotch/labAI_ML/lab7/dataset/Flowers_Homework"

def prepare_homework_dataset(src_dir, dest_dir, classes, train_size=80, val_size=20):
    # ล้างโฟลเดอร์เก่าทิ้งเสมอ เพื่อป้องกันบัค
    if os.path.exists(dest_dir):
        print(f"[ 🧹 ] กำลังล้างโฟลเดอร์เก่าทิ้งเพื่อดึงรูปใหม่...")
        shutil.rmtree(dest_dir)

    print(f"[ ⏳ ] กำลังคัดแยกรูปภาพตามเงื่อนไขโจทย์ (คลาสละ {train_size+val_size} รูป)...")
    for split in ['training', 'validation']:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
            
    for cls in classes:
        src_class_path = os.path.join(src_dir, cls)
        
        # ดึงไฟล์รูปทั้งหมดในโฟลเดอร์
        all_images = [f for f in os.listdir(src_class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(all_images)
        
        # เลือกรูปตามจำนวนที่กำหนด
        selected_images = all_images[:(train_size + val_size)]
        
        for img in selected_images[:train_size]:
            shutil.copy(os.path.join(src_class_path, img), os.path.join(dest_dir, 'training', cls, img))
        for img in selected_images[train_size:]:
            shutil.copy(os.path.join(src_class_path, img), os.path.join(dest_dir, 'validation', cls, img))
            
    print(f"[ ✅ ] จัดเตรียมข้อมูล 5 คลาส คลาสละ {train_size+val_size} รูปเสร็จสิ้น!\n")

flower_classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# ปรับสัดส่วนการแบ่งข้อมูลเป็น ฝึก 80 รูป / ทดสอบ 20 รูป (ต่อคลาส)
prepare_homework_dataset(source_dir, dataset_dir, flower_classes, train_size=80, val_size=20)

# ==========================================
# 3. กำหนดพารามิเตอร์และโหลดข้อมูลเข้าโมเดล
# ==========================================
img_size = 128
batch_size = 10 
epochs = 15 # เพิ่มจำนวนรอบเล็กน้อย เพราะใช้ Data Augmentation โมเดลจะค่อยๆ เรียนรู้

# เพิ่ม Data Augmentation สำหรับชุดข้อมูลฝึกสอน
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ชุดข้อมูลทดสอบ ใช้แค่ปรับสเกลสี (ห้ามทำ Augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

print("[ ⏳ ] โหลดชุดข้อมูลฝึกสอน (Training Set)...")
train_gen = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'training'),
    target_size=(img_size, img_size),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True
)

print("[ ⏳ ] โหลดชุดข้อมูลตรวจสอบ (Validation Set)...")
val_gen = val_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'validation'),
    target_size=(img_size, img_size),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True # สุ่มหยิบมาแสดงผลตอนทำนาย
)

num_classes = train_gen.num_classes
class_labels = list(train_gen.class_indices.keys())

# ==========================================
# 4. สร้างและฝึกโมเดล Neural Network (CNN)
# ==========================================
model = Sequential([
    Input(shape=(img_size, img_size, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # เพิ่ม Dropout 50% เพื่อป้องกันการจำข้อสอบ (Overfitting)
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print(f"\n[ 📝 ] เริ่มฝึกโมเดลจำนวน {epochs} Epochs...")
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
print("[ ✅ ] ฝึกโมเดลเสร็จสิ้น")

# ==========================================
# 5. แสดงกราฟแบบ 2 แถว (บน-ล่าง ตามเรฟเฟอเรนซ์)
# ==========================================
plt.figure(figsize=(8, 10)) 

# กราฟ Accuracy (อยู่ด้านบน)
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='ความแม่นยำ (ชุดฝึก)')
plt.plot(history.history['val_accuracy'], label='ความแม่นยำ (ชุดตรวจสอบ)')
plt.title('กราฟความแม่นยำ: Accuracy บนชุดทดสอบ', fontsize=14)
plt.xlabel('รอบการฝึก (Epoch)')
plt.ylabel('ความแม่นยำ (Accuracy)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# กราฟ Loss (อยู่ด้านล่าง)
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='ค่า Loss (ชุดฝึก)')
plt.plot(history.history['val_loss'], label='ค่า Loss (ชุดตรวจสอบ)')
plt.title('กราฟ Loss บนชุดทดสอบ', fontsize=14)
plt.xlabel('รอบการฝึก (Epoch)')
plt.ylabel('ค่า Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 6. ทดสอบทำนายภาพ 8 รูป (2 แถว 4 คอลัมน์)
# ==========================================
print("\n--- 7. ทดสอบทำนายภาพตัวอย่าง ---")
N_IMAGES_TO_SHOW = 8 

# สุ่มข้อมูลจาก Validation Generator มาทดสอบ
test_images, test_labels = next(val_gen)
predictions = model.predict(test_images)

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

for i in range(min(N_IMAGES_TO_SHOW, len(test_images))):
    image_to_show = test_images[i]
    
    actual_index = np.argmax(test_labels[i])
    pred_index = np.argmax(predictions[i])
    pred_prob = np.max(predictions[i])
    
    actual_label = class_labels[actual_index]
    pred_label = class_labels[pred_index]
    
    # กำหนดสีตามเรฟเฟอเรนซ์ (เขียว=ถูก, แดง=ผิด)
    color = "green" if pred_index == actual_index else "red"
    
    axes[i].imshow(image_to_show)
    axes[i].set_title(f"ทำนาย: {pred_label} ({pred_prob:.2f})\nจริง: {actual_label}", color=color, fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# 7. สรุปเวลาการทำงาน
# ==========================================
print("\n" + "="*50)
print(f"[ ⏱️ ] เวลารวมในการรันโปรแกรม: {time.time() - start_time:.2f} วินาที")
print("="*50)