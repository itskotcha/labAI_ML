import cv2
from ultralytics import YOLO

# 1. โหลดโมเดล YOLOv8n (ระบบจะดาวน์โหลดไฟล์ให้อัตโนมัติ ไม่ต้องหาโหลดเอง)
print("กำลังโหลดโมเดล Deep Learning...")
model = YOLO('yolov8n.pt')
print("โหลดโมเดลสำเร็จ!")

# 2. เปิดกล้องเว็บแคม
# เพิ่ม cv2.CAP_DSHOW เพื่อบังคับใช้กล้องหลักของ Windows ป้องกันภาพเขียวจากกล้องจำลอง
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# **หมายเหตุ:** หากรันแล้วกล้องยังไม่ติด หรือเกิด Error ให้เปลี่ยนเลข 0 ด้านบนเป็น 1 หรือ 2 
# เช่น cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

detected_objects = set()

# 3. ลูปหลักสำหรับการตรวจจับผ่านกล้อง
while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # นำเฟรมภาพเข้าสู่โมเดลเพื่อวิเคราะห์ (ทำ Deep Learning Object Detection)
    results = model(frame, verbose=False)
    
    current_frame_objects = set()
    
    # ดึงชื่อวัตถุที่ตรวจจับได้ในเฟรมปัจจุบัน
    for c in results[0].boxes.cls:
        class_name = model.names[int(c)]
        current_frame_objects.add(class_name)
        detected_objects.add(class_name)

    # ให้ YOLO วาดกล่องและป้ายกำกับลงบนภาพให้อัตโนมัติ
    annotated_frame = results[0].plot()

    # แสดงจำนวนชนิดวัตถุที่ไม่ซ้ำกันที่พบในเฟรม
    count_text = f"Objects detected (Unique): {len(current_frame_objects)}"
    cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # ตรวจสอบเงื่อนไขโจทย์: ต้องเจอวัตถุ 5 ชนิดในภาพเดียวกัน
    if len(current_frame_objects) >= 5:
        cv2.putText(annotated_frame, "Goal Achieved: 5 Objects Found!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # แสดงผลลัพธ์
    cv2.imshow('Lab 08 - OpenCV 5 + YOLOv8', annotated_frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# สรุปรายชื่อวัตถุทั้งหมดเมื่อปิดโปรแกรม
print(f"\nรายชื่อวัตถุทั้งหมดที่ตรวจพบ: {', '.join(detected_objects)}")