
import cv2
from ultralytics import YOLO
from pathlib import Path

# ตำแหน่งโมเดลที่ฝึกเสร็จแล้ว
model_path = Path("runs/detect/train/weights/best.pt")
# ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
if not model_path.exists():
    raise FileNotFoundError(
        f"ไม่พบไฟล์โมเดล: {model_path}\n"
        "กรุณาตรวจสอบตำแหน่งไฟล์ best.pt"
    )

# โหลดโมเดล YOLO
model = YOLO(str(model_path))

# เปิดเว็บแคม (ถ้าเปิดไม่ได้ ให้ลองเปลี่ยน 0 เป็น 1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ไม่สามารถเปิดเว็บแคมได้ กรุณาลองเปลี่ยน index เป็น 1")
    exit()

count = 0

print("เริ่มตรวจจับวัตถุแล้ว")
print("กด S เพื่อบันทึกภาพ")
print("กด Q เพื่อออกจากโปรแกรม")

while True:
    ok, frame = cap.read()

    if not ok:
        print("ไม่สามารถอ่านภาพจากเว็บแคมได้")
        break

    # ตรวจจับวัตถุ โดยกำหนดความมั่นใจขั้นต่ำ 50%
    results = model.predict(
        source=frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    # วาดกรอบ ชื่อคลาส และค่าความมั่นใจ
    annotated = results[0].plot()

    cv2.imshow("Product Detection - S: Save | Q: Quit", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        count += 1
        filename = f"capture_{count}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"บันทึกภาพแล้ว: {filename}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()