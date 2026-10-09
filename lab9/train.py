from ultralytics import YOLO

def main():
    # โหลดโมเดลต้นแบบที่เรียนรู้จาก COCO มาแล้ว (transfer learning)
    model = YOLO("yolo11n.pt")

    model.train(
        data="dataset/data.yaml",  # ไฟล์กำหนดตำแหน่งข้อมูลและชื่อคลาส
        epochs=100,                # จำนวนรอบที่ฝึก
        imgsz=640,                 # ขนาดรูปที่ป้อนเข้าโมเดล
        batch=16,                  # จำนวนรูปต่อรอบการอัปเดตน้ำหนัก
        device="cpu",                  # 0 = GPU, "cpu" = ใช้ CPU
        workers=0,                 # กันปัญหา multiprocessing บน Windows
        patience=30,               # หยุดเองถ้าไม่ดีขึ้นต่อเนื่อง 30 epoch
    )

    # วัดผลบน validation set
    metrics = model.val()
    print("Precision :", metrics.box.mp)
    print("Recall    :", metrics.box.mr)
    print("mAP50     :", metrics.box.map50)
    print("mAP50-95  :", metrics.box.map)

if __name__ == "__main__":   # จำเป็นบน Windows
    main()