import os
import urllib.request

def download_yolo():
    model_dir = 'YOLO_Model'
    cfg_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
    weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
    
    cfg_path = os.path.join(model_dir, 'yolov4-tiny.cfg')
    weights_path = os.path.join(model_dir, 'yolov4-tiny.weights')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"สร้างโฟลเดอร์ '{model_dir}' สำเร็จ")

    try:
        print("กำลังดาวน์โหลดไฟล์โครงสร้าง (yolov4-tiny.cfg)...")
        urllib.request.urlretrieve(cfg_url, cfg_path)
        
        print("กำลังดาวน์โหลดไฟล์โมเดล (yolov4-tiny.weights) ขนาด 23 MB... กรุณารอสักครู่")
        urllib.request.urlretrieve(weights_url, weights_path)
        
        print("\nดาวน์โหลดเสร็จสมบูรณ์! พร้อมใช้งานในใบงานที่ 8 แล้วครับ")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดาวน์โหลด: {e}")

if __name__ == '__main__':
    download_yolo()