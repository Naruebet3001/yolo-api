import os
import subprocess

# สร้างโฟลเดอร์ runs ถ้ายังไม่มี
os.makedirs("runs", exist_ok=True)

# คำสั่ง train YOLOv8
cmd = [
    "yolo", "detect", "train",
    "data=datasets/data.yaml",   # dataset ที่อัปโหลด (ต้องให้ชื่อ data.yaml ตรง)
    "model=yolov8n.pt",          # base model
    "epochs=10",                 # จำนวน epochs
    "imgsz=640",                 # ขนาดภาพ
    "project=runs", "name=detect/train"
]

subprocess.run(cmd)
