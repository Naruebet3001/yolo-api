import os, subprocess, sys

# เก็บ pid ของตัวเอง
with open("pid.txt", "w") as f:
    f.write(str(os.getpid()))

# run yolov8
cmd = [
    "yolo", "detect", "train",
    "data=datasets/data.yaml",
    "model=yolov8n.pt",
    "epochs=5", "imgsz=640",
    "project=runs", "name=train"
]

subprocess.run(cmd)


