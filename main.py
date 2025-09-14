from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import subprocess, os, signal, zipfile, yaml, shutil

app = FastAPI()

UPLOAD_DIR = "./uploads"
LOG_DIR = "./logs"
MODEL_DIR = "./runs/exp1/weights"

# สร้างโฟลเดอร์ที่จำเป็นหากยังไม่มี
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True) # เพิ่มการสร้างโฟลเดอร์สำหรับโมเดล

LOG_FILE = os.path.join(LOG_DIR, "train.log")
process = None

class TrainRequest(BaseModel):
    resume: bool = False

@app.get("/")
def root():
    return {"message": "YOLOv8 API is running on Render!"}

@app.post("/upload")
async def upload_files(dataset: UploadFile = File(...), yaml_file: UploadFile = File(...)):
    """
    อัปโหลดไฟล์ dataset.zip และ config.yaml
    จากนั้นแตกไฟล์และแก้ไขพาธใน config.yaml
    """
    
    # 1. จัดการการอัปโหลดไฟล์
    dataset_path = os.path.join(UPLOAD_DIR, "dataset.zip")
    yaml_path = os.path.join(UPLOAD_DIR, "config.yaml")

    # บันทึกไฟล์ที่อัปโหลด
    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    with open(yaml_path, "wb") as f:
        f.write(await yaml_file.read())

    # 2. แตกไฟล์ dataset.zip
    # ลบโฟลเดอร์ dataset เดิมทิ้งก่อนเพื่อป้องกันปัญหา
    dataset_root = os.path.join(UPLOAD_DIR, "dataset")
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
    os.makedirs(dataset_root, exist_ok=True)
    
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(dataset_root)

    # 3. แก้ไขพาธใน config.yaml ให้ถูกต้อง
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # YOLOv8 จะใช้ 'path' เป็นพาธหลักของ dataset
    # ส่วน 'train' และ 'val' เป็นโฟลเดอร์ย่อยภายในพาธหลัก
    # หาชื่อโฟลเดอร์หลักที่ถูกแตกจาก zip ไฟล์
    dataset_subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    
    # ถ้ามีโฟลเดอร์ย่อยเพียงอันเดียว ให้ใช้โฟลเดอร์นั้นเป็นพาธหลัก
    if len(dataset_subdirs) == 1:
        base_path = os.path.join(dataset_root, dataset_subdirs[0])
        yaml_data["path"] = base_path
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
    else:
        # กรณี zip ไฟล์มีโฟลเดอร์ train และ valid อยู่ข้างในโดยตรง
        yaml_data["path"] = dataset_root
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"

    # บันทึกไฟล์ YAML ที่แก้ไขแล้วทับไฟล์เดิม
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    return {"status": "uploaded", "dataset_path": dataset_root, "yaml_path": yaml_path}

@app.post("/train")
def train_model(req: TrainRequest):
    """
    เริ่มกระบวนการฝึกโมเดล YOLOv8
    """
    global process

    with open(LOG_FILE, "w") as f:
        f.write("Starting training...\n")

    # ใช้พาธของไฟล์ config.yaml ที่ได้รับการแก้ไขแล้วในโฟลเดอร์ UPLOAD_DIR
    config_path = os.path.join(UPLOAD_DIR, 'config.yaml')
    
    cmd = [
        "yolo", "detect", "train",
        f"data={config_path}",
        "model=yolov8n.pt",
        "epochs=50",
        "project=./runs",
        "name=exp1"
    ]
    if req.resume:
        cmd.append("resume=True")

    with open(LOG_FILE, "a") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    return {"status": "started", "resume": req.resume}

@app.post("/stop")
def stop_train():
    """
    หยุดกระบวนการฝึกโมเดลที่กำลังทำงานอยู่
    """
    global process
    if process and process.poll() is None:
        process.terminate()
        return {"status": "stopped"}
    return {"status": "no process running"}

@app.get("/logs")
def get_logs():
    """
    แสดงล็อกของกระบวนการฝึก
    """
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return {"logs": f.read()}
    return {"logs": "No logs yet."}

@app.get("/download")
def download_model():
    """
    ตรวจสอบและให้ลิงก์สำหรับดาวน์โหลดโมเดลที่ดีที่สุด
    """
    best_model = os.path.join(MODEL_DIR, "best.pt")
    if os.path.exists(best_model):
        return {"download_url": best_model}
    return {"error": "Model not found yet."}
