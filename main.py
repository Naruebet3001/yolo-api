from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import threading
import uuid
import yaml
import zipfile
import asyncio

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a temporary directory within the project's working directory
TEMP_DIR = "/tmp/yolo_data" 
UPLOAD_DIR = os.path.join(TEMP_DIR, "uploads")
MODEL_DIR = os.path.join(TEMP_DIR, "trained_models")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global dictionary to track training jobs
training_jobs = {}

class TrainingJob:
    def __init__(self, job_id, user_id):
        self.job_id = job_id
        self.user_id = user_id
        self.status = "pending"
        self.progress = 0
        self.model_path = None
        self.thread = None
        self.is_stopped = threading.Event()
        self.data_dir = None

    def set_status(self, status):
        self.status = status

    def set_progress(self, progress):
        self.progress = progress

    def set_model_path(self, path):
        self.model_path = path

    def stop_training(self):
        self.is_stopped.set()

def train_yolo_model(job_id: str, data_path: str, model_name: str, job: TrainingJob):
    try:
        job.set_status("training")
        
        model = YOLO("yolov8n.pt")
        
        results = model.train(
            data=data_path,
            epochs=10,
            imgsz=640,
        )
        
        job.set_status("completed")
        final_model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        model.export(format="torchscript", filename=final_model_path)
        job.set_model_path(final_model_path)
        print(f"Job {job_id} completed. Model saved to {final_model_path}")

    except Exception as e:
        job.set_status("failed")
        print(f"Job {job_id} failed: {e}")
    finally:
        if job.data_dir and os.path.exists(job.data_dir):
            shutil.rmtree(job.data_dir)
            print(f"Cleaned up data directory: {job.data_dir}")
        job.thread = None

@app.post("/upload_and_train/")
async def upload_and_train(dataset: UploadFile = File(...), yaml_file: UploadFile = File(...)):
    """Endpoint to upload dataset and YAML file, then start training."""
    job_id = str(uuid.uuid4())
    user_id = "user_123"

    try:
        job_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        zip_path = os.path.join(job_dir, dataset.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(dataset.file, buffer)
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(job_dir)
        os.remove(zip_path)

        yaml_path = os.path.join(job_dir, yaml_file.filename)
        with open(yaml_path, "wb") as buffer:
            shutil.copyfileobj(yaml_file.file, buffer)
        
        extracted_dirs = [d for d in os.listdir(job_dir) if os.path.isdir(os.path.join(job_dir, d))]
        if not extracted_dirs:
            raise Exception("No dataset folder found inside the zip file.")
            
        dataset_root = os.path.join(job_dir, extracted_dirs[0])
        
        # --- ส่วนที่แก้ไข ---
        # อ่านและอัปเดต path ในไฟล์ YAML ให้เป็นแบบ Absolute Path
       # อ่าน YAML
        # อ่าน YAML
        with open(yaml_path, "r") as f:
            data_config = yaml.safe_load(f)
        
        # เปลี่ยน train/val เป็น absolute path
        if 'train' in data_config:
            data_config['train'] = os.path.abspath(os.path.join(dataset_root, data_config['train']))
        
        if 'val' in data_config:
            val_path = os.path.join(dataset_root, data_config['val'])
            if not os.path.exists(val_path):
                # ถ้า val ไม่มี ให้ใช้ train แทน
                val_path = data_config['train']
            data_config['val'] = os.path.abspath(val_path)
        
        # เขียน YAML กลับ
        with open(yaml_path, "w") as f:
            yaml.dump(data_config, f)


        # --- สิ้นสุดการแก้ไข ---

        job = TrainingJob(job_id, user_id)
        job.data_dir = job_dir
        training_jobs[job_id] = job
        
        job.thread = threading.Thread(
            target=train_yolo_model,
            args=(job_id, yaml_path, f"model_{job_id}", job)
        )
        job.thread.start()

        return JSONResponse(
            content={"status": "Training started", "job_id": job_id},
            status_code=202
        )

    except Exception as e:
        return JSONResponse(
            content={"status": "Failed to start training", "error": str(e)},
            status_code=500
        )

@app.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Endpoint to get the status and progress of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    
    return JSONResponse(
        content={
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "model_path": job.model_path
        }
    )

@app.post("/stop/{job_id}")
async def stop_training(job_id: str):
    """Endpoint to stop a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    if job.status == "training":
        return JSONResponse(content={"status": "Stop command sent, but feature is disabled"})
    else:
        return JSONResponse(content={"status": "Job is not currently training"})

@app.get("/download/{job_id}")
async def download_model(job_id: str):
    """Endpoint to download the trained model file."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    if job.status != "completed" or not job.model_path:
        raise HTTPException(status_code=400, detail="Model is not ready for download")

    file_path = job.model_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
        
    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type='application/octet-stream')



