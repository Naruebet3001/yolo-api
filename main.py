from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import subprocess, os, signal, zipfile, yaml, shutil

app = FastAPI()

UPLOAD_DIR = "./uploads"
DATASET_DIR = "./dataset"
LOG_DIR = "./logs"
MODEL_DIR = "./runs/exp1/weights"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "train.log")
process = None

class TrainRequest(BaseModel):
    resume: bool = False

@app.get("/")
def root():
    return {"message": "YOLOv8 API is running on Render!"}

@app.post("/upload")
async def upload_files(dataset: UploadFile = File(...), yaml_file: UploadFile = File(...)):
    dataset_path = os.path.join(UPLOAD_DIR, "dataset.zip")
    yaml_path = os.path.join(UPLOAD_DIR, "config.yaml")

    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    with open(yaml_path, "wb") as f:
        f.write(await yaml_file.read())

    # แตก dataset
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)

    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    # หาโฟลเดอร์ dataset (เช่น dataset/train, dataset/valid)
    dataset_subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    dataset_root = os.path.join(DATASET_DIR, dataset_subdirs[0]) if dataset_subdirs else DATASET_DIR

    # แก้ path ใน yaml
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data["train"] = os.path.join(dataset_root, "train/images")
    yaml_data["val"] = os.path.join(dataset_root, "valid/images")

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    return {"status": "uploaded", "yaml": yaml_path, "dataset_root": dataset_root}

@app.post("/train")
def train_model(req: TrainRequest):
    global process

    with open(LOG_FILE, "w") as f:
        f.write("Starting training...\n")

    cmd = [
        "yolo", "detect", "train",
        f"data={os.path.join(UPLOAD_DIR, 'config.yaml')}",
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
    global process
    if process:
        process.terminate()
        return {"status": "stopped"}
    return {"status": "no process running"}

@app.get("/logs")
def get_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return {"logs": f.read()}
    return {"logs": "No logs yet."}

@app.get("/download")
def download_model():
    best_model = os.path.join(MODEL_DIR, "best.pt")
    if os.path.exists(best_model):
        return {"download_url": best_model}
    return {"error": "Model not found yet."}
