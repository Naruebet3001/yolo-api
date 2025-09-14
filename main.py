from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, zipfile, yaml, shutil, subprocess
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

app = FastAPI()

# ---------------- CONFIG ----------------
UPLOAD_DIR = "./uploads"
DATASET_DIR = "./dataset"
LOG_DIR = "./logs"
MODEL_DIR = "./runs/exp1/weights"
GDRIVE_FOLDER_ID = "1_7QhyLeXSQRSkXo57R3QeIBW7Q68WmMT"  # โฟลเดอร์ Google Drive

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "train.log")
process = None

# ---------------- AUTH GOOGLE DRIVE ----------------
gauth = GoogleAuth(settings={
    "client_config_backend": "service",
    "service_config": {
        "client_json_file_path": "credentials.json"  # ไฟล์ Service Account JSON ที่ต้องอัปโหลดไป Render
    }
})
gauth.ServiceAuth()  # ใช้ Service Account login
drive = GoogleDrive(gauth)

def upload_to_drive(local_path, parent_folder_id=GDRIVE_FOLDER_ID):
    file_drive = drive.CreateFile({
        'title': os.path.basename(local_path),
        'parents': [{'id': parent_folder_id}]
    })
    file_drive.SetContentFile(local_path)
    file_drive.Upload()
    return file_drive['id']

def download_from_drive(file_id, save_path):
    file_drive = drive.CreateFile({'id': file_id})
    file_drive.GetContentFile(save_path)

# ---------------- FastAPI MODELS ----------------
class TrainRequest(BaseModel):
    resume: bool = False
    dataset_id: str
    yaml_id: str

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"message": "YOLOv8 API with Google Drive is running!"}

@app.post("/upload")
async def upload_files(dataset: UploadFile = File(...), yaml_file: UploadFile = File(...)):
    dataset_path = os.path.join(UPLOAD_DIR, dataset.filename)
    yaml_path = os.path.join(UPLOAD_DIR, yaml_file.filename)

    # Save locally
    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    with open(yaml_path, "wb") as f:
        f.write(await yaml_file.read())

    # Upload to Google Drive
    dataset_id = upload_to_drive(dataset_path)
    yaml_id = upload_to_drive(yaml_path)

    # Cleanup local files
    os.remove(dataset_path)
    os.remove(yaml_path)

    return {"dataset_id": dataset_id, "yaml_id": yaml_id}

@app.post("/train")
def train_model(req: TrainRequest):
    global process

    # Download dataset + yaml from Google Drive
    dataset_zip_path = os.path.join(UPLOAD_DIR, "dataset.zip")
    yaml_path = os.path.join(UPLOAD_DIR, "config.yaml")
    download_from_drive(req.dataset_id, dataset_zip_path)
    download_from_drive(req.yaml_id, yaml_path)

    # Extract dataset
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)
    with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    # Update YAML paths
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    dataset_root = DATASET_DIR
    yaml_data["train"] = os.path.join(dataset_root, "train/images")
    yaml_data["val"] = os.path.join(dataset_root, "val/images")

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    # Start training
    with open(LOG_FILE, "w") as f:
        f.write("Starting training...\n")

    cmd = [
        "yolo", "detect", "train",
        f"data={yaml_path}",
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
