from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse
import shutil, subprocess, os, signal

app = FastAPI()

UPLOAD_DIR = "datasets"
LOG_DIR = "logs"
RUNS_DIR = "runs"
PID_FILE = "train_pid.txt"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


@app.post("/upload")
async def upload(dataset: UploadFile = File(...), config: UploadFile = File(...)):
    dataset_path = os.path.join(UPLOAD_DIR, dataset.filename)
    config_path = os.path.join(UPLOAD_DIR, config.filename)

    with open(dataset_path, "wb") as f:
        shutil.copyfileobj(dataset.file, f)
    with open(config_path, "wb") as f:
        shutil.copyfileobj(config.file, f)

    return {"status": "uploaded", "dataset": dataset.filename, "config": config.filename}


@app.get("/train")
def train():
    log_file = os.path.join(LOG_DIR, "train_log.txt")
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            ["python", "training_worker.py"],
            stdout=log,
            stderr=log,
        )
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

    return {"status": "training started", "pid": process.pid}


@app.get("/stop")
def stop():
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read())
        try:
            os.kill(pid, signal.SIGKILL)
            return {"status": f"stopped training process {pid}"}
        except ProcessLookupError:
            return {"status": f"process {pid} not found"}
    return {"status": "no process running"}


@app.get("/resume")
def resume():
    log_file = os.path.join(LOG_DIR, "train_log.txt")
    with open(log_file, "a") as log:
        process = subprocess.Popen(
            ["python", "train.py", "--resume"],
            stdout=log,
            stderr=log,
        )
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

    return {"status": "resumed training", "pid": process.pid}


@app.get("/logs")
def get_logs():
    log_file = os.path.join(LOG_DIR, "train_log.txt")
    if os.path.exists(log_file):
        return FileResponse(log_file, media_type="text/plain")
    return PlainTextResponse("No logs yet.")


@app.get("/download")
def download_model():
    model_path = os.path.join(RUNS_DIR, "detect", "train", "weights", "best.pt")
    if os.path.exists(model_path):
        return FileResponse(model_path, filename="best.pt")
    return {"status": "no model yet"}
