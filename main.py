import os
import shutil
import subprocess
import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the YOLOv8 Training API."}

RENDER_DISK_PATH = os.getenv("RENDER_DISK_PATH") or "./training_sessions"
if not os.path.exists(RENDER_DISK_PATH):
    os.makedirs(RENDER_DISK_PATH)

training_processes = {}

class TrainingRequest(BaseModel):
    training_id: str

def run_training_in_background(training_id: str, unzip_dir: str, config_file: str):
    log_file = os.path.join(RENDER_DISK_PATH, f"{training_id}_log.txt")
    
    p = subprocess.Popen(
        ["python", "training_worker.py", training_id, unzip_dir, config_file, RENDER_DISK_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    training_processes[training_id] = {'process': p, 'status': 'training', 'log': ''}
    
    try:
        for line in p.stdout:
            with open(log_file, "a") as f:
                f.write(line)
        p.wait()
        
        if p.returncode == 0:
            training_processes[training_id]['status'] = 'completed'
        else:
            training_processes[training_id]['status'] = 'failed'
            
    except Exception as e:
        training_processes[training_id]['status'] = 'failed'
        with open(log_file, "a") as f:
            f.write(f"An error occurred: {str(e)}\n")

@app.post("/upload-and-train")
async def upload_and_train(
    training_id: str = Form(...),
    dataset_zip: UploadFile = File(...),
    config_yaml: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        session_dir = os.path.join(RENDER_DISK_PATH, training_id)
        os.makedirs(session_dir, exist_ok=True)
        
        dataset_path = os.path.join(session_dir, dataset_zip.filename)
        config_path = os.path.join(session_dir, config_yaml.filename)
        
        with open(dataset_path, "wb") as f:
            shutil.copyfileobj(dataset_zip.file, f)
            
        with open(config_path, "wb") as f:
            shutil.copyfileobj(config_yaml.file, f)
            
        background_tasks.add_task(run_training_in_background, training_id, session_dir, config_path)
        
        return {"status": "success", "message": "Files uploaded and training started.", "training_id": training_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during upload: {str(e)}")

@app.post("/pause")
async def pause_training(request: TrainingRequest):
    if request.training_id not in training_processes:
        raise HTTPException(status_code=404, detail="Training not found.")
    proc = training_processes[request.training_id]['process']
    psutil.Process(proc.pid).suspend()
    training_processes[request.training_id]['status'] = 'paused'
    return {"status": "success", "message": "Training paused."}

@app.post("/resume")
async def resume_training(request: TrainingRequest):
    if request.training_id not in training_processes:
        raise HTTPException(status_code=404, detail="Training not found.")
    proc = training_processes[request.training_id]['process']
    psutil.Process(proc.pid).resume()
    training_processes[request.training_id]['status'] = 'training'
    return {"status": "success", "message": "Training resumed."}

@app.post("/stop")
async def stop_training(request: TrainingRequest):
    if request.training_id not in training_processes:
        raise HTTPException(status_code=404, detail="Training not found.")
    proc = training_processes[request.training_id]['process']
    proc.terminate()
    training_processes[request.training_id]['status'] = 'stopped'
    del training_processes[request.training_id]
    return {"status": "success", "message": "Training stopped."}

@app.get("/progress/{training_id}")
async def get_progress(training_id: str):
    log_file = os.path.join(RENDER_DISK_PATH, f"{training_id}_log.txt")
    if not os.path.exists(log_file):
        return {"status": "not_found", "message": "Log file not found."}
    
    with open(log_file, "r") as f:
        log_content = f.read()

    status = training_processes.get(training_id, {}).get('status', 'not_found')
    if status == 'not_found' and "Training completed successfully" in log_content:
        status = 'completed'
    
    return {"status": status, "log": log_content}

@app.get("/download/{training_id}")
async def download_model(training_id: str):
    model_path = os.path.join(RENDER_DISK_PATH, "runs", "detect", f"train_yolov8_{training_id}", "weights", "best.pt")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")

    return FileResponse(
        path=model_path,
        filename=f"yolov8_{training_id}_best.pt",
        media_type='application/octet-stream'
    )
