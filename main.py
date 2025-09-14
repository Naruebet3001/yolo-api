import os
import shutil
import subprocess
import json
import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

# ใช้ Environment Variable สำหรับ Render Disk Path
RENDER_DISK_PATH = os.getenv("RENDER_DISK_PATH") or "./training_sessions"
if not os.path.exists(RENDER_DISK_PATH):
    os.makedirs(RENDER_DISK_PATH)

# Dictionary สำหรับเก็บข้อมูล Process
training_processes = {}

class TrainingRequest(BaseModel):
    training_id: str
    dataset_path: str = None
    config_path: str = None

def run_training_in_background(training_id: str, dataset_path: str, config_path: str):
    log_file = os.path.join(RENDER_DISK_PATH, f"{training_id}_log.txt")
    
    # Run training_worker.py in a subprocess
    p = subprocess.Popen(
        ["python", "training_worker.py", training_id, dataset_path, config_path, RENDER_DISK_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    training_processes[training_id] = {'process': p, 'status': 'training', 'log': ''}
    
    try:
        # Read stdout line by line and save to log file
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

@app.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    training_id = request.training_id
    dataset_path = request.dataset_path
    config_path = request.config_path

    # เริ่มการเทรนใน Background Task
    background_tasks.add_task(run_training_in_background, training_id, dataset_path, config_path)
    
    return {"status": "success", "message": "Training started.", "training_id": training_id}

# API สำหรับ Pause, Resume, Stop
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

# API สำหรับตรวจสอบ Progress
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
