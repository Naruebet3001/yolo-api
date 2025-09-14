import sys
import zipfile
import os
import shutil
from ultralytics import YOLO

def run_yolo_training(training_id: str, session_dir: str, config_path: str, output_dir: str):
    print(f"Training ID: {training_id}")
    print(f"Session directory: {session_dir}")
    print(f"Config path: {config_path}")

    try:
        # Find the uploaded zip file
        zip_files = [f for f in os.listdir(session_dir) if f.endswith('.zip')]
        if not zip_files:
            raise FileNotFoundError("No .zip file found in the session directory.")
        
        dataset_zip_path = os.path.join(session_dir, zip_files[0])
        unzip_dir = os.path.join(session_dir, 'unzipped_dataset')
        
        # Unzip dataset
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
            
        # The training data is the unzipped directory
        yolo_data_path = os.path.join(unzip_dir, os.path.basename(config_path))

        # Start training
        model = YOLO("yolov8n.pt")
        model.train(
            data=yolo_data_path,
            epochs=100,
            imgsz=640,
            project=output_dir,
            name=f'train_yolov8_{training_id}'
        )
        print("Training completed successfully.")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python training_worker.py <training_id> <session_dir> <config_path> <output_dir>")
        sys.exit(1)
    
    training_id = sys.argv[1]
    session_dir = sys.argv[2]
    config_path = sys.argv[3]
    output_dir = sys.argv[4]
    
    run_yolo_training(training_id, session_dir, config_path, output_dir)
