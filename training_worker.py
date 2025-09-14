import sys
import zipfile
import os
import shutil
from ultralytics import YOLO

def run_yolo_training(training_id: str, dataset_path: str, config_path: str, output_dir: str):
    print(f"Training ID: {training_id}")
    print(f"Dataset path: {dataset_path}")
    print(f"Config path: {config_path}")

    try:
        unzip_dir = os.path.join(os.path.dirname(dataset_path), 'dataset')
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
            
        shutil.copy(config_path, unzip_dir)

        model = YOLO("yolov8n.pt")
        model.train(
            data=os.path.join(unzip_dir, os.path.basename(config_path)),
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
        print("Usage: python training_worker.py <training_id> <dataset_path> <config_path> <output_dir>")
        sys.exit(1)
    
    training_id = sys.argv[1]
    dataset_path = sys.argv[2]
    config_path = sys.argv[3]
    output_dir = sys.argv[4]
    
    run_yolo_training(training_id, dataset_path, config_path, output_dir)
