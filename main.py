from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import threading
import os
import shutil

app = Flask(__name__)

# โฟลเดอร์สำหรับเก็บไฟล์ที่อัปโหลดและผลลัพธ์
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# สถานะการเทรน
# ใช้ dict แทนการใช้ไฟล์ เพื่อให้จัดการได้ง่ายขึ้น (เหมาะสำหรับ Threading)
training_status = {}

def train_model(training_id, yaml_path, zip_path):
    try:
        # อัปเดตสถานะเป็น "training"
        training_status[training_id] = {'status': 'training', 'log': ''}
        
        # คลายไฟล์ dataset.zip
        shutil.unpack_archive(zip_path, f'{UPLOAD_FOLDER}/{training_id}/dataset')

        # เริ่มเทรน
        model = YOLO('yolov8n.pt')  # ใช้โมเดลที่ Pre-trained แล้ว
        results = model.train(data=yaml_path, epochs=5, project=f'{UPLOAD_FOLDER}/{training_id}', name='runs/detect/train')

        # เมื่อเทรนเสร็จสิ้น
        training_status[training_id]['status'] = 'completed'
        
    except Exception as e:
        # กรณีเกิดข้อผิดพลาด
        training_status[training_id]['status'] = 'error'
        training_status[training_id]['log'] = str(e)
        print(f"Error during training: {e}")

@app.route('/train', methods=['POST'])
def start_training():
    data = request.form
    training_id = data.get('training_id')
    yaml_path = data.get('yaml_path')
    zip_path = data.get('zip_path')

    if not all([training_id, yaml_path, zip_path]):
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400

    # รันการเทรนใน background thread
    thread = threading.Thread(target=train_model, args=(training_id, yaml_path, zip_path))
    thread.start()
    
    return jsonify({'status': 'training_started', 'training_id': training_id})

@app.route('/status', methods=['GET'])
def get_status():
    training_id = request.args.get('id')
    if training_id not in training_status:
        return jsonify({'status': 'not_found', 'log_output': 'Training ID not found.'}), 404
        
    status = training_status[training_id]['status']
    log_output = training_status[training_id]['log']
    
    # ดึง log จากไฟล์ล่าสุด
    if status == 'training':
        try:
            log_file = f'{UPLOAD_FOLDER}/{training_id}/runs/detect/train/results.csv' # YOLOv8 บันทึก log ในไฟล์นี้
            with open(log_file, 'r') as f:
                log_output = f.read()
        except FileNotFoundError:
            log_output = 'Training has started, but logs are not yet available.'

    return jsonify({'status': status, 'log_output': log_output})

@app.route('/stop', methods=['GET'])
def stop_training():
    training_id = request.args.get('id')
    if training_id in training_status:
        training_status[training_id]['status'] = 'stopped'
        # Note: การหยุด thread โดยตรงทำได้ยากและไม่ปลอดภัย
        # วิธีนี้จะแค่เปลี่ยนสถานะให้ระบบรู้ว่าควรหยุด แต่การเทรนจริงจะรันจนจบ epoch
        return jsonify({'status': 'training_stopped'})
    return jsonify({'status': 'error', 'message': 'Training ID not found.'}), 404

@app.route('/download', methods=['GET'])
def download_model():
    training_id = request.args.get('id')
    model_path = f'{UPLOAD_FOLDER}/{training_id}/runs/detect/train/weights/best.pt'
    
    if os.path.exists(model_path):
        return send_from_directory(
            os.path.dirname(model_path), 
            os.path.basename(model_path), 
            as_attachment=True
        )
    else:
        return jsonify({'status': 'error', 'message': 'Model not found.'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
