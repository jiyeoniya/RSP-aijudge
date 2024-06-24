from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="./datasets/RPS/data.yaml", epochs=10, batch=8, imgsz=640)
