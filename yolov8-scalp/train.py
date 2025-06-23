from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml') 
model = YOLO('yolov8n.pt') 
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  


results = model.train(data='skin.yaml', epochs=1000,batch=32, imgsz=640)

