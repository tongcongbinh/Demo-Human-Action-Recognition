from ultralytics import YOLO
# Load YOLO model
model = YOLO("yolov8n-pose.pt")

#source = "rtsp://admin:Qazxsw123@192.168.88.20:554/cam/realmonitor?channel=1&subtype=0"
#source = "0"
source = ".\\video\\class.mp4"

# Perform inference
results = model.predict(source=source, show = True)
print(results)
