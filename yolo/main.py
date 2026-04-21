from ultralytics import YOLO
import cv2
import time
import torch

model = YOLO("C:/Users/670/Desktop/ai/runs/detect/figures/yolo2/weights/best.pt")

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    t = time.perf_counter()
    
    results = model(frame, verbose=False)

    if results:
        result = results[0]
        
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{class_name}: {conf*100:.1f}%"
            
            cv2.putText(frame, label, (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Spheres and Cubes", frame)
    
    key = cv2.waitKey(10) & 0xFF 
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()