import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np

counter = 0
stage = "up" 
last_seen_time = time.time()

def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

def detect_pushups(annotated, keypoints):
    global counter, stage, last_seen_time
    
    l_shoulder, l_elbow, l_wrist = keypoints[5], keypoints[7], keypoints[9]
    l_hip, l_knee = keypoints[11], keypoints[13]
    r_shoulder, r_elbow, r_wrist = keypoints[6], keypoints[8], keypoints[10]
    r_hip, r_knee = keypoints[12], keypoints[14]

    if l_shoulder[0] > 0 and l_elbow[0] > 0:
        arm_angle = get_angle(l_shoulder, l_elbow, l_wrist)
        body_angle = get_angle(l_shoulder, l_hip, l_knee) 
        is_horizontal = abs(l_shoulder[1] - l_hip[1]) < abs(l_shoulder[0] - l_hip[0])
    elif r_shoulder[0] > 0 and r_elbow[0] > 0:
        arm_angle = get_angle(r_shoulder, r_elbow, r_wrist)
        body_angle = get_angle(r_shoulder, r_hip, r_knee)
        is_horizontal = abs(r_shoulder[1] - r_hip[1]) < abs(r_shoulder[0] - r_hip[0])
    else:
        return False

    last_seen_time = time.time() 

    if body_angle > 120 and is_horizontal: 
        
        if arm_angle < 110:
            stage = "down"
            
        if arm_angle > 155 and stage == "down":
            stage = "up"
            counter += 1
            return True 
            
    return False
model = YOLO("yolo26n-pose.pt") 
camera = cv2.VideoCapture(0)
ps = None

while camera.isOpened():
    ret, frame = camera.read()
    if not ret: break

    if time.time() - last_seen_time > 5:
        counter = 0

    results = model(frame, verbose=False)
    
    annotated_frame = frame.copy()

    if results and len(results[0].keypoints.data) > 0 and results[0].keypoints.data.shape[0] > 0:
        result = results[0]
        keypoints = result.keypoints.xy.tolist()

        annotator = Annotator(annotated_frame)
        annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
        annotated_frame = annotator.result()

        if detect_pushups(annotated_frame, keypoints[0]):
            try:
                playsound("acolyteyes2.mp3", block=False)
            except:
                pass

    cv2.putText(annotated_frame, f"count: {counter}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
    
    cv2.imshow("Pose", annotated_frame)
    
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
