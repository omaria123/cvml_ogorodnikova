import os
os.environ['TORCH_HOME'] = 'D:/active_learing' 
os.environ['TORCH_MODEL_ZOO'] = 'D:/active_learing'  

import torch
import torch.nn as nn
import cv2
import torchvision
from torchvision import transforms
import time 
from collections import deque
from pathlib import Path 

save_path = Path(__file__).parent

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    
    return model

model = build_model()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) 
])

def train(buffer):
    if len(buffer) < 10:
        return None
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = "person" if prob > 0.5 else "no_person"
    return label, prob 

class Buffer():
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)
        
    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)
        
    def __len__(self):
        return len(self.frames)
        
    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
    buffer = Buffer(maxsize=128)
    count_labeled = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFf
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if key == ord("q"):
            break
        elif key == ord("1"):
            tensor = transform(image)
            buffer.append(tensor, 1.0)
            count_labeled += 1
            print("person")
        elif key == ord("2"):
            tensor = transform(image)
            buffer.append(tensor, 0.0)
            count_labeled += 1
            print("no_person")
        elif key == ord("p"):
            t = time.perf_counter()
            label, confidence = predict(frame)
            print(f"Elapsed time {time.perf_counter() - t}")
            print(f"{label} ({confidence:.2f})")
        elif key == ord("s"):
            torch.save(model.state_dict(), save_path / "model.pth")

        if count_labeled >= buffer.frames.maxlen:
            loss = train(buffer)
            if loss:
                print(f"Loss = {loss:.4f}")
            count_labeled = 0
            
    cap.release()
    cv2.destroyAllWindows()