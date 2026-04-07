import torch
import torch.nn as nn
import cv2
import torchvision
from torchvision import transforms
from pathlib import Path

save_path = Path(__file__).parent

def build_model():
    model = torchvision.models.efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def predict(model, frame):
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(torch.tensor(predicted.item())).item() if predicted.ndim == 0 else torch.sigmoid(predicted).item()
        
    label = "Person" if prob > 0.5 else "No Person"
    return label, prob

if __name__ == "__main__":
    model_path = save_path / "model.pth"

    model = build_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        label, prob = predict(model, frame)
        
        color = (0, 255, 0) if label == "Person" else (0, 0, 255)
        text = f"{label}: {prob:.2f}"
        
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Inference Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()