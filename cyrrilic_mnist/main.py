import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms
from train_model import CyrillicCNN

CLASSES = ['I', 'Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 
           'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 
           'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 
           'Ь', 'Э', 'Ю', 'Я']

model_path = Path(__file__).parent / "cyrillic_model.pth"

model = CyrillicCNN(num_classes=34)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

canvas = np.zeros((256, 256), dtype="uint8")
cv2.namedWindow("Draw", cv2.WINDOW_GUI_NORMAL)

position = []
draw = False
threshold = 70

def on_mouse(event, x, y, flags, param):
    global draw
    global position
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    if event == cv2.EVENT_LBUTTONUP:
        draw = False
    if event == cv2.EVENT_MOUSEMOVE and draw:
        position = [y, x]

cv2.setMouseCallback("Draw", on_mouse)

def is_canvas_empty(canvas_img):
    return np.sum(canvas_img) < 100

while True:
    if position:
        cv2.circle(canvas, (position[1], position[0]), 8, 255, -1)
    
    if not is_canvas_empty(canvas):
        with torch.no_grad():
            canvas_inverted = 255 - canvas
            
            tensor = transform(canvas_inverted)
            batch = tensor.unsqueeze(0)
            
            output = model(batch)
            probabilities = torch.softmax(output, dim=1)
            top_prob, top_idx = torch.topk(probabilities, 3)
            
            top_probs = top_prob[0].cpu().numpy() * 100
            top_classes = [CLASSES[idx] for idx in top_idx[0].cpu().numpy()]
            
            if top_probs[0] >= threshold:
                print(f"\r{top_classes[0]} {top_probs[0]:.1f}% \n", end="")
            else:
                print(f"\rУверенность низкая ({top_probs[0]:.1f}%) \n", end="")
    
    key = cv2.waitKey(10) & 0xFF
    
    if key == 27:
        break
    elif key == 99:
        position = []
        canvas *= 0
    
    cv2.imshow("Draw", canvas)

cv2.destroyAllWindows()