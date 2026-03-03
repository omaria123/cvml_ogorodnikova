import cv2
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

def auto_decode_class(class_name):
    if len(class_name) == 2:
        return class_name[1]
    return class_name

def extractor(image):
    if image.ndim == 2:
        binary = image > 0
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 100
    
    lb = label(binary)
    props = regionprops(lb)
    
    if len(props) == 0:
        return np.zeros(6, dtype="f4")
    
    largest = max(props, key=lambda x: x.area)
    features = np.zeros(6, dtype="f4")
    features[0] = largest.eccentricity
    features[1] = largest.extent
    features[2] = largest.solidity
    features[3] = largest.orientation
    features[4] = largest.perimeter / largest.area 
    features[5] = largest.area_convex/largest.area
    return features

def make_train(path):
    train = []
    responses = []
    ncls = 0
    class_names = []
    
    for cls in sorted(path.glob("*")):
        if not cls.is_dir():
            continue
        decoded = auto_decode_class(cls.name)
        class_names.append(decoded)
        ncls += 1
        
        for p in cls.glob("*.png"):
            image = imread(p)
            gray = np.mean(image, 2).astype("u1")
            train_binary = gray > 0
            feat = extractor(train_binary.astype(np.uint8) * 255)
            train.append(feat)
            responses.append(ncls)
    
    train = np.array(train, dtype="f4").reshape(-1, 6)
    responses = np.array(responses, dtype="f4").reshape(-1, 1)
    return train, responses, class_names

def recognize_text(image_path, knn, class_names):
    image = imread(image_path)
    gray = np.mean(image, 2).astype("u1")
    binary = gray > 0
    
    lb = label(binary)
    props = regionprops(lb)
    
    valid_props = [p for p in props 
                  if 10 < (p.bbox[2]-p.bbox[0]) < 150 and 
                     10 < (p.bbox[3]-p.bbox[1]) < 150 and 
                     p.area > 30]
    valid_props.sort(key=lambda x: x.bbox[1])
    
    text = ""
    last_x = -100  
    
    for i, prop in enumerate(valid_props):
        if i > 0 and prop.bbox[1] - last_x > 30:  
            text += " "
        
        minr, minc, maxr, maxc = prop.bbox
        symbol = binary[minr:maxr, minc:maxc].astype(np.uint8) * 255
        feat = extractor(symbol).reshape(1, -1)
        
        ret, _, _, _ = knn.findNearest(feat, 3)  
        char_id = int(ret)
        char = class_names[char_id-1] 
        text += char
        
        last_x = maxc  
   
    text = text.replace('-i', 'i')  
    text = text.replace('i-', 'i')  

    return text

data_path = Path("task")

train_path = data_path / "train"
train, responses, class_names = make_train(train_path)
    
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

#print(f" Классы: {sorted(set(class_names))}")
    
test_images = sorted(data_path.glob("[0-6].png"))
for i, img_path in enumerate(test_images):
    text = recognize_text(img_path, knn, class_names)
    print(f"Изображение {i}: '{text}'")