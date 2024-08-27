print('Beginning imports...')
import time
import os

import cv2 as cv
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, datasets
import matplotlib.pyplot as plt

from data import LegoDataset, get_inference_transforms

# TODO: This shouldn't be hard coded
class_to_idx = {'Brick_1x1': 0, 'Brick_1x2': 1, 'Brick_1x3': 2, 'Brick_1x4': 3, 'Brick_2x2': 4, 'Brick_2x2_L': 5, 'Brick_2x2_Slope': 6, 'Brick_2x3': 7, 'Brick_2x4': 8, 'Plate_1x1': 9, 'Plate_1x1_Round': 10, 'Plate_1x1_Slope': 11, 'Plate_1x2': 12, 'Plate_1x2_Grill': 13, 'Plate_1x3': 14, 'Plate_1x4': 15, 'Plate_2x2': 16, 'Plate_2x2_L': 17, 'Plate_2x3': 18, 'Plate_2x4': 19}
NUM_CLASSES = 20

def predict(image, model):
    transforms = get_inference_transforms()

    image = transforms(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    _, pred = torch.max(output, 1)
    return pred.item()


def load_resnet50(model_path, num_classes):
    model = models.resnet50()

    out_ftrs = num_classes
    in_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_ftrs, out_ftrs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    return model


def predict_offline(image_dir, model_path, split='test'):
    image_folder = datasets.ImageFolder(image_dir)
    num_classes = NUM_CLASSES  # FIXME
    print('num_classes', num_classes)

    model = load_resnet50(model_path, num_classes)

    i = 0
    start = time.time()
    for image_path, label in image_folder.imgs:
        i += 1
        image = plt.imread(image_path)
        pred = predict(image, model)
        print(image_path, pred, '----', list(class_to_idx.keys())[pred])
        
        if i == 30:
            break
    end = time.time()
    print(f'Total time: {end-start:.3f}')

def predict_live(model_path):
    print('Enter predict_live')
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv.CAP_PROP_FPS, 36)

    model = load_resnet50(model_path, NUM_CLASSES)


    maxiter = 100
    i = 0
    with torch.no_grad():
        while True:
            ret, image = cap.read()
            if not ret:
                raise RuntimeError('Failed to read frame')
            
            pred = predict(image, model)

            print('Pred:', pred, '---', list(class_to_idx.keys())[pred])

            i += 1
            if i == maxiter:
                print('Hit maxiter')
                break

    
    print('Exiting predict_live')





if __name__ == '__main__':
    model_path = 'checkpoints/resnet50_kaggle_only.pt'
    image_dir = 'porter_data'
    # predict_offline(image_dir, model_path)

    predict_live(model_path)

