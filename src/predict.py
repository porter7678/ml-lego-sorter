print('Beginning imports...')
from collections import Counter
from datetime import datetime, timedelta
import os
import time

import cv2 as cv
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, datasets
import matplotlib.pyplot as plt

from data import LegoDataset, get_inference_transforms
from servo import Servo

class_to_idx = {'2x2': 0, '2x4': 1, 'blank': 2}
idx_to_class = {v:k for k, v in class_to_idx.items()}
NUM_CLASSES = 3

def predict(image, model):
    transforms = get_inference_transforms()

    with torch.no_grad():
        image = transforms(image)
        image = image.unsqueeze(0)
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
    maxiter = 30
    start = time.time()
    for image_path, label in image_folder.imgs:
        i += 1
        image = plt.imread(image_path)
        pred = predict(image, model)
        print(image_path, pred, '----', idx_to_class[pred])
        
        if i == maxiter:
            break
    end = time.time()
    print(f'Total time: {end-start:.3f}')

def capture_init():
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv.CAP_PROP_FPS, 36)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    return cap

def predict_live(model_path, left_label, right_label, move_interval_seconds=10, num_votes=5):
    print('Enter predict_live')

    cap = capture_init()
    model = load_resnet50(model_path, NUM_CLASSES)
    servo = Servo(left_label, right_label)

    next_move_time = datetime.now() + timedelta(seconds=5)
    pred_stack = []

    maxiter = 1000
    i = 0
    try:
        while True:
            cap_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            ret, image = cap.read()
            if not ret:
                raise RuntimeError('Failed to read frame')
            
            pred = predict(image, model)
            pred_stack.append(pred)

            print('\nPred:', pred, '---', idx_to_class[pred])
            print('Cap time', cap_time) 
            print('Now time', datetime.now().strftime("%H:%M:%S.%f")[:-3])

            i += 1
            if i == maxiter:
                print('Hit maxiter')
                break

            if datetime.now() >= next_move_time:
                recent_preds = pred_stack[-num_votes:]
                votes = [idx_to_class[vote] for vote in recent_preds if vote != class_to_idx['blank']]
                if votes:
                    counts = Counter(votes)
                    selected_vote = counts.most_common(1)[0][0]
                    
                    servo.move_arm(selected_vote)
                    
                else:
                    print('\n~~~Skipping arm move')

                next_move_time = datetime.now() + timedelta(seconds=move_interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        servo.cleanup()

    
    print('Exiting predict_live')





if __name__ == '__main__':
    model_path = 'checkpoints/resnet50_porter_data1.pt'
    image_dir = 'porter_data'
    # predict_offline(image_dir, model_path)

    left_label = '2x2'
    right_label = '2x4'
    predict_live(model_path, left_label, right_label, move_interval_seconds=3)

