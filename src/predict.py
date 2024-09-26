print("Beginning imports...")
import os
import time
from collections import Counter
from datetime import datetime, timedelta

import cv2 as cv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models

from data import LegoDataset, get_inference_transforms
from servo import MyServo, ServoThread

class_to_idx = {"2x2": 0, "2x4": 1, "blank": 2}
idx_to_class = {v: k for k, v in class_to_idx.items()}
NUM_CLASSES = 3
BLANK_CLASS_IDX = 2


def predict(image, model, transforms):
    with torch.no_grad():
        image = transforms(image)
        image = image.unsqueeze(0)
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        _, pred = torch.max(probabilities, 1)

    return pred.item(), probabilities


def load_resnet50(model_path, num_classes):
    model = models.resnet50()

    out_ftrs = num_classes
    in_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_ftrs, out_ftrs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    transforms = get_inference_transforms("resnet50")
    model.eval()
    return model, transforms


def load_mobilenetv3large(model_path, num_classes):
    model = models.mobilenet_v3_large()

    out_ftrs = num_classes
    in_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_ftrs, out_ftrs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    transforms = get_inference_transforms("mobilenetv3large")
    model.eval()
    return model, transforms


def predict_offline(image_dir, model_path, split="test"):
    image_folder = datasets.ImageFolder(image_dir)
    num_classes = NUM_CLASSES
    print("num_classes", num_classes)

    model, transforms = load_resnet50(model_path, num_classes)

    i = 0
    maxiter = 30
    start = time.time()
    for image_path, label in image_folder.imgs:
        i += 1
        image = plt.imread(image_path)
        pred = predict(image, model, transforms)
        print(image_path, pred, "----", idx_to_class[pred])

        if i == maxiter:
            break
    end = time.time()
    print(f"Total time: {end-start:.3f}")


def capture_init():
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv.CAP_PROP_FPS, 36)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    return cap


def select_from_votes(pred_stack, num_votes, votes_needed=3):
    recent_preds = pred_stack[-num_votes:]
    votes = [
        idx_to_class[vote] for vote in recent_preds if vote != class_to_idx["blank"]
    ]
    selected_vote = None
    if votes:
        counts = Counter(votes)
        top_vote, occurences = counts.most_common(1)[0]
        if occurences >= votes_needed:
            selected_vote = top_vote

    return selected_vote


def predict_live(
    model_path,
    left_label,
    right_label,
    move_interval_seconds=3,
    num_votes=15,
    blank_class_threshold=0.1,
):
    print("Enter predict_live")

    cap = capture_init()
    model, transforms = load_mobilenetv3large(model_path, NUM_CLASSES)
    servo = MyServo(left_label, right_label)
    servo_thread = ServoThread(servo)
    servo_thread.start()

    next_move_time = datetime.now() + timedelta(seconds=3)
    pred_stack = []

    try:
        while True:
            ret, image = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame")

            pred, probabilities = predict(image, model, transforms)

            # Err on the side of picking "blank"
            blank_class_prob = probabilities[0][BLANK_CLASS_IDX]
            if blank_class_prob > blank_class_threshold:
                pred = BLANK_CLASS_IDX

            pred_stack.append(pred)

            print(
                "Pred:",
                pred,
                "---",
                idx_to_class[pred],
                f"\t---({probabilities[0][0]:.2f} vs. {probabilities[0][1]:.2f}  vs. {probabilities[0][BLANK_CLASS_IDX]:.2f})",
            )

            if datetime.now() >= next_move_time:
                selected_vote = select_from_votes(pred_stack, num_votes)
                if selected_vote is None:
                    print("\n~~~Skipping arm move")
                else:
                    print(f"\n~~~Moving arm: {selected_vote}\n")
                    servo_thread.add_command(selected_vote)
                next_move_time = datetime.now() + timedelta(
                    seconds=move_interval_seconds
                )
    except KeyboardInterrupt:
        pass
    finally:
        servo_thread.stop()
        servo_thread.join()

    print("Exiting predict_live")


if __name__ == "__main__":
    model_path = "checkpoints/mobilenetv3large_porter_data5.pt"
    image_dir = "porter_data"
    # predict_offline(image_dir, model_path)

    left_label = "2x2"
    right_label = "2x4"
    predict_live(model_path, left_label, right_label)
