import os
import time
from datetime import datetime, timedelta

import cv2 as cv
from PIL import Image


def auto_capture(label, color="", save_dir="porter_data"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv.CAP_PROP_FPS, 36)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    next_save_time = datetime.now() + timedelta(seconds=5)
    save_path = os.path.join(save_dir, label)
    os.makedirs(save_path, exist_ok=True)

    while True:
        ret, image = cap.read()

        if not ret:
            print("cant receive frame. Exiting...")
            break

        if datetime.now() >= next_save_time:
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
            image_filename = os.path.join(save_path, f"{color}_{timestamp}.jpg")
            cv.imwrite(image_filename, image)
            print(f"Saved {image_filename}")
            next_save_time = datetime.now() + timedelta(seconds=1)

        cv.imshow("frame", image)
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def trigger_capture(label, color="", save_dir="porter_data"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv.CAP_PROP_FPS, 36)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    save_path = os.path.join(save_dir, label)
    os.makedirs(save_path, exist_ok=True)

    while True:
        ret, image = cap.read()

        if not ret:
            print("cant receive frame. Exiting...")
            break

        cv.imshow("frame", image)

        key = cv.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord(" "):  # Spacebar pressed
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
            image_filename = os.path.join(save_path, f"{color}_{timestamp}.jpg")
            cv.imwrite(image_filename, image)
            print(f"Saved {image_filename}")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    label = "2x4"
    color = "yellow"

    trigger_capture(label, color=color, save_dir="porter_data")
