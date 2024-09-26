"""
Results (size 224)
 - MobileNet V2: 3 fps
 - MobileNet V2 Quantized: 5 fps
 - MobileNet V3 Small: 9 fps
 - *MobileNet V3 Large: 3 fps
 - *EfficientNet V2 Small: 1.1 fps
 - EfficientNet V2 Large: 0.3 fps
 - *ResNet 50: 1.1 fps
 - VIT B 16: 0.4 fps
 - *VIT B 32: 1.3 fps

Results (size 128)
  - *MobileNet V3 Large: 8 fps
  - *EfficientNet V2 Small: 2 fps
  - EfficientNet V2 Large: 0.6 fps
  - *ResNet 50: 2.5 fps
  - ResNet 101: 1.6 fps
  - ResNet 152: 1.2 fps
  - ResNext 50: 1.9 fps
  - VGG 16 BN: 0.7 fps
  - VGG 11 BN: 1 fps

"""

import os
import time

import cv2 as cv
import torch
from PIL import Image
from torchvision import datasets, models, transforms

os.environ["QT_QPA_PLATFORM"] = "xcb"

# torch.backends.quantized.engine = "qnnpack"

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv.CAP_PROP_FPS, 36)

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# net = models.mobilenet_v3_large(weights="DEFAULT")
# net = models.efficientnet_v2_s(weights="DEFAULT")
net = models.resnet50(weights="DEFAULT")
# net = models.vit_b_32(weights="DEFAULT")
# net = models.vgg11_bn(weights="DEFAULT")

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")

        permuted = image[:, :, [2, 1, 0]]

        input_tensor = preprocess(permuted)
        input_batch = input_tensor.unsqueeze(0)

        output = net(input_batch)

        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0

        # top = list(enumerate(output[0].softmax(dim=0)))
        # top.sort(key=lambda x: x[1], reverse=True)
        # for idx, val in top[:10]:
        #     print(f"{val.item()*100:.2f}% idx: {idx}")
