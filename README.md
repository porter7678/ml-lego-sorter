# Real-time Computer Vision LEGO Sorter


![Lego Sorter in Action](media/ml_lego_sorter.gif)

This project is a computer vision-powered LEGO sorting machine. Using a Raspberry Pi and a fine-tuned MobileNet Model, LEGOs area automatically sorted by shape as they move along a conveyor belt. The system runs model inference on the Raspberry Pi via Docker, overcoming real-time performance challenges.

## Objective
Enough of ML models that only live in notebooks! I wanted to deploy an ML model into the real world using Docker for real-time inferencing. The focus of this project would be getting something deployed that works, rather than on fancy models and perfect accuracy.

# Implementation
## Mechanical Construction
After spening more time, money, and duct tape than I would care to admit trying to get the conveyor belt functioning and fussing around with gearbox ratios, I realized that I probably should have just 3D printed most of the components. Regardless, the first step of the project was building the following mechanical components:
 - A conveyor belt made of PVC pipes, spare cloth, and duct tape
 - A gear box made of a DC motor, a 5V battery, LEGO Technic gears, and a box of protein shakes (for added weight).
 - A sorting arm made with a micro servo and a ruler, mounted using cardboard, duct tape, and rubber bands
 - A Raspberry Pi attached to a camera, and controlled over SSH.
   
<div style="display: flex; justify-content: space-between;">
  <img src="media/lego_machine.png" alt="LEGO sorting machine" height="300"/>
  <img src="media/lego_gearbox.jpg" alt="Gearbox" height="300"/>
</div>

## Data
I collected the data for this project from scratch. 

Originally, I attempted using a LEGO dataset I found online supplemented with a small amount of my own data, but this proved unsuccessful.

<img src="media/train_test_split.png" alt="Train test split" height="300"/>
   
## Model
For the computer vision model, I used a fine-tuned MobileNet in PyTorch. 

Initally, I used a ResNet-50 model, which had better accuracy. However, the ResNet took several seconds to complete inference on a single image, by which time the LEGO would have already passed the sorting arm. Therefore, I compared the speed and accuracy of several different models (MobileNet, EfficientNet, ViT, VGG), and ultimately decided on MobileNetV3-Large, which could process 3 frames per second. There were other models that were faster, but at the cost of lower accuracy.



## Training


## Deployment

## Challenges
 - Accuracy vs Latency.

## Results

## Future Work

# Setup Instructions


This project was inspired by a [blog post](https://medium.com/@pacogarcia3/tensorflow-on-raspbery-pi-lego-sorter-ab60019dcf32) by Paco Garcia.
