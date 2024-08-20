import torch
import torch.nn as nn
from torchvision import models, datasets
import matplotlib.pyplot as plt

from data import LegoDataset, get_inference_transforms

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
    model.load_state_dict(torch.load(model_path))

    model.eval()
    return model


def predict_multiple(image_dir, model_path, split='test'):
    image_folder = datasets.ImageFolder(image_dir)
    num_classes = len(image_folder.classes)
    print('num_classes', num_classes)

    model = load_resnet50(model_path, num_classes)

    for image_path, label in image_folder.imgs:
        image = plt.imread(image_path)
        pred = predict(image, model)
        print(image_path, pred)


if __name__ == '__main__':
    model_path = 'checkpoints/resnet50_kaggle_only.pt'
    image_dir = 'lego_data/base_images'
    predict_multiple(image_dir, model_path)