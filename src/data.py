from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class TransposeToTensor(object):
    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        new_img = torch.Tensor(img.copy())
        return new_img


class LegoDataset(Dataset):

    def __init__(self, split, transform=None, root_dir='lego_data/base_images', seed=42):
        """
        Args:
         - split (str): train, val, or test split
        """
        self.root_dir = root_dir
        self.image_folder = datasets.ImageFolder(root=self.root_dir) # imgs is the attribute where everything lives
        self.class_names = self.image_folder.classes

        # Train/Val/Test - 70/10/20
        X, y = map(list, zip(*self.image_folder.imgs))

        if split is None:
            self.imgs, self.labels = (X, y)
        else:
            _X, X_test, _y, y_test = train_test_split(X, y, test_size=.2, random_state=seed)
            X_train, X_val, y_train, y_val = train_test_split(_X, _y, test_size=.125, random_state=seed)

            splits = {'train': (X_train, y_train),
                        'val':(X_val, y_val),
                        'test': (X_test, y_test)}

            if split not in splits:
                raise ValueError('Invalid split name')
            self.imgs, self.labels = splits[split]

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imgs[idx]
        image = plt.imread(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def get_inference_transforms(model: str):
    if model == 'resnet50':
        pretrained_transforms = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    elif model == 'mobilenetv3large':
        pretrained_transforms = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
    else:
        raise ValueError(f'Model name "{model}" not recognized')

    inference_transforms = transforms.Compose([
        TransposeToTensor(),
        pretrained_transforms,
    ])
    return inference_transforms