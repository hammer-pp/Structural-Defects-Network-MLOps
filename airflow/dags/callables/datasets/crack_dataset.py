import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CrackDataset(Dataset):
    """
    A custom PyTorch Dataset for loading crack detection images and their labels from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing image filenames and their corresponding labels.
        img_dir (str): Directory where the images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        labels_df (pd.DataFrame): DataFrame containing image filenames and labels.
        img_dir (str): Directory containing the images.
        transform (callable): Transformations to apply to the images.
        label_map (dict): Mapping from label names to integer values.

    Methods:
        __len__(): Returns the total number of samples.
        __getitem__(idx): Retrieves the image and label at the specified index.

    Returns:
        tuple: (image, label) where image is the transformed image and label is the corresponding integer label.
    """
    def __init__(self, csv_path, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'Non-cracked': 0, 'Cracked': 1}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        label = self.label_map[self.labels_df.iloc[idx, 1]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_dataloaders(batch_size=32):
    """
    Creates and returns PyTorch DataLoader objects for training, validation, and test datasets of crack images.

    Args:
        batch_size (int, optional): Number of samples per batch to load. Defaults to 32.

    Returns:
        tuple: A tuple containing three DataLoader objects for the training, validation, and test datasets, respectively.

    Notes:
        - Assumes the existence of a CrackDataset class that takes a CSV file of labels, an image directory, and a transform.
        - Images are resized to 224x224, converted to tensors, and normalized using ImageNet statistics.
        - The root directory for datasets is '../artifact_folder', with subdirectories for train, val, and test splits.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    root = '../artifact_folder'
    train_set = CrackDataset(f'{root}/train/labels.csv', f'{root}/train/images', transform)
    val_set = CrackDataset(f'{root}/val/labels.csv', f'{root}/val/images', transform)
    test_set = CrackDataset(f'{root}/test/labels.csv', f'{root}/test/images', transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )
