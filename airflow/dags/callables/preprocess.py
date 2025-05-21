import os
import numpy as np
import pandas as pd
from PIL import Image
import requests
from collections import Counter
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, Subset, random_split
from tqdm import tqdm
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess_data_callable(
    dataset_root: str = '/opt/airflow/dataset',
    artifact_folder: str = '/opt/airflow/artifact_folder',
    categories: list = ['Decks', 'Walls', 'Pavements'],
    users_csv_path: str = 'cloudinary_dataset_users.csv',
    image_size: tuple[int, int] = (256, 256),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
        Preprocesses image datasets for structural defect detection, including downloading user-contributed images, applying transformations, balancing classes, splitting into train/val/test sets, and saving the processed data.

        Args:
            dataset_root (str): Root directory containing the dataset folders for each category.
            artifact_folder (str): Directory where the processed datasets will be saved.
            categories (list): List of category folder names to include (e.g., ['Decks', 'Walls', 'Pavements']).
            users_csv_path (str): Path to the CSV file containing user-contributed image URLs and metadata.
            image_size (tuple[int, int]): Target size (width, height) for resizing images.
            train_ratio (float): Proportion of the dataset to use for training.
            val_ratio (float): Proportion of the dataset to use for validation.
            seed (int): Random seed for reproducibility.

        Returns:
            str: Success message upon completion.

        Raises:
            RuntimeError: If no datasets are loaded.
            Exception: For any other errors during preprocessing.

        Pipeline Steps:
            1. Downloads user-contributed images from URLs in the provided CSV.
            2. Defines and applies image transformations for training and validation/testing.
            3. Loads datasets from specified categories and combines them.
            4. Splits the combined dataset into training, validation, and test sets.
            5. Applies data augmentation and class balancing to the training set.
            6. Saves the processed datasets (images and labels) into the artifact folder.
    """
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)

        ### Inner Transform and Helper Functions ###
        class Denoise:
            def __call__(self, img):
                return TF.gaussian_blur(img, kernel_size=3)

        def get_transforms():
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                Denoise(),
                transforms.ToTensor(),
            ])
            val_test_transform = transforms.Compose([
                transforms.Resize(image_size),
                Denoise(),
                transforms.ToTensor(),
            ])
            return train_transform, val_test_transform

        def download_users_data(csv_path, base_dir):
            if not os.path.exists(csv_path):
                logger.warning("❌ No users CSV found. Skipping cloud user data.")
                return
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                label = row['label']
                place = row['place']
                url = row['url']

                save_dir = os.path.join(base_dir, place, label)
                os.makedirs(save_dir, exist_ok=True)
                img_name = os.path.basename(url).split("?")[0]
                save_path = os.path.join(save_dir, img_name)

                try:
                    if not os.path.exists(save_path):
                        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
                        img.save(save_path)
                        logger.info(f"{url} image is saved")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download {url}: {e}")

        def load_combined_dataset(root, cats, transform):
            datasets_list = []
            for cat in cats:
                path = os.path.join(root, cat)
                if not os.path.isdir(path):
                    logger.warning(f"❌ Dataset folder not found: {path}")
                    continue
                ds = datasets.ImageFolder(path, transform=transform)
                if len(ds) == 0:
                    logger.warning(f"⚠️ No images found in {path}")
                datasets_list.append(ds)
            if not datasets_list:
                raise RuntimeError("❌ No datasets loaded.")
            return ConcatDataset(datasets_list), datasets_list[0].classes

        def split_dataset(dataset):
            total_len = len(dataset)
            train_size = int(train_ratio * total_len)
            val_size = int(val_ratio * total_len)
            test_size = total_len - train_size - val_size
            return random_split(dataset, [train_size, val_size, test_size])

        def balance_dataset(dataset):
            labels = [label for _, label in dataset]
            label_counts = Counter(labels)
            if len(label_counts) < 2:
                logger.warning("⚠️ Not enough classes to balance.")
                return dataset
            min_class_size = min(label_counts.values())
            indices_by_class = {
                cls: [i for i, (_, label) in enumerate(dataset) if label == cls]
                for cls in label_counts
            }
            balanced_indices = []
            for cls, indices in indices_by_class.items():
                sampled = np.random.choice(indices, min_class_size, replace=False)
                balanced_indices.extend(sampled)
            np.random.shuffle(balanced_indices)
            return Subset(dataset, balanced_indices)

        def save_dataset_as_folder(dataset, save_path, split_name, class_names):
            image_folder = os.path.join(save_path, split_name, "images")
            label_csv = os.path.join(save_path, split_name, "labels.csv")
            os.makedirs(image_folder, exist_ok=True)

            data = []
            for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {split_name}"):
                filename = f"{split_name}_{idx:05d}.png"
                filepath = os.path.join(image_folder, filename)
                save_image(image, filepath)
                data.append({"filename": filename, "label": class_names[label]})
            pd.DataFrame(data).to_csv(label_csv, index=False)
            logger.info(f"📁 {split_name} saved to {image_folder} with labels in {label_csv}")

        ### PIPELINE EXECUTION ###
        logger.info("🚀 Starting preprocessing...")

        # Step 1: Download Cloudinary user data
        download_users_data(users_csv_path, dataset_root)

        # Step 2: Transforms
        train_transform, val_test_transform = get_transforms()

        # Step 3: Load + Split Dataset
        all_categories = categories + ['Users']
        full_dataset, class_names = load_combined_dataset(dataset_root, all_categories, val_test_transform)
        train_set, val_set, test_set = split_dataset(full_dataset)

        # Step 4: Train-only loading with augmentations
        train_aug_dataset, _ = load_combined_dataset(dataset_root, all_categories, train_transform)
        train_aug_set, _, _ = split_dataset(train_aug_dataset)

        balanced_train_set = balance_dataset(train_aug_set)

        # Step 5: Save processed dataset
        save_dataset_as_folder(balanced_train_set, artifact_folder, "train", class_names)
        save_dataset_as_folder(val_set, artifact_folder, "val", class_names)
        save_dataset_as_folder(test_set, artifact_folder, "test", class_names)

        logger.info("✅ Preprocessing pipeline completed.")
        return "Preprocessing completed successfully."

    except Exception as e:
        logger.exception("❌ Preprocessing failed.")
        raise
