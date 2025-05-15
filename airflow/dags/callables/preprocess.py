def preprocess_data_callable(
    dataset_root: str = '/home/santitham/airflow/dags/Structural-Defects-Network-MLOps/Dataset',
    artifact_folder: str = '/home/santitham/airflow/dags/Structural-Defects-Network-MLOps/artifact_folder',
    categories: list = ['Decks', 'Walls', 'Pavements'],
    image_size: tuple[int, int] = (256, 256),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    try:
        # Seed everything
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
            logger.info("✅ Transforms initialized.")
            return train_transform, val_test_transform

        def load_combined_dataset(root, cats, transform):
            datasets_list = []
            for cat in cats:
                path = os.path.join(root, cat)
                if not os.path.isdir(path):
                    logger.error(f"❌ Dataset folder not found: {path}")
                    raise FileNotFoundError(f"Dataset folder not found: {path}")
                ds = datasets.ImageFolder(path, transform=transform)
                if len(ds) == 0:
                    logger.warning(f"⚠️ No images found in {path}")
                datasets_list.append(ds)
                logger.info(f"✅ Loaded {cat} with {len(ds)} samples.")
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
            logger.info("⚖️ Balancing training dataset...")
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
            logger.info(f"✅ Balanced to {min_class_size} samples per class.")
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

        ### Run Pipeline ###
        train_transform, val_test_transform = get_transforms()

        full_dataset, class_names = load_combined_dataset(dataset_root, categories, val_test_transform)
        train_set, val_set, test_set = split_dataset(full_dataset)

        # Load again for train-only with augmentation
        train_aug_dataset, _ = load_combined_dataset(dataset_root, categories, train_transform)
        train_aug_set, _, _ = split_dataset(train_aug_dataset)

        balanced_train_set = balance_dataset(train_aug_set)

        save_dataset_as_folder(balanced_train_set, artifact_folder, "train", class_names)
        save_dataset_as_folder(val_set, artifact_folder, "val", class_names)
        save_dataset_as_folder(test_set, artifact_folder, "test", class_names)

        logger.info("✅ Preprocessing pipeline completed.")
        return "Preprocessing completed successfully."

    except Exception as e:
        logger.exception("❌ Preprocessing pipeline failed.")
        raise
