# utils/data_utils.py
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
import numpy as np
import os
import pickle
from PIL import Image
from typing import Tuple, List, Optional


class IP102TxtDataset(Dataset):
    """ip102 基于 txt 文件的数据集"""
    
    def __init__(self, image_dir: str, txt_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        self.classes = set()
        
        print(f"Loading ip102 data from {txt_file}...")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                img_name = parts[0]
                try:
                    label = int(parts[1])
                    self.samples.append((img_name, label))
                    self.classes.add(label)
                except ValueError:
                    print(f"Warning: Cannot parse label in line: {line}")
        
        # 确保标签从0开始连续
        if len(self.classes) > 0:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
            self.samples = [(img_name, self.class_to_idx[label]) 
                          for img_name, label in self.samples if label in self.class_to_idx]
        
        self.num_classes = len(self.classes)
        print(f"  Loaded {len(self.samples)} samples with {self.num_classes} classes")
        
        # 打印类别分布统计
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  Samples per class: min={min(label_counts.values())}, max={max(label_counts.values())}, avg={np.mean(list(label_counts.values())):.1f}")
    
    def _find_image_path(self, img_name: str) -> str:
        """查找图像文件路径"""
        # 直接路径
        direct_path = os.path.join(self.image_dir, img_name)
        if os.path.exists(direct_path):
            return direct_path
        
        # 添加扩展名
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(self.image_dir, img_name + ext)
            if os.path.exists(test_path):
                return test_path
        
        # 搜索子目录
        for root, dirs, files in os.walk(self.image_dir):
            for ext in ['.jpg', '.jpeg', '.png']:
                test_name = img_name if img_name.endswith(ext) else img_name + ext
                if test_name in files:
                    return os.path.join(root, test_name)
        
        # 返回默认路径
        return os.path.join(self.image_dir, img_name)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = self._find_image_path(img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot load image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        # 确保返回张量和整数标签
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img, label_tensor


class IP102FolderDataset(Dataset):
    """ip102 基于文件夹结构的数据集"""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        
        print(f"Loading ip102 data from folder: {root_dir}")
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
        
        class_dirs = sorted([d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d))])
        
        for class_idx, class_name in enumerate(class_dirs):
            self.classes.append(class_name)
            class_path = os.path.join(root_dir, class_name)
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_path, img_name), class_idx))
        
        self.num_classes = len(self.classes)
        print(f"  Loaded {len(self.samples)} samples from {self.num_classes} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot load image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        # 确保返回张量和整数标签
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img, label_tensor


def load_plantvillage(data_path: str, num_clients: int = 10, 
                      batch_size: int = 32, partition_method: str = 'hetero',
                      partition_alpha: float = 0.75) -> Tuple[List[DataLoader], DataLoader, int]:
    print(f"\nLoading PlantVillage dataset from {data_path}")
    print(f"{'='*60}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    if os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    else:
        train_dataset = datasets.ImageFolder(data_path, transform=train_transform)

    if os.path.exists(val_dir):
        test_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    else:
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )
        if hasattr(test_dataset, 'dataset'):
            test_dataset.dataset.transform = test_transform
        print(f"Split training set: {train_size} train, {test_size} val")

    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
        print(f"Classes: {train_dataset.classes}")
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'classes'):
        num_classes = len(train_dataset.dataset.classes)
        print(f"Classes: {train_dataset.dataset.classes}")
    else:
        num_classes = 38

    print(f"\nDataset loaded successfully!")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Image size: 224x224")

    if partition_method == 'iid':
        client_loaders = split_iid(train_dataset, num_clients, batch_size)
    else:
        client_loaders = split_non_iid(train_dataset, num_clients, batch_size, num_classes, partition_alpha)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return client_loaders, test_loader, num_classes


def load_ip102(data_path: str, num_clients: int = 10, 
               batch_size: int = 32, partition_method: str = 'hetero',
               partition_alpha: float = 0.75) -> Tuple[List[DataLoader], DataLoader, int]:
    """加载ip102_v1.1数据集"""
    print(f"\n{'='*60}")
    print(f"Loading ip102_v1.1 dataset")
    print(f"Data path: {data_path}")
    print(f"{'='*60}")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 查找数据文件
    train_txt = None
    val_txt = None
    
    for file in os.listdir(data_path):
        file_lower = file.lower()
        if file_lower == 'train.txt':
            train_txt = os.path.join(data_path, file)
        elif file_lower == 'val.txt':
            val_txt = os.path.join(data_path, file)
    
    # 查找图像目录
    image_dir = None
    for d in ['images', 'JPEGImages', 'Image', 'IMG', '']:
        potential_dir = os.path.join(data_path, d)
        if os.path.exists(potential_dir):
            image_dir = potential_dir
            break
    
    if image_dir is None:
        image_dir = data_path
    
    print(f"Image directory: {image_dir}")
    print(f"Train txt: {train_txt}")
    print(f"Val txt: {val_txt}")
    
    if train_txt and os.path.exists(train_txt):
        train_dataset = IP102TxtDataset(image_dir, train_txt, train_transform)
    else:
        train_folder = os.path.join(data_path, 'train')
        if os.path.exists(train_folder):
            train_dataset = IP102FolderDataset(train_folder, train_transform)
        else:
            raise FileNotFoundError(f"Cannot find training data in {data_path}")
    
    if val_txt and os.path.exists(val_txt):
        test_dataset = IP102TxtDataset(image_dir, val_txt, test_transform)
    else:
        val_folder = os.path.join(data_path, 'val')
        if os.path.exists(val_folder):
            test_dataset = IP102FolderDataset(val_folder, test_transform)
        else:
            # 从训练集中划分验证集
            train_size = int(0.8 * len(train_dataset))
            test_size = len(train_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, test_size]
            )
            print(f"Split training set: {train_size} train, {test_size} val")
    
    if hasattr(train_dataset, 'num_classes'):
        num_classes = train_dataset.num_classes
    elif hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    else:
        num_classes = 102
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Image size: 224x224")
    
    if partition_method == 'iid':
        client_loaders = split_iid(train_dataset, num_clients, batch_size)
    else:
        client_loaders = split_non_iid(train_dataset, num_clients, batch_size, num_classes, partition_alpha)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return client_loaders, test_loader, num_classes


def load_dummy_data(num_samples: int = 5000, num_classes: int = 10, 
                    num_clients: int = 10, batch_size: int = 32,
                    input_size: int = 224) -> Tuple[List[DataLoader], DataLoader, int]:
    """加载虚拟数据用于测试"""
    print(f"Creating dummy dataset with {num_samples} samples, {num_classes} classes")
    
    data = torch.randn(num_samples, 3, input_size, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = torch.utils.data.TensorDataset(data, labels)
    
    client_loaders = []
    samples_per_client = num_samples // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples
        indices = list(range(start_idx, end_idx))
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        client_loaders.append(loader)
        print(f"  Client {i}: {len(subset)} samples")
    
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return client_loaders, test_loader, num_classes


def get_data_loaders(config):
    """统一数据加载接口"""
    dataset_name = config.get('dataset', 'PlantVillage')
    data_dir = config.get('data_dir', 'E:/data/PlantVillage(2)')
    num_clients = config.get('client_num_in_total', 10)
    batch_size = config.get('batch_size', 32)
    partition_method = config.get('partition_method', 'hetero')
    partition_alpha = config.get('partition_alpha', 0.75)
    
    if dataset_name == 'PlantVillage':
        data_path = os.path.join(data_dir, 'PlantVillage')
        return load_plantvillage(data_path, num_clients, batch_size, partition_method, partition_alpha)
    elif dataset_name == 'ip102_v1.1':
        data_path = os.path.join(data_dir, 'ip102_v1.1')
        return load_ip102(data_path, num_clients, batch_size, partition_method, partition_alpha)
    else:
        return load_dummy_data(num_samples=2000, num_classes=10, num_clients=num_clients, batch_size=batch_size)


def split_iid(dataset, num_clients, batch_size):
    """IID划分"""
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    client_sizes = [total_size // num_clients] * num_clients
    client_sizes[-1] += total_size - sum(client_sizes)
    
    client_loaders = []
    start_idx = 0
    for client_id in range(num_clients):
        client_indices = indices[start_idx:start_idx + client_sizes[client_id]]
        client_subset = Subset(dataset, client_indices)
        # 使用自定义 collate_fn 确保类型一致
        loader = DataLoader(
            client_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=False,
            collate_fn=custom_collate_fn
        )
        client_loaders.append(loader)
        start_idx += client_sizes[client_id]
        print(f"  Client {client_id}: {len(client_subset)} samples")
    
    return client_loaders


def split_non_iid(dataset, num_clients, batch_size, num_classes, alpha=0.75):
    """非IID划分（Dirichlet分布）"""
    # 获取标签
    if hasattr(dataset, 'samples'):
        labels = [label for _, label in dataset.samples]
    else:
        labels = [dataset[i][1].item() if torch.is_tensor(dataset[i][1]) else dataset[i][1] 
                  for i in range(len(dataset))]
    
    labels = np.array(labels)
    
    # Dirichlet分布
    client_class_probs = np.random.dirichlet([alpha] * num_classes, size=num_clients)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) == 0:
            continue
        
        np.random.shuffle(class_indices)
        
        class_counts = (client_class_probs[:, class_id] * len(class_indices)).astype(int)
        class_counts[-1] = len(class_indices) - sum(class_counts[:-1])
        class_counts = np.maximum(class_counts, 0)
        
        start = 0
        for client_id, count in enumerate(class_counts):
            if count > 0 and start + count <= len(class_indices):
                client_indices[client_id].extend(class_indices[start:start+count].tolist())
                start += count
    
    # 确保每个客户端有数据
    for client_id in range(num_clients):
        if len(client_indices[client_id]) == 0:
            max_client = max(range(num_clients), key=lambda i: len(client_indices[i]))
            if len(client_indices[max_client]) > 0:
                sample_idx = client_indices[max_client].pop()
                client_indices[client_id].append(sample_idx)
    
    # 创建加载器
    client_loaders = []
    for client_id in range(num_clients):
        if len(client_indices[client_id]) > 0:
            client_subset = Subset(dataset, client_indices[client_id])
            loader = DataLoader(
                client_subset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0, 
                pin_memory=False,
                collate_fn=custom_collate_fn
            )
            client_loaders.append(loader)
            print(f"  Client {client_id}: {len(client_subset)} samples")
        else:
            print(f"  Warning: Client {client_id} has no samples!")
    
    return client_loaders


def custom_collate_fn(batch):
    """
    自定义 collate 函数，确保批次数据类型一致
    """
    images = []
    labels = []
    
    for img, label in batch:
        images.append(img)
        labels.append(torch.tensor(label, dtype=torch.long))

    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    
    return images, labels


def describe_partition(loaders):
    """Record actual source indices and sample paths, including unassigned samples."""
    dataset = loaders[0].dataset.dataset
    clients = [list(map(int, loader.dataset.indices)) for loader in loaders]
    assigned = [index for indices in clients for index in indices]
    samples = []
    dataset_root = getattr(dataset, 'root', getattr(dataset, 'image_dir', None))
    for path, label in getattr(dataset, 'samples', []):
        if dataset_root and os.path.isabs(path):
            path = os.path.relpath(path, dataset_root)
        samples.append([path.replace('\\', '/'), int(label)])
    return {
        'protocol': 'source allocation (see docs/PROVENANCE.md)',
        'total_training_samples': len(dataset),
        'assigned_samples': len(assigned),
        'unassigned_indices': sorted(set(range(len(dataset))) - set(assigned)),
        'duplicate_assignments': len(assigned) - len(set(assigned)),
        'client_indices': clients,
        'samples': samples,
        'class_to_idx': getattr(dataset, 'class_to_idx', None),
    }
