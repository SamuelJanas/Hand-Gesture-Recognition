from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from model.dataset import CustomDataset

def load_data(cfg, return_names = False) -> (DataLoader, DataLoader):

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ]),
    'test': transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    }

    dataset = datasets.ImageFolder(cfg.data.path)
    label_names = dataset.classes # ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']
    train_size = int(cfg.data.split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataset = CustomDataset(train_dataset, transform=data_transforms['train'])
    test_dataset = CustomDataset(test_dataset, transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers = cfg.data.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers = cfg.data.num_workers)

    if return_names:
        return train_loader, test_loader, label_names
    
    return train_loader, test_loader

if __name__ == "__main__":
    cfg = {
        'data': {
            'path': 'data/hagrid-test/',
            'split_ratio': 0.8,
            'image_size': 224,
            'num_workers': 4
        },
        'train': {
            'batch_size': 64,
            'num_epochs': 10,
            'learning_rate': 0.001
        },
        'seed': 42,
        'run_name': 'default'
    }
    cfg = OmegaConf.create(cfg)
    train_loader, test_loader = load_data(cfg)
    