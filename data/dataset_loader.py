import random
import numpy as np
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.image_folder import *
import sys


class InteriorNetRyDataLoader(BaseDataLoader):
    def __init__(self, opt, list_path, is_train, _batch_size, num_threads):
        dataset = InteriorNetRyFolder(opt=opt, 
            list_path =list_path, is_train=is_train)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=is_train, num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'InteriorNetRyDataLoader'
    
    def __len__(self):
        return len(self.dataset)

class ScanNetDataLoader(BaseDataLoader):
    def __init__(self, opt, list_path, is_train, _batch_size, num_threads):
        dataset = ScanNetFolder(opt=opt, 
            list_path =list_path, is_train=is_train)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=is_train, num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'ScanNetFolder'
    
    def __len__(self):
        return len(self.dataset)


class SUN360DataLoader(BaseDataLoader):
    def __init__(self, opt, list_path, is_train, _batch_size, num_threads):
        dataset = SUN360Folder(opt=opt, 
            list_path =list_path, is_train=is_train)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=is_train, num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'SUN360DataLoader'
    
    def __len__(self):
        return len(self.dataset)

