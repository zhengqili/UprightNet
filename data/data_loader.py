from data.dataset_loader import *


def CreateInteriorNetryDataLoader(opt, list_path, 
								is_train, _batch_size, 
								num_threads):
    data_loader = InteriorNetRyDataLoader(opt, list_path, 
    									  is_train, _batch_size, 
    									  num_threads) 
    return data_loader

def CreateSUN360DataLoader(opt, list_path, 
								is_train, _batch_size, 
								num_threads):
    data_loader = SUN360DataLoader(opt, list_path, 
    							   is_train, _batch_size, 
    							   num_threads) 
    return data_loader


def CreateScanNetDataLoader(opt, list_path, 
								is_train, _batch_size, 
								num_threads):
    data_loader = ScanNetDataLoader(opt, list_path, 
    								is_train, _batch_size, 
    								num_threads) 
    return data_loader

    