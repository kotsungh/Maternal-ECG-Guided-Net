import os
import glob
import torch
from util import util
    
class ADFECGDB_Dataset_Vector(torch.utils.data.Dataset):
    
    def __init__(self, parent_dir, is_train=True):
        self.filepath = glob.glob(os.path.join(parent_dir, "*.mat"))
        self.is_train = is_train
        
    def __getitem__(self, index):
        # Create MATLAB File Reader
        reader = util.MatReader(self.filepath[index])
        
        abecg = reader.read_field('ab_ecg') 
        mecg = reader.read_field('m_ecg') 
        fecg = reader.read_field('f_ecg') 
        
        fqrs = reader.read_field('f_refs')
        
        if self.is_train:
            return abecg, mecg, fecg
        else:
            return abecg, mecg, fecg, fqrs
    
    def __len__(self):
        return len(self.filepath)
