import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.meg_net import MaternalEcgGuidedNet
import torch.nn as nn

import argparse
from typing import Optional

import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from engine_train import Trainer
from dataset import ADFECGDB_Dataset

                
def prepare_dataloader(dataset: Dataset, batch_size: int, is_train: Optional[bool] = True):
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    
    if is_train:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=global_rank, shuffle=True
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=global_rank, shuffle=False
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )


def main(args):
    
    init_process_group(backend="gloo")
    
    cudnn.benchmark = True
    
    ##### Initialize dataloader
    train_root = r".\train"
    eval_root = r".\val"
    
    train_dataloader = prepare_dataloader(
        ADFECGDB_Dataset(train_root) , 
        args.batch_size, 
        is_train=True
    )
    
    eval_dataloader = prepare_dataloader(
        ADFECGDB_Dataset(eval_root), 
        args.batch_size, 
        is_train=False
    )
    
    ##### Initialize model
    model = MaternalEcgGuidedNet(
        n_depths=args.depth, 
        signal_dim=args.signal_dim, 
        abecg_kernel_size=args.abecg_kernel_size, 
        mecg_kernel_size=args.mecg_kernel_size, 
        base_channels=args.base_channels
    )
    
    ##### Initialize training
    model_name = f"megnet-depth{args.depth}-signal_dim{args.signal_dim}-abecg_kernel{args.abecg_kernel_size}-mecg_kernel{args.mecg_kernel_size}-base_channels{args.base_channels}"    
    loss_fn = nn.L1Loss()
    writer = SummaryWriter(rf".\runs\{model_name}")
    checkpoint_path = rf".\params\{model_name}.pth"
    
    trainer = Trainer(
        train_dataloader=train_dataloader, 
        eval_dataloader=eval_dataloader,
        model=model, 
        model_name=model_name,
        loss_fn=loss_fn, 
        writer=writer,
        checkpoint_path=checkpoint_path
    )    
    trainer.train(args.num_epochs)
    
    destroy_process_group()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--depth', default=6, type=int)
    parser.add_argument('--signal_dim', default=1024, type=int)
    parser.add_argument('--abecg_kernel_size', default=7, type=int)
    parser.add_argument('--mecg_kernel_size', default=5, type=int)
    parser.add_argument('--base_channels', default=16, type=int)
    args = parser.parse_args()
    main(args)
