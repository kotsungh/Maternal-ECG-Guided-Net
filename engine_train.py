import torch
from torch.nn.parallel import DistributedDataParallel 
from torch.distributed import barrier
from torch.utils.tensorboard import SummaryWriter

import os
import time
import numpy as np
from typing import Iterable, Optional



class Trainer:
    def __init__(
        self,
        train_dataloader: Iterable,
        eval_dataloader: Iterable,
        
        loss_fn: torch.nn.modules.loss,
        writer: SummaryWriter,
        
        model: torch.nn.Module,
        model_name: str,
        checkpoint_path: str,
        
        optimizer_learning_rate: Optional[float] = 1e-3,
        optimizer_weight_decay: Optional[float] = 1e-4,
        scheduler_gamma: Optional[float] = 0.5,        
    ):  
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.loss_fn = loss_fn
        self.writer = writer
        
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_gamma = scheduler_gamma
        
        self.start_epoch = 0 # initialize epoch to 0
        
        self.best_eval_loss = np.inf
        
        self.model = model
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        
        if os.path.exists(self.checkpoint_path):
            print("[Loading checkpoint]")
            self._load_checkpoint(self.checkpoint_path)  
        else:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.local_rank)
            self.ddp_model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
            
            self.optimizer = torch.optim.Adam(
                self.ddp_model.parameters(),
                lr=self.optimizer_learning_rate,
                weight_decay=self.optimizer_weight_decay
            )
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                factor=self.scheduler_gamma,
                patience=10
            )           
            
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["MODEL_STATE_DICT"])
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.local_rank)
        self.ddp_model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
        
        self.optimizer = torch.optim.Adam(
            self.ddp_model.parameters(),
            lr=self.optimizer_learning_rate,
            weight_decay=self.optimizer_weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=self.scheduler_gamma,
            patience=10
        )      
        
        self.optimizer.load_state_dict(checkpoint["OPTIMIZER_STATE_DICT"])
        self.scheduler.load_state_dict(checkpoint["SCHEDULER_STATE_DICT"])
        self.loss_fn= checkpoint["LOSS"]
        self.start_epoch = checkpoint["START_EPOCH"]
        self.best_eval_loss = checkpoint["BEST_EVAL_LOSS"]
        
        print(f"[Resuming training from checkpoint at Epoch {self.start_epoch + 1}]")
        
    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            "MODEL_STATE_DICT": self.ddp_model.module.state_dict(),
            "OPTIMIZER_STATE_DICT": self.optimizer.state_dict(),
            "SCHEDULER_STATE_DICT": self.scheduler.state_dict(),
            'LOSS': self.loss_fn,
            "START_EPOCH": epoch,
            "BEST_EVAL_LOSS": self.best_eval_loss
        }      
        torch.save(checkpoint, self.checkpoint_path)
        
    def _log_tensorboard(self, train_metric: float, eval_metric: float, epoch: int):
        self.writer.add_scalars(
            self.loss_fn.__class__.__name__, 
            {'Train': train_metric, 'Evaluation': eval_metric}, 
            global_step=epoch + 1
        )
        
    def train(self, max_epochs: int):
        
        start_time = time.time()
        stop_training_count = 0
        
        for epoch in range(self.start_epoch, max_epochs):
            if stop_training_count >= 25:
                break
            
            since = time.time()
            
            self.ddp_model.train()
            self.train_dataloader.sampler.set_epoch(epoch) 
            
            #--------------Training
            train_running_loss = 0.0
            
            for abecg, mecg, fecg in self.train_dataloader:
                abecg = abecg.to(self.local_rank, non_blocking=True)
                mecg = mecg.to(self.local_rank, non_blocking=True)
                fecg = fecg.to(self.local_rank, non_blocking=True)
                
                self.optimizer.zero_grad()
                pred_fecg = self.ddp_model(abecg, mecg)
                loss = self.loss_fn(pred_fecg, fecg)
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.synchronize()
                barrier()
                
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                train_running_loss += loss.item() * len(abecg)     
                
                barrier()
            
            barrier()
            
            #--------------Evaluation
            eval_running_loss = 0.0
            self.ddp_model.eval()
            
            with torch.no_grad():
                for abecg, mecg, fecg in self.eval_dataloader:
                    abecg = abecg.to(self.local_rank, non_blocking=True)
                    mecg = mecg.to(self.local_rank, non_blocking=True)
                    fecg = fecg.to(self.local_rank, non_blocking=True)
                    
                    pred_fecg = self.ddp_model(abecg, mecg)
                    loss = self.loss_fn(pred_fecg, fecg)
                    
                    torch.cuda.synchronize()
                    barrier()
                    
                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                    eval_running_loss += loss.item() * len(abecg)
                    
                    barrier()
                    
            barrier() 
            
            train_loss = train_running_loss / len(self.train_dataloader.dataset)
            eval_loss = eval_running_loss / len(self.eval_dataloader.dataset)
            
            self.scheduler.step(eval_loss)
            
            self._log_tensorboard(train_loss, eval_loss, epoch)
            
            if self.global_rank == 0:
                print(f'[Epoch {epoch + 1}]')
                print(f'[Train loss: {train_loss:.4f}]')
                print(f'[Evaluation loss: {eval_loss:.4f}]')
                
                if eval_loss <= self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(epoch)
                    stop_training_count = 0
                else:
                    stop_training_count += 1
                    
                print(f'[Best evaluation loss: {self.best_eval_loss:.4f}]')
                
                time_elapsed = time.time() - since
                print('[Training one epoch complete in {:.0f}m {:.0f}s]'.format(time_elapsed // 60, time_elapsed % 60))
                print()
 
        if self.global_rank == 0:
            total_time = time.time() - start_time
            print('[Training complete in {:.0f}m {:.0f}s]'.format(total_time // 60, total_time % 60))
            print()
            
            
            