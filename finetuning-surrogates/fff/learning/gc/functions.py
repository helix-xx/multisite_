"""Functions that use the model through interfaces designed for workflow engines"""
import os
import time
from collections import defaultdict
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import ase
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataListLoader, NeighborLoader

# from torch.utils.data import DataLoader

from fff.learning.gc.data import AtomsDataset
from fff.learning.gc.models import SchNet
from fff.learning.base import BaseLearnableForcefield, ModelMsgType
from fff.learning.util.messages import TorchMessage

from torch_geometric.nn import data_parallel
from torch_geometric.data import Batch
## torch DDP, not completed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import datetime
import json

import logging
logger = logging.getLogger(__name__)

import socket
import random
import subprocess

def generate_random_port():
    return random.randrange(1024, 65536)

def check_port(port):
    command = "netstat -tuln | grep {}".format(port)  # Linux 或 macOS
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    return len(output) > 0

def get_available_port():
    while True:
        port = generate_random_port()
        # print(f"check port {port} available or note")
        if not check_port(port):
            return port

def eval_batch_DP(model: SchNet, batch: Data) -> (torch.Tensor, torch.Tensor):
    """Get the energies and forces for a certain batch of molecules

    Args:
        model: Model to evaluate
        batch: Batch of data to evaluate
    Returns:
        Energy and forces for the batch
    """
    # batch.pos.requires_grad = True
    force_batch = []
    data = [i.pos for i in batch]
    for _ in data:
        _.to('cuda')
        _.requires_grad = True
    energ_batch = model(batch)
    for _ in data:
        force = -torch.autograd.grad(energ_batch, _, grad_outputs=torch.ones_like(energ_batch), retain_graph=True)[0]
        force_batch.append(force)
    # force_batch = -torch.autograd.grad(energ_batch, batch.pos, grad_outputs=torch.ones_like(energ_batch), retain_graph=True)[0]
    return energ_batch, torch.cat(force_batch, dim=0).to('cuda')

def eval_batch(model: SchNet, batch: Data) -> (torch.Tensor, torch.Tensor):
    """Get the energies and forces for a certain batch of molecules

    Args:
        model: Model to evaluate
        batch: Batch of data to evaluate
    Returns:
        Energy and forces for the batch
    """
    batch.pos.requires_grad = True
    
    energ_batch = model(batch)
    force_batch = -torch.autograd.grad(energ_batch, batch.pos, grad_outputs=torch.ones_like(energ_batch), retain_graph=True)[0]
    return energ_batch, force_batch


class GCSchNetForcefield(BaseLearnableForcefield):
    """Standardized interface to Graphcore's implementation of SchNet"""

    def evaluate(self,
                 model_msg: ModelMsgType,
                 atoms: list[ase.Atoms],
                 batch_size: int = 256,
                 device: str = 'cpu',
                 cpu=1,
                 gpu=0) -> tuple[list[float], list[np.ndarray]]:
        model = self.get_model(model_msg)

        # Place the model on the GPU in eval model
        model.eval()
        model.to(device)

        with TemporaryDirectory() as tmp:
            # Make the data loader
            with open(os.devnull, 'w') as fp, redirect_stderr(fp):
                dataset = AtomsDataset.from_atoms(atoms, root=tmp)
                loader = DataListLoader(dataset, batch_size=batch_size)

            # Run all entries
            energies = []
            forces = []
            for batch in loader:
                # Move the data to the array
                batch.to(device)

                # Get the energies then compute forces with autograd
                energ_batch, force_batch = eval_batch(model, batch)

                # Split the forces
                n_atoms = batch.n_atoms.cpu().detach().numpy()
                forces_np = force_batch.cpu().detach().cpu().numpy()
                forces_per = np.split(forces_np, np.cumsum(n_atoms)[:-1])

                # Add them to the output lists
                energies.extend(energ_batch.detach().cpu().numpy().tolist())
                forces.extend(forces_per)

        model.to('cpu')  # Move it off the GPU memory

        return energies, forces

    # def train(self,
    def train_basic(self,
              model_msg: ModelMsgType,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 128,
              learning_rate: float = 1e-3,
              huber_deltas: (float, float) = (0.5, 1),
              energy_weight: float = 0.1,
              reset_weights: bool = False,
              patience: int = None,
              cpu=1,
              gpu=1) -> (TorchMessage, pd.DataFrame):

        model = self.get_model(model_msg)
        model.to(device)
        
        # Unpack some inputs
        huber_eng, huber_force = huber_deltas

        # If desired, re-initialize weights
        if reset_weights:
            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        # Start the training process
        with TemporaryDirectory(prefix='spk') as td:
            td = Path(td)
            # Save the batch to an ASE Atoms database
            with open(os.devnull, 'w') as fp, redirect_stderr(fp):
                train_dataset = AtomsDataset.from_atoms(train_data, td / 'train')
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                valid_dataset = AtomsDataset.from_atoms(valid_data, td / 'valid')
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            # Make the trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            if patience is None:
                patience = num_epochs // 8
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.8, min_lr=1e-6)

            # Store the best loss
            best_loss = torch.inf

            # Loop over epochs
            log = []
            model.train()
            start_time = time.perf_counter()
            for epoch in range(num_epochs):
                # Iterate over all batches in the training set
                train_losses = defaultdict(list)
                for batch in train_loader:
                    # transimit_time = time.time()
                    batch.to(device)
                    # print(f"transimit_time:{time.time()-transimit_time}")

                    optimizer.zero_grad()

                    # Compute the energy and forces
                    energy, force = eval_batch(model, batch)

                    # Get the forces in energy and forces
                    energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss

                    # Iterate backwards
                    total_loss.backward()
                    optimizer.step()

                    # Add the losses to a log
                    with torch.no_grad():
                        train_losses['train_loss_force'].append(force_loss.item())
                        train_losses['train_loss_energy'].append(energy_loss.item())
                        train_losses['train_loss_total'].append(total_loss.item())

                # Compute the average loss for the batch
                train_losses = dict((k, np.mean(v)) for k, v in train_losses.items())

                # Get the validation loss
                valid_losses = defaultdict(list)
                for batch in valid_loader:
                    batch.to(device)
                    energy, force = eval_batch(model, batch)

                    # Get the loss of this batch
                    energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss

                    with torch.no_grad():
                        valid_losses['valid_loss_force'].append(force_loss.item())
                        valid_losses['valid_loss_energy'].append(energy_loss.item())
                        valid_losses['valid_loss_total'].append(total_loss.item())

                valid_losses = dict((k, np.mean(v)) for k, v in valid_losses.items())

                # Reduce the learning rate
                scheduler.step(valid_losses['valid_loss_total'])

                # Save the best model if possible
                if valid_losses['valid_loss_total'] < best_loss:
                    best_loss = valid_losses['valid_loss_total']
                    torch.save(model, td / 'best_model')

                # Store the log line
                print(f"epoch:{epoch}, time:{time.perf_counter() - start_time}, train_loss:{train_losses}, valid_loss:{valid_losses}")
                log.append({'epoch': epoch, 'time': time.perf_counter() - start_time, **train_losses, **valid_losses})

            # Load the best model back in
            best_model = torch.load(td / 'best_model', map_location='cpu')

            return TorchMessage(best_model), pd.DataFrame(log)
        
        
    def train_DP(self,
              model_msg: ModelMsgType,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 128,
              learning_rate: float = 1e-3,
              huber_deltas: (float, float) = (0.5, 1),
              energy_weight: float = 0.1,
              reset_weights: bool = False,
              patience: int = None,
              cpu=1,
              gpu=1) -> (TorchMessage, pd.DataFrame):

        model = self.get_model(model_msg)
        model.to(device)
        ## lets use multiple GPUs
        if torch.cuda.device_count()>1:
            model = data_parallel.DataParallel(model)

        
        # Unpack some inputs
        huber_eng, huber_force = huber_deltas

        # If desired, re-initialize weights
        if reset_weights:
            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        # Start the training process
        with TemporaryDirectory(prefix='spk') as td:
            td = Path(td)
            # Save the batch to an ASE Atoms database
            with open(os.devnull, 'w') as fp, redirect_stderr(fp):
                train_dataset = AtomsDataset.from_atoms(train_data, td / 'train')
                # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True)

                valid_dataset = AtomsDataset.from_atoms(valid_data, td / 'valid')
                # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, shuffle=False)


            # Make the trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            if patience is None:
                patience = num_epochs // 8
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.8, min_lr=1e-6)

            # Store the best loss
            best_loss = torch.inf

            # Loop over epochs
            log = []
            model.train()
            start_time = time.perf_counter()
            for epoch in range(num_epochs):
                # Iterate over all batches in the training set
                train_losses = defaultdict(list)
                for batch in train_loader:
                    # batch.to(device)

                    optimizer.zero_grad()


                    # get list info (dataparallel)
                    b_n_atoms = torch.Tensor([i.n_atoms for i in batch]).to(device)
                    b_y = torch.Tensor([i.y for i in batch]).to(device)
                    b_f = torch.cat([i.f for i in batch],dim=0).to(device)
                    # Compute the energy and forces
                    energy, force = eval_batch_DP(model, batch)

                    # Get the forces in energy and forces
                    # energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    # force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    energy_loss = F.huber_loss(energy / b_n_atoms, b_y / b_n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, b_f, reduction='mean', delta=huber_force)
                    
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss

                    # Iterate backwards
                    total_loss.backward()
                    optimizer.step()

                    # Add the losses to a log
                    with torch.no_grad():
                        train_losses['train_loss_force'].append(force_loss.item())
                        train_losses['train_loss_energy'].append(energy_loss.item())
                        train_losses['train_loss_total'].append(total_loss.item())

                # Compute the average loss for the batch
                train_losses = dict((k, np.mean(v)) for k, v in train_losses.items())

                # Get the validation loss
                valid_losses = defaultdict(list)
                for batch in valid_loader:
                    # batch.to(device)
                    energy, force = eval_batch_DP(model, batch)

                    # get list info (dataparallel)
                    b_n_atoms = torch.Tensor([i.n_atoms for i in batch]).to(device)
                    b_y = torch.Tensor([i.y for i in batch]).to(device)
                    b_f = torch.cat([i.f for i in batch],dim=0).to(device)
                    # Get the loss of this batch
                    # energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    # force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    energy_loss = F.huber_loss(energy / b_n_atoms, b_y / b_n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, b_f, reduction='mean', delta=huber_force)
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss

                    with torch.no_grad():
                        valid_losses['valid_loss_force'].append(force_loss.item())
                        valid_losses['valid_loss_energy'].append(energy_loss.item())
                        valid_losses['valid_loss_total'].append(total_loss.item())

                valid_losses = dict((k, np.mean(v)) for k, v in valid_losses.items())

                # Reduce the learning rate
                scheduler.step(valid_losses['valid_loss_total'])

                # Save the best model if possible
                if valid_losses['valid_loss_total'] < best_loss:
                    best_loss = valid_losses['valid_loss_total']
                    torch.save(model, td / 'best_model')

                # Store the log line
                print(f"epoch:{epoch}, time:{time.perf_counter() - start_time}, train_loss:{train_losses}, valid_loss:{valid_losses}")
                log.append({'epoch': epoch, 'time': time.perf_counter() - start_time, **train_losses, **valid_losses})

            # Load the best model back in
            best_model = torch.load(td / 'best_model', map_location='cpu')
            return TorchMessage(best_model), pd.DataFrame(log)
    

    def train_DDP(self,
              local_rank:int,
              nproc_per_node,
              nnode,
              node_rank,
              model_msg: ModelMsgType,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cuda',
              batch_size: int = 128,
              learning_rate: float = 1e-3,
              huber_deltas: (float, float) = (0.5, 1),
              energy_weight: float = 0.1,
              reset_weights: bool = False,
              patience: int = None,
              save_path=None,
              cpu=1) -> (TorchMessage, pd.DataFrame):
        
        ## setup for DDP
        print_log = open("/home/lizz_lab/cse12232433/running.log", "w")
        print(f"model_msg 111111: {model_msg}", file=print_log)
        prepare_time = time.time()
        global_rank = local_rank + node_rank * nproc_per_node
        world_size = nnode * nproc_per_node
        os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12345'
        # set unused port
        port = get_available_port()
        os.environ['MASTER_PORT'] = str(port)
        # dist.init_process_group('nccl', rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device=torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", init_method='env://', rank=global_rank, world_size=world_size, timeout=datetime.timedelta(seconds=5))

        
        print(f"model_msg 2222: {model_msg}", file=print_log)
        model = self.get_model(model_msg)
        model.to(device)
        ## lets use multiple GPUs
        model = DistributedDataParallel(model, device_ids=[local_rank])

        
        # Unpack some inputs
        huber_eng, huber_force = huber_deltas

        # If desired, re-initialize weights
        if reset_weights:
            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        # print(f"prepare time consume{time.time()-prepare_time}")
        logger.debug(f"prepare time consume{time.time()-prepare_time}")
        # Start the training process
        with TemporaryDirectory(prefix='spk') as td:
            td = Path(td)
            # Save the batch to an ASE Atoms database
            with open(os.devnull, 'w') as fp, redirect_stderr(fp):
                train_dataset = AtomsDataset.from_atoms(train_data, td / 'train')
                train_sampler = DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)

                valid_dataset = AtomsDataset.from_atoms(valid_data, td / 'valid')
                valid_sampler = DistributedSampler(valid_dataset)
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)


            # Make the trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            if patience is None:
                patience = num_epochs // 8
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.8, min_lr=1e-6)

            # Store the best loss
            best_loss = torch.inf

            # Loop over epochs
            log = []
            model.train()
            start_time = time.perf_counter()
            for epoch in range(num_epochs):
                # Iterate over all batches in the training set
                train_losses = defaultdict(list)
                train_sampler.set_epoch(epoch)
                for batch in train_loader:
                    # transimit_time = time.time()
                    batch.to(device)
                    # print(f"batch transimit_time:{time.time()-transimit_time}")

                    optimizer.zero_grad()

                    # get list info (dataparallel)
                    # b_n_atoms = torch.Tensor([i.n_atoms for i in batch])
                    # b_y = torch.Tensor([i.y for i in batch])
                    # b_f = torch.cat([i.f for i in batch],dim=0)
                    # Compute the energy and forces
                    energy, force = eval_batch(model, batch)

                    # Get the forces in energy and forces
                    energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    # energy_loss = F.huber_loss(energy / b_n_atoms, b_y / b_n_atoms, reduction='mean', delta=huber_eng)
                    # force_loss = F.huber_loss(force, b_f, reduction='mean', delta=huber_force)
                    
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss

                    # Iterate backwards
                    # back_ward_time = time.time()
                    total_loss.backward()
                    optimizer.step()
                    # print(f"backward time {time.time() - back_ward_time}")

                    # Add the losses to a log
                    with torch.no_grad():
                        train_losses['train_loss_force'].append(force_loss.item())
                        train_losses['train_loss_energy'].append(energy_loss.item())
                        train_losses['train_loss_total'].append(total_loss.item())

                # Compute the average loss for the batch
                train_losses = dict((k, np.mean(v)) for k, v in train_losses.items())

                # Get the validation loss
                valid_losses = defaultdict(list)
                for batch in valid_loader:
                    batch.to(device)
                    energy, force = eval_batch(model, batch)

                    # get list info (dataparallel)
                    # b_n_atoms = torch.Tensor([i.n_atoms for i in batch])
                    # b_y = torch.Tensor([i.y for i in batch])
                    # b_f = torch.cat([i.f for i in batch],dim=0)
                    # Get the loss of this batch
                    energy_loss = F.huber_loss(energy / batch.n_atoms, batch.y / batch.n_atoms, reduction='mean', delta=huber_eng)
                    force_loss = F.huber_loss(force, batch.f, reduction='mean', delta=huber_force)
                    # energy_loss = F.huber_loss(energy / b_n_atoms, b_y / b_n_atoms, reduction='mean', delta=huber_eng)
                    # force_loss = F.huber_loss(force, b_f, reduction='mean', delta=huber_force)
                    total_loss = energy_weight * energy_loss + (1 - energy_weight) * force_loss
                    
                    with torch.no_grad():
                        valid_losses['valid_loss_force'].append(force_loss.item())
                        valid_losses['valid_loss_energy'].append(energy_loss.item())
                        valid_losses['valid_loss_total'].append(total_loss.item())
                valid_losses = dict((k, np.mean(v)) for k, v in valid_losses.items())

                # Reduce the learning rate
                scheduler.step(valid_losses['valid_loss_total'])
                # print(f"epoch:{epoch}, time:{time.perf_counter() - start_time}, train_loss:{train_losses}, valid_loss:{valid_losses}")
                logger.debug(f"epoch:{epoch}, time:{time.perf_counter() - start_time}, train_loss:{train_losses}, valid_loss:{valid_losses}")
                # Save the best model if possible
                dist.barrier()
                if valid_losses['valid_loss_total'] < best_loss and local_rank == 0:

                    best_loss = valid_losses['valid_loss_total']
                    # best_model_state_dict = model.module.state_dict()
                    # best_model = self.get_model(model_msg).load_state_dict(best_model_state_dict)
                    torch.save(model.module, td / 'best_model')

                # Store the log line
                log.append({'epoch': epoch, 'time': time.perf_counter() - start_time, **train_losses, **valid_losses})
                dist.barrier()

            # Load the best model back in
            if local_rank ==0:
                # save to save_path in this process
                # best_model = torch.load(td / 'best_model')
                import shutil
                # shutil.move(td / 'best_model', 'best_model')
                # shutil.move(td / 'best_model', os.environ['HOME'] + '/best_model')
                shutil.move(td / 'best_model', save_path / 'best_model')
                # with open(os.environ['HOME'] + '/training-history.json', 'w') as fp:
                with open(save_path / 'training-history.json', 'w') as fp:
                    print(json.dumps(pd.DataFrame(log).to_dict(orient='list')), file=fp)
            else:
                pass
            # clean up DDP
            dist.destroy_process_group()
            
        
    # def start_DDP(self, model_msg, num_epochs,patience,reset_weights, huber_deltas, train_data, valid_data, gpu:list[int], device="cuda", cpu=1, *args, **kwargs):
    def train(self, model_msg, train_data, valid_data, num_epochs, device="cuda", patience:int = None ,reset_weights:bool = False, huber_deltas: (float, float) = (0.5, 1),  gpu:list[int] = [1],  cpu=1, parallel=0, *args, **kwargs):
        """entry function to choose a train method

        Args:
            model_msg (_type_): torch model message
            num_epochs (_type_): _description_
            patience (_type_): _description_
            reset_weights (_type_): _description_
            huber_deltas (_type_): _description_
            train_data (_type_): _description_
            valid_data (_type_): _description_
            gpu (list[int]): _description_
            device (str, optional): _description_. Defaults to "cuda".
            cpu (int, optional): _description_. Defaults to 1.
            parallel (int, optional): 0 to choose basic train on one GPU, 1 to choose DP train, 2 to choose DDP train. Defaults to 2.

        Returns:
            _type_: _description_
        """
        
        # logger.debug("model_msg: ${model_msg}")
        # print to file
        # print_log = open("/home/lizz_lab/cse12232433/running.log", "w")
        # print(f"model_msg: {model_msg}", file=print_log)
        # print_log.close()
        from functools import partial, update_wrapper
        def _wrap(func, **kwargs):
            out = partial(func, **kwargs)
            update_wrapper(out, func)
            return out
        if parallel == 2:
            with TemporaryDirectory(dir=os.environ['HOME'], prefix="DDP_save_path_") as save_path:
                save_path = Path(save_path)
                run_DDP = _wrap(self.train_DDP,
                                model_msg=model_msg,
                                train_data=train_data,
                                valid_data=valid_data,
                                nproc_per_node=len(gpu), nnode=1, node_rank=0,  num_epochs=num_epochs,device=device,patience=patience,reset_weights=reset_weights,huber_deltas=huber_deltas,save_path=save_path, cpu=cpu)
                # print("start DDP training")
                logger.debug("start DDP training on multiple GPUs")
                mp.spawn(run_DDP, nprocs=len(gpu),join=True)
                # best_model = torch.load(os.environ['HOME'] + '/best_model')
                # log = pd.read_json(os.environ['HOME'] + '/training-history.json')
                best_model = torch.load(save_path / 'best_model')
                log = pd.read_json(save_path / 'training-history.json')
                return TorchMessage(best_model), log
        elif parallel == 1:
            raise NotImplementedError
        elif parallel == 0:
            # need to manually manage what resources run on
            return self.train_basic(model_msg=model_msg, num_epochs=num_epochs, patience=patience, reset_weights=reset_weights, huber_deltas=huber_deltas, train_data=train_data, valid_data=valid_data)
