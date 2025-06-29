import os
import argparse
from copy import deepcopy


import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from misc import add_dict_to_argparser
from data import get_target_fn
from network import ToyNet
from diffusion import Follmer


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(device=0, seed=42, mode='denoiser',
                    data_name='m1', nsample=5000, output='logs/m1/ckpt',
                    data_dim=1, cond_dim=5, sigma_data=1, M=1,
                    bsz=200, train_steps=5000, lr=1e-3, 
                    dump_freq=1000, print_freq=500,
                    ode_solver='euler', sde_solver='euler-maruyama', 
                    eps0=1e-3, eps1=1e-3, num_steps=1000, heun_steps=13, teacher_path='')
    add_dict_to_argparser(parser, defaults)
    return parser

def denoiser_matching(args):
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.seed)

    target_fn = get_target_fn(args.data_name)
    cond = np.random.randn(args.nsample, args.cond_dim)
    data = target_fn(cond)
    data = torch.from_numpy(data).float()
    cond = torch.from_numpy(cond).float()
    dataset = TensorDataset(data, cond)
    loader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, drop_last=True)
    def create_infinite_dataloader(loader):
        while True:
            yield from loader
    loader = create_infinite_dataloader(loader)
    model = ToyNet(args.data_dim, args.cond_dim, hidden_dims=[32, 16]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sde = Follmer(args)

    for step in range(1, args.train_steps+1):
        batch, cond = next(loader)
        batch  = batch.to(device)
        cond = cond.to(device)
        optim.zero_grad()
        loss = sde.compute_dsm_loss(model, batch, cond)
        loss.backward()
        optim.step()
        if step % args.print_freq == 0:
            print(f"Step[{step}/{args.train_steps}], Loss {loss.item():.4f}")
        if step % args.dump_freq == 0 or step == args.train_steps:
            torch.save(dict(model=model.state_dict(), optim=optim.state_dict, step=step), 
                    f"{args.output}/{step}.pth")

def trajectory_matching(args):
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    np.random.seed(args.seed)

    target_fn = get_target_fn(args.data_name)
    cond = np.random.randn(args.nsample, args.cond_dim)
    data = target_fn(cond)
    data_cond = np.hstack([data, cond]).astype("float32")
    data_cond = torch.from_numpy(data_cond)
    dataset = TensorDataset(data_cond[:, :args.data_dim], data_cond[:, args.data_dim:])
    loader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, drop_last=True)
    def create_infinite_dataloader(loader):
        while True:
            yield from loader
    loader = create_infinite_dataloader(loader)
    model = ToyNet(args.data_dim, args.cond_dim, True, hidden_dims=[32, 16]).to(device)
    teacher_model = ToyNet(args.data_dim, args.cond_dim, hidden_dims=[32, 16]).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device)['model'])
    model.load_state_dict(teacher_model.state_dict(), strict=False)
    target_model = deepcopy(model).to(device)
    for params in teacher_model.parameters():
        params.requires_grad_(False)
    for params in target_model.parameters():
        params.requires_grad_(False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sde = Follmer(args)

    for step in range(1, args.train_steps+1):
        batch, cond = next(loader)
        batch  = batch.to(device)
        cond = cond.to(device)
        optim.zero_grad()
        loss = sde.compute_traj_loss(model, target_model, teacher_model, batch, cond)
        loss.backward()
        optim.step()
        target_model.load_state_dict(model.state_dict(), strict=False)
        if step % args.print_freq == 0:
            print(f"Step[{step}/{args.train_steps}], Loss {loss.item():.4f}")
        if step % args.dump_freq == 0 or step == args.train_steps:
            torch.save(dict(model=model.state_dict(), optim=optim.state_dict, step=step), 
                    f"{args.output}/traj-{step}.pth")


if __name__ == '__main__':
    args = create_parser().parse_args()
    if args.mode == 'denoiser':
        denoiser_matching(args)
    elif args.mode == 'trajectory':
        trajectory_matching(args)
    else:
        raise ValueError(f"unsupported training mode {args.mode}")

