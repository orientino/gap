"""
Training script for GCN/GAT on graph regression datasets.
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.amp import autocast
from torch.optim import SGD, Adam

from .data import get_dataloaders
from .model import GAT, GCN
from .model_grit import GRIT, AddFullRRWP


def cosine_scheduler(base_lr, final_lr, total_steps, warm_steps=0):
    warm_schedule = np.array([])
    if warm_steps > 0:
        warm_schedule = np.linspace(0, base_lr, warm_steps + 1)[1:]
    iters = np.arange(total_steps - warm_steps)
    schedule = final_lr + 0.5 * (base_lr - final_lr) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warm_schedule, schedule))
    assert len(schedule) == total_steps
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=str, default="gcn")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--data", type=str, default="zinc12k")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warm_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_embed", type=int, default=256)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dir_output", type=str, required=True)
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--ksteps", type=int, default=21)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dir_output = os.path.join(args.dir_output, args.run_name)
    os.makedirs(dir_output, exist_ok=True)
    wandb.init(project="adam-sgd-gap", mode="disabled" if args.debug else "online")
    wandb.config.update(args)
    wandb.config.update({"bs": args.bs * args.accum_steps}, allow_val_change=True)

    pre_transform = AddFullRRWP(walk_length=args.ksteps) if args.m == "grit" else None
    tr_loader, vl_loader, steps_per_epoch = get_dataloaders(
        args.data,
        dir_data=args.dir_data,
        batch_size=args.bs,
        n_workers=args.n_workers,
        pre_transform=pre_transform,
    )
    print(f"tr ds size: {len(tr_loader.dataset):,} graphs")
    print(f"vl ds size: {len(vl_loader.dataset):,} graphs")

    if args.m == "gat":
        model = GAT(
            n_layers=args.n_layers, d_embed=args.d_embed, n_heads=args.n_heads
        ).cuda()
    elif args.m == "grit":
        model = GRIT(
            n_layers=args.n_layers,
            d_embed=args.d_embed,
            n_heads=args.n_heads,
            ksteps=args.ksteps,
        ).cuda()
    else:
        model = GCN(n_layers=args.n_layers, d_embed=args.d_embed).cuda()
    if args.compile:
        model = torch.compile(model)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6}M")

    steps_per_epoch = (steps_per_epoch // args.accum_steps) * args.accum_steps
    total_steps = args.epochs * steps_per_epoch // args.accum_steps
    if args.opt == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    elif args.opt == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.mom, args.mom))
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")
    scheduler = cosine_scheduler(
        base_lr=args.lr,
        final_lr=0,
        total_steps=total_steps,
        warm_steps=int(args.warm_ratio * total_steps),
    )

    tr_loss = 0.0
    best_loss = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for step, data in enumerate(tr_loader):
            if step >= steps_per_epoch:
                break

            data = data.cuda(non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(data)
            loss = F.l1_loss(pred.float(), data.y) / args.accum_steps
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.accum_steps == 0:
                lr = scheduler[global_step]
                for p in optimizer.param_groups:
                    p["lr"] = lr
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                with torch.no_grad():
                    mult = lr / args.lr
                    for n, p in model.named_parameters():
                        p.data.mul_(1 - args.wd * mult)

                if global_step % args.log_interval == 0:
                    print(f"step {global_step} tr_loss {tr_loss:.4f} lr {lr:.6f}")
                    wandb.log(
                        {
                            "train/loss": tr_loss,
                            "train/lr": lr,
                            "train/wd": args.wd * mult,
                            "train/w_norm": torch.sqrt(
                                sum(p.data.norm() ** 2 for p in model.parameters())
                            ).item(),
                        }
                    )
                if math.isnan(tr_loss):
                    return
                tr_loss = 0.0

                if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                    model.eval()
                    vl_loss, vl_n = 0.0, 0
                    with torch.no_grad():
                        for vl_data in vl_loader:
                            vl_data = vl_data.cuda(non_blocking=True)
                            with autocast("cuda", dtype=torch.bfloat16):
                                vl_pred = model(vl_data)
                            vl_loss += F.l1_loss(
                                vl_pred.float(), vl_data.y, reduction="sum"
                            ).item()
                            vl_n += vl_data.y.size(0)
                    vl_loss = vl_loss / vl_n
                    print(f"step {global_step} vl_loss {vl_loss:.4f}")
                    wandb.log({"val/loss": vl_loss})
                    ckpt = {
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(dir_output, "last.pth"))
                    if vl_loss < best_loss:
                        best_loss = vl_loss
                        torch.save(ckpt, os.path.join(dir_output, "best.pth"))
                    model.train()


if __name__ == "__main__":
    main()
