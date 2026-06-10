"""
Training script with WSD scheduler and split checkpointing for Adam/SGD gap analysis.

Run two phases:
  1. w/o --dir_ckpt: WSD training, saves every save_every_steps.
  2. w/  --dir_ckpt: loads a checkpoint, runs only the decay.
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.amp import autocast
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset

from .data import get_dataloaders
from .model import gpt_small


def wsd_scheduler(base_lr, final_lr, total_steps, warm_steps=0, cool_steps=0):
    warm_schedule = np.array([])
    cool_schedule = np.array([])
    if warm_steps > 0:
        warm_schedule = np.linspace(0, base_lr, warm_steps + 1)[1:]
    if cool_steps > 0:
        cool_schedule = np.linspace(base_lr, final_lr, cool_steps + 1)[1:]
    stable_schedule = np.array([base_lr] * (total_steps - warm_steps - cool_steps))
    schedule = np.concatenate((warm_schedule, stable_schedule, cool_schedule))
    assert len(schedule) == total_steps
    return schedule


def train(
    model,
    optimizer,
    scheduler,
    tr_loader,
    vl_loader,
    criterion,
    args,
    dir_output,
    target_steps,
    steps_per_epoch,
    save_every_steps=0,
):
    tr_loss = 0.0
    best_loss = float("inf")
    global_step = 0

    while global_step < target_steps:
        model.train()
        for step, (x, y) in enumerate(tr_loader):
            if step >= steps_per_epoch:
                break

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = (
                    criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    / args.accum_steps
                )
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.accum_steps == 0:
                lr = scheduler[global_step]
                for p in optimizer.param_groups:
                    p["lr"] = lr
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # decouple weight decay from learning rate
                # pytorch adamw: p -= lr * wd * p
                # true decouple: p -= wd * p
                with torch.no_grad():
                    mult = lr / args.lr
                    for _, p in model.named_parameters():
                        p.data.mul_(1 - args.wd * mult)

                # Log train metrics
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

                # Log eval metrics
                if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                    model.eval()
                    vl_loss, vl_n = 0.0, 0
                    with torch.no_grad():
                        for x, y in vl_loader:
                            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                            with autocast("cuda", dtype=torch.bfloat16):
                                out = model(x)
                            vl_loss += criterion(
                                out.view(-1, out.size(-1)), y.view(-1)
                            ).item() * x.size(0)
                            vl_n += x.size(0)
                    vl_loss /= vl_n
                    vl_ppl = math.exp(min(vl_loss, 20))

                    print(f"step {global_step} vl_loss {vl_loss:.4f}")
                    wandb.log({"val/loss": vl_loss, "val/ppl": vl_ppl})
                    w = model.state_dict()
                    torch.save(w, os.path.join(dir_output, "last.pth"))
                    if vl_loss < best_loss:
                        best_loss = vl_loss
                        torch.save(w, os.path.join(dir_output, "best.pth"))
                    model.train()

                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    torch.save(
                        {"step": global_step, "model": model.state_dict()},
                        os.path.join(dir_output, f"step_{global_step}.pth"),
                    )
                    print(f"saved checkpoint at step {global_step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--data", type=str, default="fineweb")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warm_ratio", type=float, default=0.1)
    parser.add_argument("--cool_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_embed", type=int, default=384)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dir_output", type=str, required=True)
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--dir_ckpt", type=str, default="")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_every_ratio", type=float, default=0.1)
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

    tr_loader, vl_loader, steps_per_epoch = get_dataloaders(
        dataset=args.data,
        dir_data=args.dir_data,
        seq_len=args.seq_len,
        batch_size=args.bs,
        n_workers=args.n_workers,
    )
    tr_ds = tr_loader.dataset
    perm = np.random.permutation(len(tr_ds))
    print(f"tr ds size: {len(tr_ds):,} samples")
    print(f"vl ds size: {len(vl_loader.dataset):,} samples")

    model = gpt_small(
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_embed=args.d_embed,
    ).cuda()
    if args.compile:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.opt == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    elif args.opt == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.mom, args.mom))
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    # Phase 2 training. There are two cases for
    #   one-epoch training: make sure that data order is identical as the phase 1.
    #       then, we need to skip the first steps containing already seen samples.
    #   multi-epoch training: make sure that the total number of steps is identical.
    #       here, we do not care about the data order to be exactly the same as phase 1.

    if args.dir_ckpt:
        ckpt = torch.load(args.dir_ckpt, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        print(f"load checkpoint from step {ckpt['step']}")

        if args.epochs == 1:
            start = ckpt["step"] * args.accum_steps * args.bs
            tr_loader = DataLoader(
                Subset(tr_ds, perm[start:]),
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.n_workers,
            )
            print(f"skip first {start:} training samples")
            steps_per_epoch = (len(tr_loader) // args.accum_steps) * args.accum_steps
            total_steps = steps_per_epoch // args.accum_steps
        else:
            steps_per_epoch = (steps_per_epoch // args.accum_steps) * args.accum_steps
            total_steps = args.epochs * steps_per_epoch // args.accum_steps
            total_steps -= ckpt["step"]

        print(f"dataset size: {len(tr_loader.dataset)}")
        cool_steps = int(args.cool_ratio * total_steps)
        scheduler = wsd_scheduler(args.lr, 0, total_steps, 0, cool_steps)

        train(
            model,
            optimizer,
            scheduler,
            tr_loader,
            vl_loader,
            criterion,
            args,
            dir_output,
            total_steps,
            steps_per_epoch,
        )
    else:
        if args.epochs == 1:
            tr_loader = DataLoader(
                Subset(tr_ds, perm),
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.n_workers,
            )
        print(f"dataset size: {len(tr_loader.dataset)}")
        steps_per_epoch = (steps_per_epoch // args.accum_steps) * args.accum_steps
        total_steps = args.epochs * steps_per_epoch // args.accum_steps
        warm_steps = int(args.warm_ratio * total_steps)
        cool_steps = int(args.cool_ratio * total_steps)
        scheduler = wsd_scheduler(args.lr, 0, total_steps, warm_steps, cool_steps)

        train(
            model,
            optimizer,
            scheduler,
            tr_loader,
            vl_loader,
            criterion,
            args,
            dir_output,
            total_steps,
            steps_per_epoch,
            int(args.save_every_ratio * total_steps),
        )


if __name__ == "__main__":
    main()
