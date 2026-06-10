"""
Training script for ViT-S/16 on ImageNet-1k with DDP.
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

from .data import get_dataloaders
from .model import vit_small_patch16
from .model_convnext import convnext_tiny
from .model_resnet import resnet18, resnet50
from .model_transition import resnet_transition


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


def mixup(x, y, n_classes, p=0.2):
    """https://github.com/google-research/big_vision/blob/main/big_vision/utils.py#L1146"""
    a = np.random.beta(p, p)
    a = max(a, 1 - a)  # ensure a >= 0.5 so that `unrolled x` is dominant
    mixed_x = a * x + (1 - a) * x.roll(1, dims=0)
    y_onehot = torch.zeros(y.size(0), n_classes, device=y.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)  # one-hot encoding
    mixed_y = a * y_onehot + (1 - a) * y_onehot.roll(1, dims=0)
    return mixed_x, mixed_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=str, default="vit")
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--data", type=str, default="c10")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--warm_ratio", type=float, default=0.1)
    parser.add_argument("--mixup_p", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_embed", type=int, default=384)
    parser.add_argument("--depthwise", action="store_true")
    parser.add_argument("--patchify_stem", action="store_true")
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--gelu", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--dir_output", type=str, required=True)
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
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

    tr_loader, vl_loader, n_classes, steps_per_epoch = get_dataloaders(
        dataset=args.data,
        dir_data=args.dir_data,
        batch_size=args.bs,
        n_workers=args.n_workers,
        aug=args.aug,
    )
    print(f"tr ds size: {len(tr_loader.dataset):,} images")
    print(f"vl ds size: {len(vl_loader.dataset):,} images")

    if args.m == "convnext":
        model = convnext_tiny(n_classes=n_classes).cuda()
    elif args.m == "resnet18":
        model = resnet18(n_classes=n_classes).cuda()
    elif args.m == "resnet50":
        model = resnet50(n_classes=n_classes).cuda()
    elif args.m == "resnet_transition":
        model = resnet_transition(
            n_classes=n_classes,
            depthwise=args.depthwise,
            patchify_stem=args.patchify_stem,
            layernorm=args.layernorm,
            gelu=args.gelu,
        ).cuda()
    else:
        model = vit_small_patch16(
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_embed=args.d_embed,
            n_classes=n_classes,
        ).cuda()
    if args.compile:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()

    # in underparametrized setting, drop the last few number of steps
    # so that they are not accumulated together with the next epoch.
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
    best_acc = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        # Train
        model.train()
        for step, (x, y) in enumerate(tr_loader):
            if step >= steps_per_epoch:
                break

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                if args.mixup_p > 0:
                    x, y_soft = mixup(x, y, n_classes, p=args.mixup_p)
                    logits = model(x)
                    loss = -torch.sum(y_soft * torch.log_softmax(logits, dim=1), dim=1).mean()  # fmt: skip
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                loss = loss / args.accum_steps
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
                    for n, p in model.named_parameters():
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

                # Log eval metrics and save
                if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                    model.eval()
                    vl_loss, vl_correct1, vl_correct5, vl_n = 0, 0, 0, 0
                    with torch.no_grad():
                        for x, y in vl_loader:
                            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                            with autocast("cuda", dtype=torch.bfloat16):
                                out = model(x)
                            vl_loss += criterion(out, y).item() * x.size(0)
                            top5 = out.topk(5, dim=1)[1]
                            vl_correct1 += top5[:, 0].eq(y).sum().item()
                            vl_correct5 += top5.eq(y.view(-1, 1)).sum().item()
                            vl_n += x.size(0)
                    metrics = [vl_loss, vl_correct1, vl_correct5, vl_n]
                    metrics = torch.tensor(metrics, device="cuda")
                    vl_loss, vl_acc1, vl_acc5 = (
                        metrics[0].item() / metrics[3].item(),
                        metrics[1].item() / metrics[3].item() * 100,
                        metrics[2].item() / metrics[3].item() * 100,
                    )
                    print(f"step {global_step} vl_acc1 {vl_acc1:.2f}")
                    wandb.log(
                        {
                            "val/loss": vl_loss,
                            "val/acc1": vl_acc1,
                            "val/acc5": vl_acc5,
                        }
                    )
                    w = model.state_dict()
                    torch.save(w, os.path.join(dir_output, "last.pth"))
                    if vl_acc1 > best_acc:
                        best_acc = vl_acc1
                        torch.save(w, os.path.join(dir_output, "best.pth"))
                    model.train()


if __name__ == "__main__":
    main()
