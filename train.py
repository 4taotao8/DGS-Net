"""Train the released DGS-Net architecture on generated scene annotations."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dgs_net.dataset import GraspDataset
from dgs_net.loss import get_loss_and_acc
from dgs_net.model import PointTransformerV3, dataloader_preprosess


def run_epoch(model, loader, optimizer=None, accumulation_steps=1):
    if len(loader) == 0:
        raise ValueError('No batches available; check the dataset and batch size.')
    if accumulation_steps < 1:
        raise ValueError('accumulation_steps must be at least 1.')
    training = optimizer is not None
    model.train(training)
    totals = {}
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (points, targets) in enumerate(loader, 1):
        inputs = dataloader_preprosess(points)
        with torch.set_grad_enabled(training):
            predictions = model(inputs, targets)
            losses, metrics = get_loss_and_acc(predictions, targets)
            if not torch.isfinite(losses['total_loss']):
                raise RuntimeError(f'Non-finite loss at batch {step}; inspect its grasp labels.')
            if training:
                group_start = ((step - 1) // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(loader) - group_start)
                (losses['total_loss'] / group_size).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        for name, value in {**losses, **metrics}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / len(loader) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--resume', type=Path, help='Initialize weights from a state_dict checkpoint')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    if not (args.data_root / 'scenes').is_dir():
        parser.error(f'Missing scene data in {args.data_root / "scenes"}; '
                     'set --data-root to the dataset directory.')
    cfg = yaml.safe_load(Path('config.yaml').read_text())
    train_cfg = cfg['train']
    batch_size = args.batch_size or train_cfg['batchsize']
    epochs = args.epochs or train_cfg['epoches']
    workers = train_cfg['workers'] if args.workers is None else args.workers
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU and the CUDA extensions listed in README.md.')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    loaders = {
        split: DataLoader(GraspDataset(args.data_root, split=split), batch_size=batch_size,
                          shuffle=(split == 'train'), num_workers=workers, drop_last=True)
        for split in ('train', 'test')
    }
    model = PointTransformerV3(in_channels=3, batch_size=batch_size, is_training=True).cuda()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'],
                                 betas=(0.9, 0.999), eps=1e-8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(epochs):
        train_metrics = run_epoch(model, loaders['train'], optimizer,
                                  train_cfg['accumulation_steps'])
        test_metrics = run_epoch(model, loaders['test'])
        print(f"epoch {epoch + 1}/{epochs}: train={train_metrics['total_loss']:.4f}, "
              f"test={test_metrics['total_loss']:.4f}, pose_acc={test_metrics['pose_acc']:.4f}")
        if test_metrics['total_loss'] < best_loss:
            best_loss = test_metrics['total_loss']
            torch.save(model.state_dict(), args.output_dir / 'best.pth')


if __name__ == '__main__':
    main()
