import argparse
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler

from dataset import Dataset
from model import Model
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='tolokers-tab',
                        choices=['tolokers-tab', 'questions-tab', 'city-reviews', 'browser-games', 'hm-categories',
                                 'web-fraud', 'city-roads-M', 'city-roads-L', 'avazu-devices', 'hm-prices',
                                 'web-traffic', 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers',
                                 'questions', 'cora', 'citeseer', 'pubmed', 'coauthor-cs', 'coauthor-physics',
                                 'amazon-computers', 'amazon-photo', 'lastfm-asia', 'facebook', 'ogbn-arxiv',
                                 'ogbn-products'])

    # Additional node features.
    parser.add_argument('--use_node_embeddings', default=False, action='store_true',
                        help='DeepWalk node embeddings can be used for the city-roads-M and city-roads-L datasets '
                             '(where they are very beneficial).')

    # Numerical features preprocessing.
    parser.add_argument('--numerical_features_transform', type=str, default='quantile-transform-normal',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'],
                        help='Only used for TabGraphs datasets that have numerical features.')
    parser.add_argument('--numerical_features_nan_imputation_strategy', type=str, default='most_frequent',
                        choices=['mean', 'median', 'most_frequent'],
                        help='Only used for TabGraphs datasets that have NaNs in numerical features.')

    # PLR embeddings for numerical features.
    parser.add_argument('--plr', default=False, action='store_true',
                        help='Use PLR embeddings for numerical features. Can only be used for TabGraphs datasets.')
    parser.add_argument('--plr_frequencies_dim', type=int, default=48, help='Only used if plr is True.')
    parser.add_argument('--plr_frequencies_scale', type=float, default=0.01, help='Only used if plr is True.')
    parser.add_argument('--plr_embedding_dim', type=int, default=16, help='Only used if plr is True.')
    parser.add_argument('--plr_lite', default=False, action='store_true', help='Only used if plr is True.')

    # Regression targets preprocessing.
    parser.add_argument('--regression_targets_transform', type=str, default='none',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'])

    # Model architecture.
    parser.add_argument('--model', type=str, default='GraphSAGE',
                        choices=['ResNet', 'GCN', 'GraphSAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['none', 'LayerNorm', 'BatchNorm'])

    # regularization
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args


def train_step(model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()

    with autocast(enabled=amp):
        preds = model(graph=dataset.graph, x=dataset.features)
        loss = dataset.loss_fn(input=preds[dataset.train_idx], target=dataset.targets[dataset.train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset, amp=False):
    model.eval()

    with autocast(enabled=amp):
        preds = model(graph=dataset.graph, x=dataset.features)

    metrics = dataset.compute_metrics(preds)

    return metrics


def main():
    args = get_args()

    torch.manual_seed(0)

    dataset = Dataset(name=args.dataset,
                      add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                      use_node_embeddings=args.use_node_embeddings,
                      numerical_features_transform=args.numerical_features_transform,
                      numerical_features_nan_imputation_strategy=args.numerical_features_nan_imputation_strategy,
                      regression_targets_transform=args.regression_targets_transform,
                      device=args.device)

    logger = Logger(args, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, args.num_runs + 1):
        model = Model(model_name=args.model,
                      num_layers=args.num_layers,
                      features_dim=dataset.features_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=dataset.targets_dim,
                      num_heads=args.num_heads,
                      hidden_dim_multiplier=args.hidden_dim_multiplier,
                      normalization=args.normalization,
                      dropout=args.dropout,
                      use_plr=args.plr,
                      numerical_features_mask=dataset.numerical_features_mask,
                      plr_frequencies_dim=args.plr_frequencies_dim,
                      plr_frequencies_scale=args.plr_frequencies_scale,
                      plr_embedding_dim=args.plr_embedding_dim,
                      use_plr_lite=args.plr_lite)

        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp)
                metrics = evaluate(model=model, dataset=dataset, amp=args.amp)
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})

        logger.finish_run()
        model.cpu()
        dataset.next_data_split()

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
