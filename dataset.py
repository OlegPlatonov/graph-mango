import os
import yaml
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import dgl

from sklearn.preprocessing import (FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
                                   QuantileTransformer, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import train_test_split

from torch_geometric import datasets as pyg_datasets
from ogb.nodeproppred import DglNodePropPredDataset


class Dataset:
    # Datasets by source.
    # Automatic downloading is currently not supported for TabGraphs datasets. If you want to use one of these datasets,
    # put it in the data directory.
    tabgraphs_dataset_names = ['tolokers-tab', 'questions-tab', 'city-reviews', 'browser-games', 'hm-categories',
                               'web-fraud', 'city-roads-M', 'city-roads-L', 'avazu-devices', 'hm-prices',
                               'web-traffic']
    pyg_dataset_names = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'cora', 'citeseer',
                         'pubmed', 'coauthor-cs', 'coauthor-physics', 'amazon-computers', 'amazon-photo', 'lastfm-asia',
                         'facebook']
    ogb_dataset_names = ['ogbn-arxiv', 'ogbn-products']

    # Datasets by task.
    multiclass_classification_dataset_names = ['browser-games', 'hm-categories', 'roman-empire', 'amazon-ratings',
                                               'cora', 'citeseer', 'pubmed', 'coauthor-cs', 'coauthor-physics',
                                               'amazon-computers', 'amazon-photo', 'lastfm-asia', 'facebook',
                                               'ogbn-arxiv', 'ogbn-products']
    binary_classification_dataset_names = ['tolokers-tab', 'questions-tab', 'city-reviews', 'web-fraud', 'minesweeper',
                                           'tolokers', 'questions']
    regression_dataset_names = ['city-roads-M', 'city-roads-L', 'avazu-devices', 'hm-prices', 'web-traffic']

    # Not all datasets obtained from PyG have predefined data splits. Random class stratified splits will be used for
    # other datasets.
    pyg_datasets_with_predefined_splits_names = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers',
                                                 'questions']

    transforms = {
        'none': partial(FunctionTransformer, func=lambda x: x, inverse_func=lambda x: x),
        'standard-scaler': partial(StandardScaler, copy=False),
        'min-max-scaler': partial(MinMaxScaler, clip=False, copy=False),
        'robust-scaler': partial(RobustScaler, unit_variance=True, copy=False),
        'power-transform-yeo-johnson': partial(PowerTransformer, method='yeo-johnson', standardize=True, copy=False),
        'quantile-transform-normal': partial(QuantileTransformer, output_distribution='normal', subsample=None,
                                             random_state=0, copy=False),
        'quantile-transform-uniform': partial(QuantileTransformer, output_distribution='uniform', subsample=None,
                                              random_state=0, copy=False)
    }

    def __init__(self, name, add_self_loops=False, use_node_embeddings=False,
                 numerical_features_transform='none', numerical_features_nan_imputation_strategy='most_frequent',
                 regression_targets_transform='none', device='cpu'):
        print('Preparing data...')
        if name in self.tabgraphs_dataset_names:
            graph, features, numerical_features_mask, targets, train_idx_list, val_idx_list, test_idx_list = \
                self.get_tabgraphs_dataset(
                    name,
                    use_node_embeddings=use_node_embeddings,
                    numerical_features_nan_imputation_strategy=numerical_features_nan_imputation_strategy,
                    numerical_features_transform=numerical_features_transform
                )
        elif name in self.pyg_dataset_names:
            graph, features, targets, train_idx_list, val_idx_list, test_idx_list = self.get_pyg_dataset(name)
        elif name in self.ogb_dataset_names:
            graph, features, targets, train_idx_list, val_idx_list, test_idx_list = self.get_ogb_dataset(name)
        else:
            raise ValueError(f'Unkown dataset name: {name}.')

        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)

        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        if name in self.multiclass_classification_dataset_names:
            task = 'multiclass_classification'
            metric = 'accuracy'
            loss_fn = F.cross_entropy
            targets_dim = len(targets.unique())
            targets = targets.to(torch.int64)
        elif name in self.binary_classification_dataset_names:
            task = 'binary_classification'
            metric = 'AP'
            loss_fn = F.binary_cross_entropy_with_logits
            targets_dim = 1
            targets = targets.to(torch.float32)
        elif name in self.regression_dataset_names:
            task = 'regression'
            metric = 'R2'
            loss_fn = F.mse_loss
            targets_dim = 1
            targets_orig = targets.clone()
            targets_transform = self.transforms[regression_targets_transform]()
        else:
            raise RuntimeError(f'The task for dataset {name} is not known.')

        self.name = name
        self.task = task
        self.metric = metric
        self.device = device

        self.graph = graph.to(device)
        self.features = features.to(device)
        self.numerical_features_mask = (
            numerical_features_mask.to(device) if name in self.tabgraphs_dataset_names else None
        )
        self.targets = targets.to(device)

        self.features_dim = features.shape[1]
        self.targets_dim = targets_dim

        self.loss_fn = loss_fn

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        if task == 'regression':
            self.targets_orig = targets_orig.to(device)
            self.targets_transform = targets_transform
            self._transform_regression_targets()

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits
        if self.task == 'regression':
            self._transform_regression_targets()

    def compute_metrics(self, preds):
        if self.metric == 'accuracy':
            preds = preds.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.targets[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.targets[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.targets[self.test_idx]).float().mean().item()

        elif self.metric == 'AP':
            targets = self.targets.cpu().numpy()
            preds = preds.cpu().numpy()

            train_idx = self.train_idx.cpu().numpy()
            val_idx = self.val_idx.cpu().numpy()
            test_idx = self.test_idx.cpu().numpy()

            train_metric = average_precision_score(y_true=targets[train_idx], y_score=preds[train_idx]).item()
            val_metric = average_precision_score(y_true=targets[val_idx], y_score=preds[val_idx]).item()
            test_metric = average_precision_score(y_true=targets[test_idx], y_score=preds[test_idx]).item()

        elif self.metric == 'R2':
            targets_orig = self.targets_orig.cpu().numpy()
            preds_orig = self.targets_transform.inverse_transform(preds.cpu().numpy()[:, None]).squeeze(1)

            train_idx = self.train_idx.cpu().numpy()
            val_idx = self.val_idx.cpu().numpy()
            test_idx = self.test_idx.cpu().numpy()

            train_metric = r2_score(y_true=targets_orig[train_idx], y_pred=preds_orig[train_idx])
            val_metric = r2_score(y_true=targets_orig[val_idx], y_pred=preds_orig[val_idx])
            test_metric = r2_score(y_true=targets_orig[test_idx], y_pred=preds_orig[test_idx])

        else:
            raise ValueError(f'Unknown metric: {self.metric}.')

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    def _transform_regression_targets(self):
        self.targets = self.targets_orig.clone()
        labeled_idx = torch.cat([self.train_idx, self.val_idx, self.test_idx], axis=0)
        self.targets_transform.fit(self.targets[self.train_idx][:, None].cpu().numpy())
        self.targets[labeled_idx] = torch.tensor(
            self.targets_transform.transform(self.targets[labeled_idx][:, None].cpu().numpy()).squeeze(1),
            device=self.device
        )

    @staticmethod
    def get_tabgraphs_dataset(name, use_node_embeddings, numerical_features_nan_imputation_strategy,
                              numerical_features_transform):
        with open(f'data/{name}/info.yaml', 'r') as file:
            info = yaml.safe_load(file)

        features_df = pd.read_csv(f'data/{name}/features.csv', index_col=0)
        numerical_features = features_df[info['num_feature_names']].values.astype(np.float32)
        binary_features = features_df[info['bin_feature_names']].values.astype(np.float32)
        categorical_features = features_df[info['cat_feature_names']].values.astype(np.float32)
        targets = features_df[info['target_name']].values.astype(np.float32)

        if numerical_features.shape[1] > 0:
            numerical_features_transform = Dataset.transforms[numerical_features_transform]()
            numerical_features = numerical_features_transform.fit_transform(numerical_features)

            if info['has_nans_in_num_features']:
                imputer = SimpleImputer(missing_values=np.nan, strategy=numerical_features_nan_imputation_strategy,
                                        copy=False)
                numerical_features = imputer.fit_transform(numerical_features)

        if categorical_features.shape[1] > 0:
            one_hot_encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
            categorical_features = one_hot_encoder.fit_transform(categorical_features)

        if use_node_embeddings:
            node_embeddings = np.load(f'data/{name}/node_embeddings.npz')['node_embeds']

        train_mask = pd.read_csv(f'data/{name}/train_mask.csv', index_col=0).values.reshape(-1)
        train_idx = np.where(train_mask)[0]
        val_mask = pd.read_csv(f'data/{name}/valid_mask.csv', index_col=0).values.reshape(-1)
        val_idx = np.where(val_mask)[0]
        test_mask = pd.read_csv(f'data/{name}/test_mask.csv', index_col=0).values.reshape(-1)
        test_idx = np.where(test_mask)[0]

        edges_df = pd.read_csv(f'data/{name}/edgelist.csv')
        edges = edges_df.values[:, :2]

        features = np.concatenate([numerical_features, binary_features, categorical_features], axis=1)
        if use_node_embeddings:
            features = np.concatenate([features, node_embeddings], axis=1)

        numerical_features_mask = np.zeros(features.shape[1], dtype=bool)
        numerical_features_mask[:numerical_features.shape[1]] = True

        features = torch.from_numpy(features)
        numerical_features_mask = torch.from_numpy(numerical_features_mask)
        targets = torch.from_numpy(targets)

        edges = torch.from_numpy(edges)
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(features), idtype=torch.int32)

        train_idx_list = [torch.from_numpy(train_idx)]
        val_idx_list = [torch.from_numpy(val_idx)]
        test_idx_list = [torch.from_numpy(test_idx)]

        return graph, features, numerical_features_mask, targets, train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def get_pyg_dataset(name):
        if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
            dataset = pyg_datasets.HeterophilousGraphDataset(name=name, root='data')
        elif name in ['cora', 'citeseer', 'pubmed']:
            dataset = pyg_datasets.Planetoid(name=name, root='data')
        elif name in ['coauthor-cs', 'coauthor-physics']:
            dataset = pyg_datasets.Coauthor(name=name.split('-')[1], root=os.path.join('data', 'coauthor'))
        elif name in ['amazon-computers', 'amazon-photo']:
            dataset = pyg_datasets.Amazon(name=name.split('-')[1], root=os.path.join('data', 'amazon'))
        elif name == 'lastfm-asia':
            dataset = pyg_datasets.LastFMAsia(root=os.path.join('data', name))
        elif name == 'facebook':
            dataset = pyg_datasets.FacebookPagePage(root=os.path.join('data', name))
        else:
            raise ValueError(f'Unknown PyG dataset name: {name}.')

        pyg_graph = dataset[0]
        source_nodes, target_nodes = pyg_graph.edge_index
        num_nodes = len(pyg_graph.y)
        dgl_graph = dgl.graph((source_nodes, target_nodes), num_nodes=num_nodes, idtype=torch.int)
        features = pyg_graph.x
        targets = pyg_graph.y

        # Get data splits.
        if name in Dataset.pyg_datasets_with_predefined_splits_names:
            # These datasets have 10 predefined data splits.
            num_splits = pyg_graph.train_mask.shape[1]
            train_idx_list = [torch.where(pyg_graph.train_mask[:, i])[0] for i in range(num_splits)]
            val_idx_list = [torch.where(pyg_graph.val_mask[:, i])[0] for i in range(num_splits)]
            test_idx_list = [torch.where(pyg_graph.test_mask[:, i])[0] for i in range(num_splits)]
        else:
            # 10 random stratified by class data splits will be created.
            train_idx_list, val_idx_list, test_idx_list = [], [], []
            for i in range(10):
                train_idx, val_and_test_idx = train_test_split(torch.arange(num_nodes), test_size=0.75, random_state=i,
                                                               stratify=targets)
                val_idx, test_idx = train_test_split(val_and_test_idx, test_size=0.66, random_state=i,
                                                     stratify=targets[val_and_test_idx])

                train_idx_list.append(train_idx.sort()[0])
                val_idx_list.append(val_idx.sort()[0])
                test_idx_list.append(test_idx.sort()[0])

        return dgl_graph, features, targets, train_idx_list, val_idx_list, test_idx_list

    @staticmethod
    def get_ogb_dataset(name):
        dataset = DglNodePropPredDataset(name, root='data')
        graph, targets = dataset[0]
        graph = graph.int()
        targets = targets.squeeze(1)
        features = graph.ndata['feat']
        del graph.ndata['feat']

        split_idx = dataset.get_idx_split()
        train_idx_list = [split_idx['train']]
        val_idx_list = [split_idx['valid']]
        test_idx_list = [split_idx['test']]

        return graph, features, targets, train_idx_list, val_idx_list, test_idx_list
