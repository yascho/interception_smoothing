import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv


def create_model(hparams):
    arch = hparams['arch']
    if arch == "GCN":
        return AblationGCN(hparams).to(hparams["device"])
    elif arch == "GAT":
        return AblationGAT(hparams).to(hparams["device"])
    elif arch == "GATv2":
        return AblationGATv2(hparams).to(hparams["device"])
    else:
        raise Exception("Not implemented")


class AblationLayer(nn.Module):
    def __init__(self, dim, p_train_nodes, p_eval_nodes,
                 p_train_edges, p_eval_edges):
        super().__init__()
        self.p_train_edges = p_train_edges
        self.p_eval_edges = p_eval_edges

        self.p_train_nodes = p_train_nodes
        self.p_eval_nodes = p_eval_nodes

        self.token = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.token.unsqueeze(0))

    def forward(self, x, edge_idx, edge_attr):
        # ablation of nodes
        x = x.clone()  # clone important in current implementation
        p = self.p_train_nodes if self.training else self.p_eval_nodes
        # i: idx of nodes to **ablate** (not kept)
        i = torch.bernoulli(torch.ones(x.shape[0]) * p).bool()
        x[i, :] = self.token.repeat((i.sum(), 1))  # ablate input

        # ablation of edges
        p = self.p_train_edges if self.training else self.p_eval_edges
        # i: idx of edges to **keep** (not ablated)
        i = torch.bernoulli(torch.ones(edge_idx.shape[1]) * (1-p)).bool()
        # clone important (edge_idx is reused)
        edge_idx = edge_idx.clone()[:, i]
        if edge_attr is not None:
            edge_attr = edge_attr.clone()[i]
        return x, edge_idx, edge_attr


class AblationGAT(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.conv1 = GATConv(hparams['in_channels'],
                             hparams['hidden_channels'],
                             heads=hparams['k_heads'],
                             edge_dim=1,
                             dropout=hparams['p_dropout'])
        self.conv2 = GATConv(hparams['k_heads']*hparams['hidden_channels'],
                             hparams['out_channels'],
                             edge_dim=1,
                             dropout=hparams['p_dropout'])
        self.p_dropout = hparams['p_dropout']
        self.p_dropout_skip = hparams['p_dropout_skip']

        self.ablate = AblationLayer(hparams['in_channels'],
                                    hparams['p_train_nodes'],
                                    hparams['p_eval_nodes'],
                                    hparams['p_train_edges'],
                                    hparams['p_eval_edges'])

        self.with_skip = hparams['with_skip']
        self.empty = torch.empty(2, 0).long().to(hparams["device"])

    def forward(self, x, edge_idx, edge_attr):

        skip = 0
        if self.with_skip:
            hidden = F.relu(self.conv1(x, self.empty))
            hidden = F.dropout(hidden, p=self.p_dropout_skip,
                               training=self.training)
            skip = self.conv2(hidden, self.empty)

        x, edge_idx, edge_attr = self.ablate(x, edge_idx, edge_attr)

        hidden = F.elu(self.conv1(x, edge_idx, edge_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx, edge_attr)

        return hidden + skip


class AblationGCN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.conv1 = GCNConv(hparams['in_channels'],
                             hparams['hidden_channels'])
        self.conv2 = GCNConv(
            hparams['hidden_channels'], hparams['out_channels'])

        self.p_dropout = hparams['p_dropout']
        self.p_dropout_skip = hparams['p_dropout_skip']

        self.ablate = AblationLayer(hparams['in_channels'],
                                    hparams['p_train_nodes'],
                                    hparams['p_eval_nodes'],
                                    hparams['p_train_edges'],
                                    hparams['p_eval_edges'])

        self.with_skip = hparams['with_skip']
        self.empty = torch.empty(2, 0).long().to(hparams["device"])

    def forward(self, x, edge_idx, edge_attr):

        skip = 0
        if self.with_skip:
            hidden = F.relu(self.conv1(x, self.empty))
            hidden = F.dropout(hidden, p=self.p_dropout_skip,
                               training=self.training)
            skip = self.conv2(hidden, self.empty)

        x, edge_idx, edge_attr = self.ablate(x, edge_idx, edge_attr)

        hidden = F.relu(self.conv1(x, edge_idx, edge_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx, edge_attr)

        return hidden + skip


class AblationGATv2(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.conv1 = GATv2Conv(hparams['in_channels'],
                               hparams['hidden_channels'],
                               heads=hparams['k_heads'],
                               edge_dim=1,
                               dropout=hparams['p_dropout'])
        self.conv2 = GATv2Conv(hparams['k_heads']*hparams['hidden_channels'],
                               hparams['out_channels'],
                               edge_dim=1,
                               dropout=hparams['p_dropout'])
        self.p_dropout = hparams['p_dropout']
        self.p_dropout_skip = hparams['p_dropout_skip']

        self.ablate = AblationLayer(hparams['in_channels'],
                                    hparams['p_train_nodes'],
                                    hparams['p_eval_nodes'],
                                    hparams['p_train_edges'],
                                    hparams['p_eval_edges'])

        self.with_skip = hparams['with_skip']
        self.empty = torch.empty(2, 0).long().to(hparams["device"])

    def forward(self, x, edge_idx, edge_attr):

        skip = 0
        if self.with_skip:
            hidden = F.relu(self.conv1(x, self.empty))
            hidden = F.dropout(hidden, p=self.p_dropout_skip,
                               training=self.training)
            skip = self.conv2(hidden, self.empty)

        x, edge_idx, edge_attr = self.ablate(x, edge_idx, edge_attr)

        hidden = F.elu(self.conv1(x, edge_idx, edge_attr))
        hidden = F.dropout(hidden, p=self.p_dropout, training=self.training)
        hidden = self.conv2(hidden, edge_idx, edge_attr)

        return hidden + skip
