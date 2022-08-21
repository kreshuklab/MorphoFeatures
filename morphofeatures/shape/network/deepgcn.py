import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from .torch_nn import BasicConv
from .torch_vertex import GraphConv2d, ResDynBlock2d, DilatedKnnGraph


class DeepGCN(nn.Module):
    """
    Implementation adapted from:
    https://github.com/lightaime/deep_gcns_torch/blob/master/examples/modelnet_cls/architecture.py
    Credits to DeepGCNs.org
    """
    def __init__(self, act='relu',
                 bias=True,
                 in_channels=6,
                 channels=64,
                 out_channels=64,
                 dropout=0.0,
                 k=12,
                 norm='batch',
                 knn='matrix',
                 epsilon=0.2,
                 stochastic=True,
                 conv='edge',
                 emb_dims=1024,
                 n_blocks=14,
                 projection_head=True,
                 use_dilation=True):
        super().__init__()
        c_growth = channels
        self.n_blocks = n_blocks

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(
            in_channels, channels, conv, act, norm, bias=False
        )

        if use_dilation:
            self.backbone = Seq(
                *[ResDynBlock2d(channels, k, i+1, conv, act, norm, bias, stochastic, epsilon, knn)
                  for i in range(self.n_blocks - 1)]
            )
        else:
            self.backbone = Seq(
                *[ResDynBlock2d(channels, k, 1, conv, act, norm, bias, stochastic, epsilon, knn)
                  for _ in range(self.n_blocks - 1)]
            )
        fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = BasicConv(
            [fusion_dims, emb_dims], 'leakyrelu', norm, bias=False
        )
        self.prediction = Seq(
            *[BasicConv([emb_dims * 2, 512], 'leakyrelu', norm, drop=dropout),
              BasicConv([512, 256], 'leakyrelu', norm, drop=dropout),
              BasicConv([256, out_channels], None, None)]
        )
        if projection_head:
            self.projection_head = nn.Linear(out_channels, 64)
        else:
            self.projection_head = None
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, points, features):
        points = points.unsqueeze(-1)
        features = features.unsqueeze(-1)
        # feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        feats = [self.head(features, self.knn(points))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))

        feats = torch.cat(feats, dim=1)
        fusion = self.fusion_block(feats)
        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        out = self.prediction(torch.cat((x1, x2), dim=1)).squeeze(-1).squeeze(-1)
        if self.projection_head:
            return self.projection_head(out), out
        else:
            return out, out


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='runs/Configurations/xyz-normals-contrastive-DeepGCN-NewAugment-cells/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        net_params = yaml.load(f, Loader=yaml.FullLoader)['model']['kwargs']

    feats = torch.rand((2, 6, 1024, 1), dtype=torch.float)
    points = torch.rand((2, 3, 1024, 1), dtype=torch.float)   
    num_neighbors = 20

    print('Input size {}'.format(feats.size()))
    net = DeepGCN(net_params)
    out, h = net(points, feats)

    print('Output size {}'.format(out.size()))
