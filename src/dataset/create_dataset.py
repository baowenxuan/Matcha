import torch

from .CSBM import CSBM
from .Synthetic import SynCora_Masked, SynProducts
from .NCDataset import load_twitch_dataset, load_ogb_arxiv

from .utils import match_edge_homophily


def create_dataset(args):
    if args.dataset == 'csbm':

        n = 5000
        p = 2000
        mu = args.csbm_mu
        delta_mu = args.csbm_delta_mu

        datasets_tr = [CSBM(root=args.data_dir, num_nodes=n, dim_feat=p,
                            mu1=mu, mu2=-mu,
                            d=args.csbm_src_degree, h=args.csbm_src_homophily), ]

        datasets_te = [CSBM(root=args.data_dir, num_nodes=n, dim_feat=p,
                            mu1=mu + delta_mu, mu2=-mu + delta_mu,
                            d=args.csbm_tgt_degree, h=args.csbm_tgt_homophily), ]

        args.in_channels = p
        args.out_channels = 2
        args.loss_func = 'ce'
        args.eval_func = 'acc'

    elif args.dataset == 'syncora-masked':

        datasets_tr = [SynCora_Masked(root=args.data_dir, homophily=args.syn_src_homophily, seed=args.syn_seed), ]
        datasets_te = [SynCora_Masked(root=args.data_dir, homophily=args.syn_tgt_homophily, seed=args.syn_seed), ]

        args.in_channels = 1433
        args.out_channels = 5
        args.loss_func = 'ce'
        args.eval_func = 'acc'

    elif args.dataset == 'synproducts':

        datasets_tr = [SynProducts(root=args.data_dir, homophily=args.syn_src_homophily, seed=args.syn_seed), ]
        datasets_te = [SynProducts(root=args.data_dir, homophily=args.syn_tgt_homophily, seed=args.syn_seed), ]

        args.in_channels = 100
        args.out_channels = 10
        args.loss_func = 'ce'
        args.eval_func = 'acc'

    elif args.dataset == 'twitch-homo2hetero':

        datasets_tr = [load_twitch_dataset(data_dir=args.data_dir, lang='DE'), ]
        datasets_te = [load_twitch_dataset(data_dir=args.data_dir, lang='ENGB'), ]

        target_edge_homo = 0.2

        for dataset in [datasets_te, ]:
            for graph in dataset:
                data, y = graph.graph, graph.label.view(-1)
                match_edge_homophily(data, y, target_edge_homo, random_seed=42, verbose=True)  # this is inplace

        torch.manual_seed(args.seed)  # set seed back.

        args.in_channels = datasets_tr[0].d
        args.out_channels = 2  # binary classification
        args.loss_func = 'ce'
        args.eval_func = 'acc'


    elif args.dataset == 'ogb-arxiv-homo2hetero':

        datasets_tr = [load_ogb_arxiv(data_dir=args.data_dir, year_bound=[1950, 2011], proportion=1.0), ]
        datasets_te = [load_ogb_arxiv(data_dir=args.data_dir, year_bound=[2014, 2020], proportion=1.0), ]

        target_edge_homo = 0.2

        for dataset in [datasets_te, ]:
            for graph in dataset:
                data, y = graph.graph, graph.label.view(-1)
                match_edge_homophily(data, y, target_edge_homo, random_seed=42, verbose=True)  # this is inplace

        torch.manual_seed(args.seed)  # set seed back.

        args.in_channels = datasets_tr[0].d
        args.out_channels = datasets_tr[0].c
        args.loss_func = 'ce'
        args.eval_func = 'acc'

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return datasets_tr, datasets_te
