import os
import argparse
from dataset.CSBM import CSBM
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data', help='Root directory')
    args = parser.parse_args()

    root = os.path.expanduser(args.root)

    n = 5000
    p = 2000

    np.random.seed(0)

    # homophily shift

    mu = 0.03
    delta_mu = 0.00
    d = 5
    h = 0.8

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)

    mu = 0.03
    delta_mu = 0.00
    d = 5
    h = 0.2

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)

    # degree shift

    mu = 0.03
    delta_mu = 0.00
    d = 10
    h = 0.8

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)

    mu = 0.03
    delta_mu = 0.00
    d = 2
    h = 0.8

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)

    # homophily shift + attribute shift

    mu = 0.03
    delta_mu = 0.02
    d = 5
    h = 0.8

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)

    mu = 0.03
    delta_mu = 0.02
    d = 5
    h = 0.2

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)

    # degree shift + attribute shift

    mu = 0.03
    delta_mu = 0.02
    d = 10
    h = 0.8

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)

    mu = 0.03
    delta_mu = 0.02
    d = 2
    h = 0.8

    CSBM(root=root, num_nodes=n, dim_feat=p, mu1=mu + delta_mu, mu2=-mu + delta_mu,
         d=d, h=h)
