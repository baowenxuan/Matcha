import os
import argparse
import torch
import numpy as np
import random

from dataset import create_dataset
from model import create_model
from algo import create_algo
from utils import pickle_load, pickle_save


def main(args):
    datasets_tr, datasets_te = create_dataset(args)
    algo = create_algo(args)
    model = create_model(args)

    ######## ######## ######## ######## Train ######## ######## ######## ########

    # Load model and skip training...
    if args.load_model_path != 'none' and os.path.exists(args.load_model_path):
        state, tr_losses, tr_metrics = pickle_load(args.load_model_path, False)
        model.load_state_dict(state_dict=state)
        args.tr_rounds = 0  # skip training
        tr_losses, tr_metrics = algo.train(model=model, datasets=datasets_tr, args=args)
        print('Skip training, load model from %s' % args.load_model_path)

    # Train
    else:
        tr_losses, tr_metrics = algo.train(model=model, datasets=datasets_tr, args=args)

        # Save model
        if args.save_model_path != 'none':
            state = model.state_dict()
            pickle_save((state, tr_losses, tr_metrics), args.save_model_path, mode='wb')
            print('Save model to %s' % args.save_model_path)

    # Print training results
    tr_loss, tr_metric = np.mean(tr_losses), np.mean(tr_metrics)
    print(f'Train:\tLoss ({args.loss_func}): {tr_loss:.4f}\tMetric ({args.eval_func}): {tr_metric:.4f}')

    ######## ######## ######## ######## Test ######## ######## ######## ########

    te_losses, te_metrics = algo.adapt_and_test(model, datasets_te, args)

    te_metric = np.mean(te_metrics)
    print(f'Test: Metric ({args.eval_func}): {te_metric:.4f}')

    # Save adaptation metrics for analysis, visualization, etc.

    if args.save_metrics_path != 'none':
        obj = {
            'args': args,
            'metrics': te_metrics,
        }
        pickle_save(obj, args.save_metrics_path, mode='ab')


def args_parser():
    parser = argparse.ArgumentParser()

    ######## ######## ######## ######## Datasets ######## ######## ######## ########

    parser.add_argument('--dataset', type=str, default='cora')

    # CSBM

    parser.add_argument('--csbm_src_degree', type=int, default=5,
                        help='degree of CSBM dataset source graph')

    parser.add_argument('--csbm_tgt_degree', type=int, default=5,
                        help='degree of CSBM dataset source graph')

    parser.add_argument('--csbm_src_homophily', type=float, default=0.8,
                        help='homophily of CSBM dataset source graph')

    parser.add_argument('--csbm_tgt_homophily', type=float, default=0.8,
                        help='homophily of CSBM dataset source graph')

    parser.add_argument('--csbm_mu', type=float, default=0.03,
                        help='magnitude of mean in CSBM dataset')

    parser.add_argument('--csbm_delta_mu', type=float, default=0.00,
                        help='attribute shift in CSBM dataset')

    # Syn-Cora and Syn-Products

    parser.add_argument('--syn_src_homophily', type=float, default=0.8,
                        help='homophily of syn-cora or syn-products source')

    parser.add_argument('--syn_tgt_homophily', type=float, default=0.2,
                        help='homophily of syn-cora or syn-products target')

    parser.add_argument('--syn_seed', type=int, default=1,
                        help='syn dataset seed, 1, 2, 3')

    ######## ######## ######## ######## Setup ######## ######## ######## ########

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--algo', type=str, default='erm',
                        help='algorithm to train and adapt the model')

    parser.add_argument('--base_tta', type=str, default='erm',
                        help='base TTA algorithms used by Matcha')

    parser.add_argument('--gnn', type=str, default='gprgnn',
                        help='GNN model structure')

    ######## ######## ######## ######## Training ######## ######## ######## ########

    # Training

    parser.add_argument('--tr_lr', type=float, default=0.01,
                        help='learning rate for training')

    parser.add_argument('--tr_optimizer', type=str, default='adam',
                        help='optimizer for training')

    parser.add_argument('--tr_rounds', type=int, default=500,
                        help='rounds for training')

    # Adaptation

    parser.add_argument('--ada_lr', type=float, default=1,
                        help='learning rate for adaptation')

    parser.add_argument('--ada_optimizer', type=str, default='sgd',
                        help='optimizer for adaptation')

    parser.add_argument('--ada_rounds', type=int, default=100,
                        help='rounds for adaptation')

    # parser.add_argument('--adarc_temperature', type=float, default=1,
    #                     help='temperature for adarc when using the logits of base TTA')
    #
    # parser.add_argument('--adarc_loss', type=str, default='pic',
    #                     help='loss function used for AdaRC, [pic, entropy, pl, ]')

    ######## ######## ######## ######## Configs for Baselines ######## ######## ######## ########

    parser.add_argument('--t3a_filter_K', type=int, default=-1,
                        help='hyperparameter of T3A: number of feature to store for each class, -1 = store all, '
                             'could be slightly less than the number of samples for a class')

    parser.add_argument('--tent_lr', type=float, default=10.0,
                        help='learning rate for Tent')

    parser.add_argument('--tent_rounds', type=int, default=1,
                        help='rounds for Tent')

    parser.add_argument('--adanpc_beta', type=float, default=0.0,
                        help='min confidence to add to the queue')

    parser.add_argument('--adanpc_k', type=int, default=1000,
                        help='only the first k elements are considered')

    parser.add_argument('--adanpc_temperature', type=float, default=0.1,
                        help='tau for KNN-based softmax, smaller tau -> more one-hot prediction')

    parser.add_argument('--soga_lr', type=float, default=0.1,
                        help='learning rate for Tent')

    parser.add_argument('--soga_rounds', type=int, default=1,
                        help='rounds for Tent')

    parser.add_argument('--soga_struct_lambda', type=float, default=1.0,
                        help='learning rate for Tent')

    parser.add_argument('--soga_neigh_lambda', type=float, default=1.0,
                        help='rounds for Tent')

    # Torch config

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether to use cuda')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers of dataloader')

    # directories
    parser.add_argument('--data_dir', type=str, default='~/data/GraphTTA',
                        help='where the data is stored')

    parser.add_argument('--save_model_path', type=str, default='none',
                        help='save a trained model after training')

    parser.add_argument('--load_model_path', type=str, default='none',
                        help='skip training, load a trained model')

    parser.add_argument('--save_metrics_path', type=str, default='none',
                        help='save the results')

    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # You may remove these if you don't need deterministic result
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    main(args)
