import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy

from model import create_loss, create_metric, create_optimizer
from utils import pickle_load, pickle_save
from .ERM import ERM
from .Tent import Tent
from .T3A import T3A


def pic_loss(feats, prob):

    num_classes = prob.shape[1]
    means = (prob.T @ feats) / prob.sum(dim=0).view(num_classes, 1)  # weight average, c * k
    sq_dist = torch.square(torch.cdist(feats, means, p=2))  # num_nodes * c
    var_intra = (prob * sq_dist).sum()
    var_total = torch.sum(torch.square(feats - feats.mean(dim=0)))
    loss = var_intra / var_total

    return loss


class Matcha(ERM):

    def adapt_and_pred_single(self, model, X, E, Y, args, base_tta, mask=None):

        num_rounds = args.ada_rounds
        num_classes = args.out_channels

        model.eval()
        model.partial_freeze()

        self.featurizer = model.get_featurizer()
        self.classifier = model.get_classifier()

        optimizer = create_optimizer(model, args.ada_optimizer, args.ada_lr)
        eval_func = create_metric(args.eval_func)
        metrics = []
        losses = []

        for r in tqdm(range(num_rounds)):
            optimizer.zero_grad()

            # Call Base TTA algorithm to get pseudo-cluster
            model_for_basetta = deepcopy(model)
            tta_logits = base_tta.adapt_and_pred_single(model_for_basetta, X, E, args)
            base_tta.reset()
            prob = F.softmax(tta_logits, dim=1).detach()
            feats = self.featurizer(X, E)
            loss = pic_loss(feats, prob)

            loss.backward()
            optimizer.step()

            # Evaluation

            with torch.no_grad():
                if mask is not None:
                    tta_logits, Y = tta_logits[mask], Y[mask]

                metric = eval_func(tta_logits, Y).item()
                metrics.append(metric)

                if r % 5 == 0:
                    tqdm.write(f'Epoch {r:3d} \t Metric ({args.eval_func}): {metric:.4f}')

                losses.append(loss.item())

        return 0, metrics[-1]  # Alternative: use max(matrics) to see the best acc it ever got

    def adapt_and_test(self, model, datasets, args):

        model_cache = deepcopy(model.state_dict())

        if args.base_tta == 'erm':
            base_TTA = ERM(args)

        elif args.base_tta == 'tent':
            base_TTA = Tent(args)

        elif args.base_tta == 't3a':
            base_TTA = T3A(args)
        #
        # elif args.base_tta == 'adanpc':
        #     base_TTA = AdaNPC(args)
        #     base_TTA.training_y = self.training_y
        #     base_TTA.training_z = self.training_z
        #
        # elif args.base_tta == 'soga':
        #     base_TTA = SOGA(args)

        all_losses, all_metrics = [], []

        for dataset in datasets:

            mask = None

            if hasattr(dataset, 'graph'):  # NCDataset (Twitch and OGB-Arxiv)

                X = dataset.graph['node_feat'].to(self.device)
                E = dataset.graph['edge_index'].to(self.device)
                Y = dataset.label.to(self.device)

                if hasattr(dataset, 'test_mask'):  # OGB-Arxiv
                    mask = dataset.test_mask.to(self.device)

            else:  # CSBM, Syn-Cora and Syn-Products

                X = dataset[0].x.to(self.device)
                E = dataset[0].edge_index.to(self.device)
                Y = dataset[0].y.to(self.device)

                if hasattr(dataset[0], 'test_mask'):
                    mask = dataset[0].test_mask.to(self.device)

            loss, metric = self.adapt_and_pred_single(model, X, E, Y, args, base_TTA, mask)
            all_metrics.append(metric)

            model.load_state_dict(model_cache)

        return all_losses, all_metrics

