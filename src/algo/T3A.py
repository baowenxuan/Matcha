import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_loss, create_metric, create_optimizer

from .ERM import ERM


class T3A(ERM):

    def adapt_and_pred_single(self, model, X, E, args):

        # Hyperparameters

        self.filter_K = args.t3a_filter_K
        self.num_classes = args.out_channels

        model.eval()
        model.requires_grad_(False)

        self.featurizer = model.get_featurizer()
        self.classifier = model.get_classifier()

        # Warm Up
        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1),
                                                         num_classes=self.num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.softmax = torch.nn.Softmax(-1)

        # Adapt
        z = self.featurizer(X, E)
        p = self.classifier(z)
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()  # trim support set
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = supports.T @ labels
        logits = z @ torch.nn.functional.normalize(weights, dim=0)

        return logits

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(self.device)

        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s)))).to(self.device)
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
