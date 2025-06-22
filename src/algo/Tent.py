import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_loss, create_metric, create_optimizer

from .ERM import ERM


class Tent(ERM):

    def adapt_and_pred_single(self, model, X, E, args):

        model.eval()
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        optimizer = create_optimizer(model, 'sgd', args.tent_lr)

        for r in range(args.tent_rounds):
            optimizer.zero_grad()
            logits = model(X, E)
            x = logits
            loss = -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
            loss.backward()
            optimizer.step()

        logits = model(X, E)

        return logits
