import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from model import create_model, create_optimizer, create_metric, create_loss


class ERM:

    def __init__(self, args):
        self.device = args.device

    def train(self, model, datasets, args):
        """
        Train the model
        """

        loss_func = create_loss(args.loss_func)
        eval_func = create_metric(args.eval_func)
        optimizer = create_optimizer(model=model, lr=args.tr_lr, optimizer_name=args.tr_optimizer)

        model.train()

        for epoch in tqdm(range(args.tr_rounds)):
            # in each epoch use a random permutation of graphs, if there are multiple

            random_idxs = torch.randperm(len(datasets))

            for idx in random_idxs:
                dataset = datasets[idx]

                mask = None

                if hasattr(dataset, 'graph'):  # NCDataset (Twitch and OGB-Arxiv)

                    X = dataset.graph['node_feat'].to(self.device)
                    E = dataset.graph['edge_index'].to(self.device)
                    Y = dataset.label.to(self.device)

                    # This should not be activated during training for Twitch and OGB-Arxiv
                    # if hasattr(dataset, 'train_mask'):
                    #     mask = dataset.train_mask.to(self.device)
                    # elif hasattr(dataset, 'mask'):
                    #     mask = dataset.mask.to(self.device)

                else:  # CSBM, Syn-Cora and Syn-Products

                    X = dataset[0].x.to(self.device)
                    E = dataset[0].edge_index.to(self.device)
                    Y = dataset[0].y.to(self.device)

                    if hasattr(dataset[0], 'train_mask'):  # Syn-Cora
                        mask = dataset[0].train_mask.to(self.device)
                    # elif hasattr(dataset[0], 'mask'):
                    #     mask = dataset[0].mask.to(self.device)

                optimizer.zero_grad()
                out = model(X, E)
                if mask is not None:
                    out, Y = out[mask], Y[mask]

                loss = loss_func(out, Y)
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

        return self._test(model, datasets, args)

    def _test(self, model, datasets, args):
        """
        Test the model across datasets
        """

        loss_func = create_loss(args.loss_func)
        eval_func = create_metric(args.eval_func)

        losses, metrics = [], []

        model.eval()

        for dataset in datasets:  # no need to shuffle

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

            with torch.no_grad():
                out = model(X, E)

            if mask is not None:
                out, Y = out[mask], Y[mask]

            losses.append(loss_func(out, Y).item())
            metrics.append(eval_func(out, Y).item())

        return losses, metrics

    def adapt_and_test(self, model, datasets, args):
        return self._test(model, datasets, args)

    def adapt_and_pred_single(self, model, X, E, args):
        return model(X, E)

    def reset(self):
        pass
