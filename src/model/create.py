import torch.nn.functional as F
import torch.optim as optim

from .GPRGNN import GPRGNN


def create_loss(name='ce'):
    """
    loss function must be differentiable
    """
    if name == 'ce':
        return F.cross_entropy
    elif name == 'ent':
        return lambda x, y: -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
    else:
        raise NotImplementedError('Unknown loss name: %s' % name)


def create_metric(name='acc'):
    """
    metric function can be any function with scalar output.
    """
    if name == 'acc':
        return lambda logits, target: logits.argmax(dim=1).view(-1).eq(target.view(-1)).float().mean()

    else:
        raise NotImplementedError('Unknown metric name: %s' % name)


def create_model(args):
    if args.gnn == 'gprgnn':
        # K, alpha, Init, Gamma=None, bias=True, **kwargs
        args.K, args.alpha, args.Init, args.Gamma = 9, 1.0, 'PPR', None
        args.hidden = 32
        args.ppnp = 'GPR_prop'
        args.dropout = 0.5
        args.dprate = 0.5
        model = GPRGNN(args)

    elif args.gnn == 'gprgnn-shallow':
        # K, alpha, Init, Gamma=None, bias=True, **kwargs
        args.K, args.alpha, args.Init, args.Gamma = 5, 1.0, 'PPR', None
        args.hidden = 8
        args.ppnp = 'GPR_prop'
        args.dropout = 0.5
        args.dprate = 0.5
        model = GPRGNN(args)

    elif args.gnn == 'gprgnn-wide':
        # K, alpha, Init, Gamma=None, bias=True, **kwargs
        args.K, args.alpha, args.Init, args.Gamma = 5, 1.0, 'PPR', None
        args.hidden = 128
        args.ppnp = 'GPR_prop'
        args.dropout = 0.5
        args.dprate = 0.5
        model = GPRGNN(args)

    else:
        raise ValueError('Unknown GNN model: %s' % args.gnn)

    model.to(args.device)

    return model


def create_optimizer(model, optimizer_name, lr):

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    else:
        raise NotImplementedError('Unknown optimizer. ')

    return optimizer
