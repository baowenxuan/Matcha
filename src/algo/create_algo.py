from .ERM import ERM
from .Matcha import Matcha


def create_algo(args):
    if args.algo == 'erm':
        algo = ERM(args)

    elif args.algo == 'matcha':
        algo = Matcha(args)

    else:
        raise ValueError('Unknown agent')

    return algo
