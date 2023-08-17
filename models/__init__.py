from .cnn import Net, simpleMLP
from .transformer import vision
from .adv_network import get_adv

def make_model(args):
    if args.model == 'transformer':
        return vision(image_size=args.image_size,
                        patch_size = args.patch_size,
                        num_classes = args.num_classes,
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads,
                        mlp_dim=args.mlp_dim,
                        attention_dropout=args.attention_dropout)
    

    elif args.model == 'adv':
        return get_adv(args)
    else:
        raise NotImplementedError(f'Unknown model: {args.model}')