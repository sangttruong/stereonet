from argparse import ArgumentParser, Namespace
import torch


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.
    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--task', type=str, default='regression',
                        help='Regression or classification task')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--split_path', type=str,
                        help='Path to .npy file containing train/val/test split indices')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model file to load for evaluation time')
    parser.add_argument('--eval_output_dir', type=str, default=None,
                        help='Directory to store outputs of evaluation, including predictions and attention visualization')


    # Training arguments
    parser.add_argument('--epoch', type=int, default=0,
                        help='Starting epoch')
    parser.add_argument('--checkpoint_load_path', type=str,
                        help='Path to model to load as checkpoint when resuming training')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Path to save checkpoint model')
    parser.add_argument('--viz_dir', type=str,
                        help='Path to save attention visualization')

    parser.add_argument('--n_epochs', type=int, default=60,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of workers to use in dataloader')
    parser.add_argument('--no_shuffle', action='store_true', default=False,
                        help='Whether or not to retain default ordering during training')
    parser.add_argument('--shuffle_pairs', action='store_true', default=False,
                        help='Whether or not to shuffle only pairs of stereoisomers')

    # Model arguments
    parser.add_argument('--gnn_type', type=str,
                        choices=['gin', 'gcn', 'dmpnn', 'orig_dmpnn'],
                        help='Type of gnn to use')
    parser.add_argument('--global_chiral_features', action='store_true', default=False,
                        help='Use global chiral atom features')
    parser.add_argument('--chiral_features', action='store_true', default=False,
                        help='Use local chiral atom features')
    parser.add_argument('--ft_boost', action='store_true', default=False, help='whether to concatenate R/S features after each MP layer')
    parser.add_argument('--hidden_size', type=int, default=32,
                        help='Dimensionality of hidden layers')
    parser.add_argument('--depth', type=int, default=2,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout probability')
    parser.add_argument('--graph_pool', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'attn', 'set2set'],
                        help='How to aggregate atom representations to molecule representation')
    parser.add_argument('--message', type=str, default='sum',
                        choices=['sum', 'tetra_pd', 'tetra_permute', 'tetra_permute_concat'],
                        help='How to pass neighbor messages')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of final FFN layers')
    parser.add_argument('--skip_coef', type=float, default=1.,
                        help='How much information retained in skip connections')

    # Attention arguments
    parser.add_argument('--attn_type', type=str, default='gat', choices=['gat', 'tang'],
                        help='Attention type. GAT or that used in Tang 2020')
    parser.add_argument('--gat_act', type=str, default='leakyrelu',
                        choices=['leakyrelu', 'relu'], help='Activation function used in GAT')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Alpha used in leakyReLU in GAT')
    parser.add_argument('--gat_depth', type=int, default=2,
                        help='number of GAT attention layers')
    parser.add_argument('--heads', type=int, default=3,
                        help='Number of attention heads')
    parser.add_argument('--attn_dropout', type=float, default=0.,
                        help='Dropout probability for attention')
    parser.add_argument('--concat', action='store_true', default=False,
                        help='concatenate heads or take average in multihead attention')

def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.
    :param args: Arguments.
    """
    if args.message.startswith('tetra'):
        setattr(args, 'tetra', True)
    else:
        setattr(args, 'tetra', False)

    # shuffle=False for custom sampler
    if args.shuffle_pairs:
        setattr(args, 'no_shuffle', True)

    setattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).
    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
