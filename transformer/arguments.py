import argparse


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--mode', choices=['train, test'], default='train', help='execution mode')
    parser.add_argument('--exp_name', default='IWSLTOriginal', type=str, help='name of the experiment')

    # Parameters for the model
    parser.add_argument('--batch_size', default=3000, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=512, type=int, help='size of hidden dimention for all layers')
    parser.add_argument('--num_blocks', default=6, type=int, help='number of blocks')
    parser.add_argument('--ff_dim', default=2048, type=int, help='size of dimention for feed forward part')
    parser.add_argument('--num_heads', default=8, type=int, help='number of multi-head attention heads')

    # Data
    parser.add_argument('--dataset', choices=['IWSLT, MULTI30K'], default='MULTI30k',
                        help='Choose either IWSLT or MULTI30K')
    parser.add_argument('--tokenize', default=True, type=bool, help='tokenize the dataset')
    parser.add_argument('--lower', default=True, type=bool, help='lowercase the dataset')
    parser.add_argument('--min_freq', type=int, default=2, help='min frequency')
    parser.add_argument('--max_length', type=int, default=100, help='max length of sentences')

    # Training parameters
    parser.add_argument('--epoch', default=20, type=int, help='maximum number of epochs')
    parser.add_argument('--load_model', default=None, type=str, help='path to the pre-trained model')
    # parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    # parser.add_argument('--lr_warm', default=0.0001, type=float, help='learning rate for warmed model')
    # parser.add_argument('--lr_decay', default=0.5, type=float,
    #                     help='decay learning rate if the validation performance drops')
    parser.add_argument('--save_to', default='saved_model/', type=str, help='save trained model to')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='set the dropout for the model')

    args = parser.parse_args()

    return args
