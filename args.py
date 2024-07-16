# arg.py
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description='Train a knife detection model.')
    parser.add_argument('--model_mode', type=str, default='tf_efficientnet_b0', help='Load a model from timm')
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--train_datacsv', type=str, default='dataset/train.csv', help='Load training data from csv file')
    parser.add_argument('--test_datacsv', type=str, default='dataset/test.csv', help='Load testing data from csv file')
    parser.add_argument('--saved_checkpoint_path', type=str, default=None, help='Path to the trained model file')
    parser.add_argument('--n_classes', type=int, default=192, help='Number of classes in the dataset')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--resized_img_weight', type=int, default=None, help='Resized image width')
    parser.add_argument('--resized_img_height', type=int, default=None, help='Resized image height')
    parser.add_argument('--evaluate-only', action='store_true', help='Load and evaluate the model directly without training')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the pre-trained model for evaluation')

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Input batch size for training')
        

    # ************************************************************
    # Data augmentation
    # ************************************************************ 
    parser.add_argument('--brightness', type=float, default=0.2, help='Brightness adjustment factor (0 means no change)')
    parser.add_argument('--contrast', type=float, default=0.2, help='Contrast adjustment factor (0 means no change)')
    parser.add_argument('--saturation', type=float, default=0.2, help='Saturation adjustment factor (0 means no change)')
    parser.add_argument('--hue', type=float, default=0.2, help='Hue adjustment factor (0 means no change)')
    parser.add_argument('--random_rotation', type=int, default=0, help='Max degree of random rotations')
    parser.add_argument('--vertical_flip', type=float, default=0, help='Probability of vertical flip')
    parser.add_argument('--horizontal_flip', type=float, default=0, help='Probability of horizontal flip')
    parser.add_argument(
        "--random-erase",
        action="store_true",
        help="use random erasing for data augmentation",
    )

    parser.add_argument(
        "--color-aug",
        action="store_true",
        help="randomly alter the intensities of RGB channels",
    )
    
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="optimization algorithm (see optimizers.py)",
    )
    parser.add_argument(
        "--learning_rate", default=0.0003, type=float, help="initial learning rate"
    )    
    parser.add_argument(
        "--weight-decay", default=5e-04, type=float, help="weight decay"
    )
    # sgd
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum factor for sgd and rmsprop",
    )
    parser.add_argument(
        "--sgd-dampening", default=0, type=float, help="sgd's dampening for momentum"
    )
    parser.add_argument(
        "--sgd-nesterov",
        action="store_true",
        help="whether to enable sgd's Nesterov momentum",
    )
    # rmsprop
    parser.add_argument(
        "--rmsprop-alpha", default=0.99, type=float, help="rmsprop's smoothing constant"
    )
    # adam/amsgrad
    parser.add_argument(
        "--adam-beta1",
        default=0.9,
        type=float,
        help="exponential decay rate for adam's first moment",
    )
    parser.add_argument(
        "--adam-beta2",
        default=0.999,
        type=float,
        help="exponential decay rate for adam's second moment",
    )
    
    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="multi_step",
        help="learning rate scheduler (see lr_schedulers.py)",
    )
    parser.add_argument(
        "--stepsize",
        default=[20, 40],
        nargs="+",
        type=int,
        help="stepsize to decay learning rate",
    )
    parser.add_argument("--gamma", default=0.1, type=float, help="learning rate decay")

    return parser

def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        "optim": parsed_args.optim,
        "lr": parsed_args.learning_rate,
        "weight_decay": parsed_args.weight_decay,
        "momentum": parsed_args.momentum,
        "sgd_dampening": parsed_args.sgd_dampening,
        "sgd_nesterov": parsed_args.sgd_nesterov,
        "rmsprop_alpha": parsed_args.rmsprop_alpha,
        "adam_beta1": parsed_args.adam_beta1,
        "adam_beta2": parsed_args.adam_beta2,
    }
    
def lr_scheduler_kwargs(parsed_args):
    """
    Build kwargs for lr_scheduler in lr_schedulers.py from
    the parsed command-line arguments.
    """
    return {
        "lr_scheduler": parsed_args.lr_scheduler,
        "stepsize": parsed_args.stepsize,
        "gamma": parsed_args.gamma,
    }
