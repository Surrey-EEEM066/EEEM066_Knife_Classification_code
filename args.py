# arg.py
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description='Train a knife detection model.')
    parser.add_argument('--model_mode', type=str, default='tf_efficientnet_b0', help='Load a model from timm')
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--train_datacsv', type=str, default='dataset/train.csv', help='Load training data from csv file')
    parser.add_argument('--test_datacsv', type=str, default='dataset/test.csv', help='Load testing data from csv file')
    parser.add_argument('--saved_checkpoint_path', type=str, default=None, help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=None, help='Input batch size for training')
    parser.add_argument('--n_classes', type=int, default=192, help='Number of classes in the dataset')
    parser.add_argument('--learning_rate', type=float, default=None, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--resized_img_weight', type=int, default=None, help='Resized image width')
    parser.add_argument('--resized_img_height', type=int, default=None, help='Resized image height')
    parser.add_argument('--evaluate-only', action='store_true', help='Load and evaluate the model directly without training')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the pre-trained model for evaluation')
    parser.add_argument('--brightness', type=float, default=0.2, help='Brightness adjustment factor (0 means no change)')
    parser.add_argument('--contrast', type=float, default=0, help='Contrast adjustment factor (0 means no change)')
    parser.add_argument('--saturation', type=float, default=0, help='Saturation adjustment factor (0 means no change)')
    parser.add_argument('--hue', type=float, default=0, help='Hue adjustment factor (0 means no change)')
    parser.add_argument('--random_rotation', type=int, default=180, help='Max degree of random rotations')
    parser.add_argument('--vertical_flip', type=float, default=0.5, help='Probability of vertical flip')
    parser.add_argument('--horizontal_flip', type=float, default=0.5, help='Probability of horizontal flip')
    
    return parser.parse_args()