from run import *
from load_dataset import *
import parameters as params


if __name__ == '__main__':
    train_tokens, train_labels = load_tabular_dataset(params.dataset_root)

    start_unsupervised_pad(train_tokens, train_labels)