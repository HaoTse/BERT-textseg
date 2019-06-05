import os
import argparse
import torch

from utils.my_logging import init_logger
from utils.get_example import Examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_path", default='data/small')
    parser.add_argument("--tgt_path", default='data/small')

    parser.add_argument('--log_file', default='')

    args = parser.parse_args()

    init_logger(args.log_file)

    example = Examples(high_granularity=True)
    
    print('[Preprocess] Processing data...')

    train_data = example.get_example(os.path.join(args.src_path, 'train'), dataset='train')
    dev_data = example.get_example(os.path.join(args.src_path, 'dev'), dataset='dev')
    test_data = example.get_example(os.path.join(args.src_path, 'test'), dataset='test')
    
    print(f'[Preprocess] Get training data with size {len(train_data)}')
    print(f'[Preprocess] Get developing data with size {len(dev_data)}')
    print(f'[Preprocess] Get testing data with size {len(test_data)}')

    torch.save(train_data, os.path.join(args.tgt_path, 'train.pt'))
    torch.save(dev_data, os.path.join(args.tgt_path, 'dev.pt'))
    torch.save(test_data, os.path.join(args.tgt_path, 'test.pt'))
