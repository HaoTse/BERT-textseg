import os
import random
import argparse

import numpy as np
import torch

from segment import load_model, segment
from utils.my_logging import logger, init_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='data/text.txt/')
    parser.add_argument("--output_file", default='result/seg.txt/')

    parser.add_argument("--model_path", default='models/small/')
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument('--log_file', default='')
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    init_logger(args.log_file)

    # initial device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print("[Segment] device: {} n_gpu: {}".format(device, args.n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    print("[Segment] Read input file...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        inputs = f.readlines()
    # remove space
    inputs = [i.replace(" ", "") for i in inputs]

    # load model
    model = load_model(args, device)
    segs = segment(inputs, model, args.eval_batch_size, device)
    print("[Segment] Finish segmentation, writing to output file...")

    # create output path
    dir_path = os.path.dirname(args.output_file)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(segs))
