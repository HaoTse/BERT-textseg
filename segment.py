import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME

from train import select_seg
from summarizer.model_builder import Summarizer
from utils.get_example import Examples
from utils.my_logging import logger, init_logger


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask):
        self.sent_idx = sent_idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.clss_ids = clss_ids
        self.clss_mask = clss_mask


def get_dataset(features):
    """Pack the features into dataset"""
    all_sent_idx = torch.tensor([f.sent_idx for f in features], dtype=torch.int)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_clss_ids = torch.tensor([f.clss_ids for f in features], dtype=torch.long)
    all_clss_mask = torch.tensor([f.clss_mask for f in features], dtype=torch.long)
    
    return TensorDataset(all_sent_idx, all_input_ids, all_input_mask, all_segment_ids, all_clss_ids, all_clss_mask)


def convert_examples_to_features(examples):
    """Loads a data file into a list of `InputBatch`s."""

    max_seq_length = max([len(e['src_idx']) for e in examples])
    max_cls_lenght = max([len(e['cls_ids']) for e in examples])

    features = []
    for (ex_index, example) in enumerate(examples):
        input_ids = example['src_idx']
        segment_ids = example['segments_ids']
        clss_ids = example['cls_ids']

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        clss_mask = [1] * len(clss_ids)
        clss_padding = [0] * (max_cls_lenght - len(clss_ids))
        clss_ids += clss_padding
        clss_mask += clss_padding

        assert len(clss_ids) == max_cls_lenght
        assert len(clss_mask) == max_cls_lenght

        features.append(
                InputFeatures(sent_idx=ex_index,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              clss_ids=clss_ids,
                              clss_mask=clss_mask))
    return features


def load_model(args, device):
    # initial necessary argument
    args.cache_dir = ''
    args.param_init = 0.0
    args.param_init_glorot = False
    args.bert_model = 'bert-base-chinese'

    # Prepare model
    model_file = os.path.join(args.model_path, WEIGHTS_NAME)
    config_file = os.path.join(args.model_path, CONFIG_NAME)
    print('[Segment] Load model...')
    config = BertConfig.from_json_file(config_file)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)

    model.to(device)
    # load check points
    model.load_cp(torch.load(model_file))
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model


def segment(texts, model, batch_size, device):
    # get dataloader
    example = Examples(high_granularity=True)
    seg_example = example.convert_to_example(texts)
    seg_features = convert_examples_to_features(seg_example)
    seg_data = get_dataset(seg_features)
    
    seg_sampler = SequentialSampler(seg_data)
    seg_dataloader = DataLoader(seg_data, sampler=seg_sampler, batch_size=batch_size)

    model.eval()

    # segment
    preds = []
    for sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask in tqdm(seg_dataloader, desc="Segment"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        clss_ids = clss_ids.to(device)
        clss_mask = clss_mask.to(device)

        with torch.no_grad():
            sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask)

        sent_scores = sent_scores.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        # select segment
        seg_num = 0.3
        # find selected sentences
        _pred, _selected_ids = select_seg(sent_idx, seg_example, sent_scores, mask, spec='amount_prob', crit=seg_num)
        preds.extend(_pred)

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default='models/small/')
    parser.add_argument("--eval_batch_size",
                        default=1,
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

    docu = '本文總結了十個可穿戴產品的設計原則，而這些原則，同樣也是筆者認爲是這個行業最吸引人的地方：1.爲人們解決重複性問題；2.從人開始，而不是從機器開始；3.要引起注意，但不要刻意；4.提升用戶能力，而不是取代人'
    # load model
    model = load_model(args, device)
    seg = segment([docu], model, args.eval_batch_size, device)

    print('[Input] ', docu)
    print('[Output] ', seg[0])
