import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from utils.my_logging import logger, init_logger
from utils.accuracy import convert_segeval_format, acc_computer
from summarizer.model_builder import Summarizer


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_id):
        self.sent_idx = sent_idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.clss_ids = clss_ids
        self.clss_mask = clss_mask
        self.label_id = label_id


def convert_examples_to_features(examples):
    """Loads a data file into a list of `InputBatch`s."""

    max_seq_length = max([len(e['src_idx']) for e in examples])
    max_cls_lenght = max([len(e['cls_ids']) for e in examples])

    features = []
    for (ex_index, example) in enumerate(examples):
        input_ids = example['src_idx']
        segment_ids = example['segments_ids']
        clss_ids = example['cls_ids']
        label_id = example['labels']

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
        label_id += clss_padding

        assert len(clss_ids) == max_cls_lenght
        assert len(clss_mask) == max_cls_lenght
        assert len(label_id) == max_cls_lenght

        features.append(
                InputFeatures(sent_idx=ex_index,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              clss_ids=clss_ids,
                              clss_mask=clss_mask,
                              label_id=label_id))
    return features


def get_dataset(features):
    """Pack the features into dataset"""
    all_sent_idx = torch.tensor([f.sent_idx for f in features], dtype=torch.int)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_clss_ids = torch.tensor([f.clss_ids for f in features], dtype=torch.long)
    all_clss_mask = torch.tensor([f.clss_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    
    return TensorDataset(all_sent_idx, all_input_ids, all_input_mask, all_segment_ids, all_clss_ids, all_clss_mask, all_label_ids)


def accuracy(out, labels, mask):
    def _mean(l):
        return float(sum(l)) / float(len(l))

    pred = [convert_segeval_format(x, m) for x, m in zip(out, mask)]
    gold = [convert_segeval_format(x, m) for x, m in zip(labels, mask)]
    assert len(pred) == len(gold)

    pk, windiff, bound_sim = zip(*[acc_computer(p, g) for p, g in zip(pred, gold)])
    
    return _mean(pk), _mean(windiff), _mean(bound_sim)


def select_seg(sent_idx, examples, scores, mask, spec='prob', crit=0.5):
    """Find the prediction and selected ids."""
    # check criterion
    if spec == 'prob' or spec == 'amount_prob':
        if crit > 1 or crit < 0:
            raise RuntimeError('Criterion error')
    elif spec == 'amount':
        if crit < 2:
            raise RuntimeError('Criterion error')
    else:
        raise RuntimeError('Criterion error')

    sent_scores = scores + mask
    selected_ids = np.argsort(-sent_scores, 1) # [batch size, clss length, 1]

    pred = []
    result = np.zeros_like(sent_scores)

    for i, idx in enumerate(selected_ids):
        _pred = []
        _idx = []

        # get example data
        example = examples[sent_idx[i]]
        src_str = example['src_txt']

        if spec == 'amount' and crit > len(src_str) * 0.5:
            raise RuntimeError('Criterion error, specific amount too much')

        for j in selected_ids[i][:len(src_str)]:
            if(j >= len(src_str)):
                continue
            src_str[j] = src_str[j] + ' <sep>'
            _idx.append(j)

            if spec == 'prob':
                if sent_scores[i][j] - 1 <= crit:
                    break
            elif spec == 'amount':
                if len(_idx) >= crit:
                    break
            elif spec == 'amount_prob':
                if len(_idx) >= len(src_str) * crit:
                    break

        _pred = ' '.join(src_str)

        pred.append(_pred)
        result[i, _idx] = 1
    
    return pred, result


def train(args):
    # initial device and gpu number
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("[Train] device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # get train example
    train_examples = None
    num_train_optimization_steps = None

    print('[Train] Load data...')
    train_examples = torch.load(os.path.join(args.data_path, 'train.pt'))
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    print('[Train] Finish loading data...')

    # Prepare model
    print('[Train] Initial model...')
    if args.bert_config:
        model_file = os.path.join(args.bert_config, WEIGHTS_NAME)
        config_file = os.path.join(args.bert_config, CONFIG_NAME)
        print('[Train] Load model...')
        config = BertConfig.from_json_file(config_file)
        model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
        # load check points
        model.load_cp(torch.load(model_file))
    else:
        model = Summarizer(args, device, load_pretrained_bert=True)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    # prepare training data
    train_features = convert_examples_to_features(train_examples)
    print("[Train] ***** Running training *****")
    print(f"[Train]   Num examples = {len(train_examples)}")
    print(f"[Train]   Batch size = {args.train_batch_size}")
    print(f"[Train]   Num steps = {num_train_optimization_steps}")
    train_data = get_dataset(train_features)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # prepare evaluation data
    eval_dataloader = None
    eval_path = os.path.join(args.data_path, 'dev.pt')
    if os.path.isfile(eval_path):
        print(f"[Train] ***** Prepare evaluation data from {eval_path} *****")

        eval_examples = torch.load(eval_path)
        eval_features = convert_examples_to_features(eval_examples)
        eval_data = get_dataset(eval_features)
        
        print(f"[Train]   Num examples = {len(eval_examples)}")
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    global_step = 0
    tr_loss = 0
    
    loss_f = torch.nn.BCELoss(reduction='none')
    best_performance = 1000000
    best_epoch = 0
    output_model_file = os.path.join(args.model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(args.model_path, CONFIG_NAME)
    # begin training
    for epoch_i in range(int(args.num_train_epochs)):
        print('[ Epoch', epoch_i, ']')
        
        # training
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_ids = batch

            sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask)

            loss = loss_f(sent_scores, label_ids.float())
            loss = (loss * mask.float()).sum()
            loss = loss/loss.numel()
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        
        train_loss = tr_loss / nb_tr_steps
        cur_loss = train_loss

        # evaluation
        if eval_dataloader and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            model.eval()

            eval_loss, eval_pk, eval_windiff, eval_bound_sim = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                clss_ids = clss_ids.to(device)
                clss_mask = clss_mask.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask)

                    tmp_eval_loss = loss_f(sent_scores, label_ids.float())
                    tmp_eval_loss = (tmp_eval_loss * mask.float()).sum()
                    tmp_eval_loss = tmp_eval_loss/tmp_eval_loss.numel()

                sent_scores = sent_scores.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                # find selected sentences
                _pred, selected_ids = select_seg(sent_idx, eval_examples, sent_scores, mask)
                tmp_pk, tmp_windiff, tmp_bound_sim = accuracy(selected_ids, label_ids, mask)

                eval_loss += tmp_eval_loss.mean().item()
                eval_pk += tmp_pk
                eval_windiff += tmp_windiff
                eval_bound_sim += tmp_bound_sim

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_pk = eval_pk / nb_eval_examples
            eval_windiff = eval_windiff / nb_eval_examples
            eval_bound_sim = eval_bound_sim / nb_eval_examples
            cur_loss = eval_loss
            
        # output result of an epoch
        print(f'  - (Training)   loss: {train_loss: 8.5f}')
        print(f'  - (Validation) loss: {eval_loss: 8.5f}, Pk: {100 * eval_pk: 3.3f} %, Pk: {100 * eval_windiff: 3.3f} %, Pk: {100 * eval_bound_sim: 3.3f} %')
        
        # record best model
        if cur_loss < best_performance:
            best_performance = cur_loss
            best_epoch = epoch_i
            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            print("[Train] Successfully save the best model.")


def test(args):
    # initial device and gpu number
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("[Test] device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_file = os.path.join(args.model_path, WEIGHTS_NAME)
    config_file = os.path.join(args.model_path, CONFIG_NAME)
    # Prepare model
    print('[Test] Load model...')
    config = BertConfig.from_json_file(config_file)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    if args.fp16:
        model.half()
    model.to(device)
    # load check points
    model.load_cp(torch.load(model_file))
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare testing data
    eval_dataloader = None
    eval_path = os.path.join(args.data_path, 'test.pt')
    print(f"[Test] ***** Prepare testing data from {eval_path} *****")

    eval_examples = torch.load(eval_path)
    eval_features = convert_examples_to_features(eval_examples)
    eval_data = get_dataset(eval_features)
    
    print(f"[Test]   Num examples = {len(eval_examples)}")
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # testing
    loss_f = torch.nn.BCELoss(reduction='none')
    refs, preds = [], []
    if eval_dataloader and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model.eval()

        eval_loss, eval_pk, eval_windiff, eval_bound_sim = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            clss_ids = clss_ids.to(device)
            clss_mask = clss_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask)

                tmp_eval_loss = loss_f(sent_scores, label_ids.float())
                tmp_eval_loss = (tmp_eval_loss * mask.float()).sum()
                tmp_eval_loss = tmp_eval_loss/tmp_eval_loss.numel()

            sent_scores = sent_scores.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            # find selected sentences
            _pred, selected_ids = select_seg(sent_idx, eval_examples, sent_scores, mask)
            preds.extend(_pred)
            tmp_pk, tmp_windiff, tmp_bound_sim = accuracy(selected_ids, label_ids, mask)

            eval_loss += tmp_eval_loss.mean().item()
            eval_pk += tmp_pk
            eval_windiff += tmp_windiff
            eval_bound_sim += tmp_bound_sim

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_pk = eval_pk / nb_eval_examples
        eval_windiff = eval_windiff / nb_eval_examples
        eval_bound_sim = eval_bound_sim / nb_eval_examples
            
        # print result
        print(f'[Test] loss: {eval_loss: 8.5f}, Pk: {100 * eval_pk: 3.3f} %, Pk: {100 * eval_windiff: 3.3f} %, Pk: {100 * eval_bound_sim: 3.3f} %')

    # output results
    path_dir = args.result_path
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, 'preds.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(preds))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='train', type=str, choices=['train', 'test'])
    parser.add_argument("--data_path", default='data/small')
    # parser.add_argument("--model_path", default='models/')
    parser.add_argument("--model_path", default='models/small/')
    parser.add_argument("--result_path", default='results/train/')

    parser.add_argument("--bert_model", default='bert-base-chinese', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_config", type=str,
                        help="Bert config path.")

    parser.add_argument('--log_file', default='')
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--param_init", default=0, type=float)
    parser.add_argument("--param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()
    init_logger(args.log_file)

    # initial remote debugger
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    # create result folder
    if os.path.exists(args.result_path) and os.listdir(args.result_path) and args.mode != 'train':
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.result_path))
    if not os.path.exists(args.result_path) and args.mode != 'train':
        os.makedirs(args.result_path)
    # create model folder
    if os.path.exists(args.model_path) and os.listdir(args.model_path) and args.mode == 'train':
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.model_path))
    if not os.path.exists(args.model_path) and args.mode == 'train':
        os.makedirs(args.model_path)

    # begin mode work
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)