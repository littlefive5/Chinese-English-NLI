# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function
import os
import argparse
import logging
import os
import random
import glob

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from modeling import ESIM, ESIM2
from transformers import BertTokenizer
import json
from utils import read_cross_examples, convert_examples_to_features, write_predictions, eval_cross
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
logger = logging.getLogger(__name__)


ChineseTokenizer =  BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
EnglishTokenizer =  BertTokenizer(vocab_file='bert-base-uncased-vocab.txt',tokenize_chinese_chars=False)
MultiTokenizer = BertTokenizer(vocab_file='bert-base-multilingual-uncased-vocab.txt')
MODEL_CLASSES = ['esim_single','esim_double']

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cached_examples(args,tokenizer_a, tokenizer_b=None,evaluate=False, output_examples=False,is_double=True):
    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_type.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_cross_examples(input_file=input_file)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer_a=tokenizer_a,
                                                tokenizer_b=tokenizer_b,
                                                max_seq_length=args.max_seq_length,
                                                is_double=is_double)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    
    # Convert to Tensors and build dataset
    all_input_a_ids = torch.tensor([f.input_a_id for f in features], dtype=torch.long)
    all_input_b_ids = torch.tensor([f.input_b_id for f in features], dtype=torch.long)
    all_input_a_mask = torch.tensor([f.input_a_mask for f in features], dtype=torch.long)
    all_input_b_mask = torch.tensor([f.input_b_mask for f in features], dtype=torch.long)
    all_input_a_length = torch.tensor([f.input_a_length for f in features], dtype=torch.long)
    all_input_b_length = torch.tensor([f.input_b_length for f in features], dtype=torch.long)    
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_a_ids, all_input_b_ids, all_input_a_mask,
                            all_input_b_mask, all_input_a_length,all_input_b_length,all_labels)
    if output_examples:
        return dataset, examples, features
    return dataset
def train(args, train_dataset, model, tokenizer_a,tokenizer_b):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_a_id':       batch[0],
                      'input_b_id':  batch[1],
                      'input_a_mask': batch[2],
                      'input_b_mask':   batch[3],
                      'input_a_length': batch[4],
                      'input_b_length': batch[5],
                      'label': batch[6]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        accuracy,confusion = evaluate(args, model, tokenizer_a,tokenizer_b)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel 
                    model_file = os.path.join(args.output_dir, "model.pth")
                    torch.save(model_to_save, model_file)
                    #model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer_a,tokenizer_b):
    dataset, examples, features = load_and_cached_examples(args, tokenizer_a, tokenizer_b, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_a_id':       batch[0],
                      'input_b_id':  batch[1],
                      'input_a_mask': batch[2],
                      'input_b_mask':   batch[3],
                      'input_a_length': batch[4],
                      'input_b_length': batch[5],
                      'label': batch[6]}
            outputs = model(**inputs)
            loss = outputs[0]
            predict = torch.argmax(outputs[1], -1)
            all_results.extend(to_list(predict.view(-1)))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    all_labels = write_predictions(examples,all_results,output_prediction_file)
    accuracy, confusion = eval_cross(all_labels,all_results)
    return accuracy,confusion

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--validate_file", default=None, type=str, required=True,
                        help="json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    ## Other parameters
    parser.add_argument("--max_seq_length", default=48, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--embeds_dim", default=384, type=int, 
                        help="the embedding size of ESIM ")
    parser.add_argument("--dropout", default=0.5, type=float, 
                        help="the dropout rate of ESIM ")
    parser.add_argument("--hidden_size", default=512, type=int, 
                        help="the hideen size of ESIM ")
    parser.add_argument("--linear_size", default=512, type=int, 
                        help="the linear size of ESIM ")
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    # Set seed
    set_seed(args)
    args.model_type = args.model_type.lower()
    if args.model_type == 'esim_single':
        tokenizer_a = ChineseTokenizer
        tokenizer_b = EnglishTokenizer
        model = ESIM(args,ChineseTokenizer.vocab_size,EnglishTokenizer.vocab_size)
        is_double = True
    elif args.model_type == 'esim_double':
        tokenizer_a = MultiTokenizer
        tokenizer_b = None
        model = ESIM2(args,MultiTokenizer.vocab_size)
        is_double = False

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cached_examples(args, tokenizer_a, tokenizer_b, evaluate=False, output_examples=False,is_double=is_double)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer_a, tokenizer_b)
        logger.info(" global_step = %s, average loss = %s ", global_step, tr_loss)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        #model_to_save.save(args.output_dir)
        model_file = os.path.join(args.output_dir, "model.pth")
        torch.save(model_to_save, model_file)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = torch.load(model_file)
        # model.to(args.device)

    if args.do_eval:
        #logger.info("Evaluate the following checkpoints: %s", checkpoints)
        model_file = os.path.join(args.output_dir, "model.pth")
        model = torch.load(model_file)
        model.to(args.device)
        accuracy,confusion = evaluate(args, model, tokenizer_a, tokenizer_b)
        print("accuracy:",accuracy)
        print(confusion)

if __name__ == "__main__":
    main()
