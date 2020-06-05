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

import collections
import logging
import os
import unicodedata
from typing import List, Optional
from transformers import BertTokenizer
import json
from sklearn.metrics import confusion_matrix,classification_report,precision_recall_fscore_support,accuracy_score
import logging
logger = logging.getLogger(__name__)
id2label = {0:'right',1:'entailment',2:'contradiction'}
label2id={'right':0,'entailment':1,'contradiction':2}
class_names = ['right','entailment','contradiction']
class CrossExample(object):
    '''
    texta is a Chinese sentence
    textb is a English sentence
    label is an element in {0,1,2} where 0 means "right",1 means "entailment", 2 means "contradiction"
    '''
    def __init__(self,texta,textb,unique_id,label=None):
        self.texta = texta
        self.textb = textb
        self.label = label
        self.unique_id = unique_id
class InputFeature(object):

    def __init__(self, 
                 input_a_id,
                 input_b_id,
                 input_a_mask,
                 input_b_mask,
                 input_a_length,
                 input_b_length,
                 unique_id,
                 label):
        self.input_a_id = input_a_id
        self.input_b_id = input_b_id
        self.input_a_mask = input_a_mask
        self.input_b_mask = input_b_mask
        self.input_a_length = input_a_length
        self.input_b_length = input_b_length
        self.unique_id = unique_id
        self.label = label


def split_sentence(paragraph_text):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    doc_tokens = [] # context里面包含的一个个单词，按顺序放进去，等价于paragraph['context'].split()
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace=True
        else:
            if prev_is_whitespace: 
                doc_tokens.append(c) 
            else:
                doc_tokens[-1] += c #拼凑出整个单词
            prev_is_whitespace=False
    return doc_tokens


def read_cross_examples(input_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
    examples = []
    for entry in input_data:
        #print(entry)
        sentence1 = entry['sentence1']
        sentence2 = entry['sentence2']
        label = entry['label']
        unique_id = entry['unique_id']
        sentence1_tokens = split_sentence(sentence1)
        sentence2_tokens = split_sentence(sentence2)
        example = CrossExample(sentence1_tokens,sentence2_tokens,unique_id,label2id[label])
        examples.append(example)

    return examples


def convert_examples_to_features(examples,tokenizer_a, max_seq_length,tokenizer_b=None,
                                pad_token=0,is_double=True,mask_padding_with_zero=True):
    features = []
    max_length=0
    for example in examples:
        sentence1 = example.texta
        sentence2 = example.textb
        label = example.label
        unique_id = example.unique_id
        sentence1_token = []
        for token in sentence1:
            sub_tokens = tokenizer_a.tokenize(token)
            for sub_token in sub_tokens:
                sentence1_token.append(sub_token)
        sentence2_token = []
        for token in sentence2:
            if is_double:
                sub_tokens = tokenizer_b.tokenize(token)
            else:
                sub_tokens = tokenizer_a.tokenize(token)
            for sub_token in sub_tokens:
                sentence2_token.append(sub_token)
        #print(sentence1_token)
        input_a_id = tokenizer_a.convert_tokens_to_ids(sentence1_token)
        if is_double:
            input_b_id = tokenizer_b.convert_tokens_to_ids(sentence2_token)
        else:
            input_b_id = tokenizer_a.convert_tokens_to_ids(sentence2_token)
        input_a_mask = [1 if mask_padding_with_zero else 0] * len(input_a_id)
        input_b_mask = [1 if mask_padding_with_zero else 0] * len(input_b_id)
        input_a_length = len(input_a_id)
        input_b_length = len(input_b_id)
        while len(input_a_id) < max_seq_length:
            input_a_id.append(pad_token)
            input_a_mask.append(0 if mask_padding_with_zero else 1)

        while len(input_b_id) < max_seq_length:
            input_b_id.append(pad_token)
            input_b_mask.append(0 if mask_padding_with_zero else 1)
        if max_length<len(input_a_id):
            max_length = len(input_a_id)
        if max_length<len(input_b_id):
            max_length = len(input_b_id)
        #print(len(input_b_id),len(input_a_id))
        assert len(input_a_id) == max_seq_length
        assert len(input_a_mask) == max_seq_length
        assert len(input_b_id) == max_seq_length
        assert len(input_b_mask) == max_seq_length
        features.append(
                InputFeature(
                    input_a_id=input_a_id,
                    input_b_id=input_b_id,
                    input_a_mask=input_a_mask,
                    input_b_mask=input_b_mask,
                    input_a_length = input_a_length,
                    input_b_length = input_b_length,
                    unique_id = unique_id,
                    label=label))
    print(max_length)
    return features


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "predict"])

def write_predictions(all_examples, all_predicts,output_prediction_file):
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    ans = {}
    labels = []
    for index, example in enumerate(all_examples):
        unique_id = example.unique_id
        answer = id2label[all_predicts[index]]
        labels.append(example.label)
        ans[unique_id] = answer
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(ans, indent=4) + "\n")
    return labels

def eval_cross(labels,predictions):
    acc = accuracy_score(labels, predictions)
    confusion = classification_report(labels, predictions, target_names=class_names)
    return acc,confusion



