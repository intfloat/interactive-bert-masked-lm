"""
Interactive masked language model with pre-trained BERT.

To download pre-trained BERT vocabulary and models, please run "bash ./download.sh"
"""

import os
import argparse

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument('--bert-model', type=str, default='bert_models/bert-base-uncased.tar.gz', help='path to bert model')
parser.add_argument('--bert-vocab', type=str, default='bert_models/bert-base-uncased-vocab.txt', help='path to bert vocabulary')
parser.add_argument('--topk', type=int, default=5, help='show top k predictions')

PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'


def to_bert_input(tokens, bert_tokenizer):
    token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
    sep_idx = tokens.index('[SEP]')
    segment_idx = token_idx * 0
    segment_idx[(sep_idx + 1):] = 1
    mask = (token_idx != 0)
    return token_idx.unsqueeze(0), segment_idx.unsqueeze(0), mask.unsqueeze(0)


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.exists(args.bert_model), '{} does not exist'.format(args.bert_model)
    assert os.path.exists(args.bert_vocab), '{} does not exist'.format(args.bert_vocab)
    assert args.topk > 0, '{} should be positive'.format(args.topk)

    print('Initialize BERT vocabulary from {}...'.format(args.bert_vocab))
    bert_tokenizer = BertTokenizer(vocab_file=args.bert_vocab)
    print('Initialize BERT model from {}...'.format(args.bert_model))
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model)

    while True:
        message = input('Enter your message: ').strip()
        tokens = bert_tokenizer.tokenize(message)
        if len(tokens) == 0:
            continue
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx, segment_idx, mask = to_bert_input(tokens, bert_tokenizer)
        with torch.no_grad():
            logits = bert_model(token_idx, segment_idx, mask, masked_lm_labels=None)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        mask_cnt = 0
        for idx, token in enumerate(tokens):
            if token == MASK:
                mask_cnt += 1
                print('Top {} predictions for {}th {}:'.format(args.topk, mask_cnt, MASK))
                topk_prob, topk_indices = torch.topk(probs[idx, :], args.topk)
                topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())
                for prob, tok in zip(topk_prob, topk_tokens):
                    print('{} {}'.format(tok, prob))
                print('='*80)
