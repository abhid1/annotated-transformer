#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date: 2018-12-02 20:54
import spacy
import time
import torch
import torch.nn as nn
import numpy as np
from torchtext import data, datasets

from transformer.model import make_model
from transformer.greedy import greedy_decode
from transformer.label_smoothing import LabelSmoothing
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.my_iterator import MyIterator, rebatch
from transformer.noam_opt import NoamOpt
from transformer.noam_opt import get_std_opt
from transformer.arguments import init_config

from transformer.metrics import evaluate_bleu

# GPUs to use
devices = [0]  # Or use [0, 1] etc for multiple GPUs

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'

max_src_in_batch = 25000
max_tgt_in_batch = 25000

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def run_epoch(data_iter, model, loss_compute, SRC=None, TGT=None, valid_iter=None):
    """
    Standard Training and Logging Function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print('Iteration: %d Loss %f Tokens per Sec: %f' % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

        # Validate every 150 iterations
        if i % 150 == 1:
            model.eval()
            run_validation_bleu_score(model.module, SRC, TGT, valid_iter)

    return total_loss / total_tokens


def batch_size_fn(new, count, size_so_far):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch
    global max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def run_validation_bleu_score(model, SRC, TGT, valid_iter):

    translate = []
    tgt = []

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1].cuda()
        src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi[BOS_WORD])
        # print('Translation:', end='\t')
        for k in range(out.size(0)):
            translate_str = []
            for i in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[k, i]]
                if sym == EOS_WORD:
                    break
                # print(sym, end=' ')
                translate_str.append(sym)
            # print()
            # print('Target:', end='\t')
            tgt_str = []
            for j in range(1, batch.trg.size(0)):
                sym = TGT.vocab.itos[batch.trg.data[j, k]]
                if sym == EOS_WORD:
                    break
                # print(sym, end=' ')
                tgt_str.append(sym)
            print()

            translate.append(translate_str)
            tgt.append([tgt_str])

    # TODO: Calculate BLEU Score
    # Essential for sacrebleu calculations
    translation_sentences = [" ".join(x) for x in translate]
    target_sentences = [" ".join(x) for x in tgt[0]]

    print(translation_sentences)
    print(target_sentences)

    return evaluate_bleu(translation_sentences, target_sentences)


def train(args):

    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD, lower=args.lower)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD, lower=args.lower)

    # Load IWSLT Data ---> German to English Translation
    if args.dataset == 'IWSLT':
        train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                                 filter_pred=lambda x: len(vars(x)['src']) <= args.max_length and len(
                                                     vars(x)['trg']) <= args.max_length)
    else:
        train, val, test = datasets.Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                                 filter_pred=lambda x: len(vars(x)['src']) <= args.max_length and len(
                                                     vars(x)['trg']) <= args.max_length)

    # Frequency of words in the vocabulary
    SRC.build_vocab(train.src, min_freq=args.min_freq)
    TGT.build_vocab(train.trg, min_freq=args.min_freq)

    print("Size of source vocabulary:", len(SRC.vocab))
    print("Size of target vocabulary:", len(TGT.vocab))

    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    model = make_model(len(SRC.vocab), len(TGT.vocab), n=args.num_blocks, d_model=args.hidden_dim, d_ff=args.ff_dim,
                       h=args.num_heads, dropout=args.dropout)
    print("Model made with n:", args.num_blocks, "hidden_dim:", args.hidden_dim, "feed forward dim:", args.ff_dim,
          "heads:", args.num_heads, "dropout:", args.dropout)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    if args.load_model:
        print("Loading model from [%s]" % args.load_model)
        model.load_state_dict(torch.load(args.load_model))

    # UNCOMMENT WHEN RUNNING ON RESEARCH MACHINES - run on GPU
    # model.cuda()

    # Used by original authors, hurts perplexity but improves BLEU score
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

    # UNCOMMENT WHEN RUNNING ON RESEARCH MACHINES - run on GPU
    # criterion.cuda()

    train_iter = MyIterator(train, batch_size=args.batch_size, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=args.batch_size, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False,
                            sort=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    #model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
    #                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # Use standard optimizer -- As used in the paper
    model_opt = get_std_opt(model)

    # TODO: Keep track of current loss, but instead do BLEU
    best_loss = 100000000

    for epoch in range(args.epoch):
        print("=" * 80)
        print("Epoch ", epoch + 1)
        print("=" * 80)
        print("Training...")
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt), SRC, TGT, valid_iter)

        print("Validation...")
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None), None, None, None)
        print(loss)
        run_validation_bleu_score(model, SRC, TGT, valid_iter)

        if best_loss > loss:
            best_loss = loss
            # TODO: Save when BLEU score is higher than the most highest
            model_file = args.save_to + args.exp_name + 'validation.bin'
            print('Saving model without optimizer [%s]' % model_file)
            torch.save(model_par.state_dict(), model_file)


if __name__ == '__main__':
    args = init_config()
    print(args)

    # Seed the RNG
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'train':
        train(args)
