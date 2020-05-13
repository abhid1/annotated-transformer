#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date: 2018-12-02 20:54
import spacy
import torch
import torch.nn as nn
import numpy as np
from torchtext import data, datasets

from transformer.flow import make_model, batch_size_fn, run_epoch
from transformer.greedy import greedy_decode
from transformer.label_smoothing import LabelSmoothing
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.my_iterator import MyIterator, rebatch
from transformer.noam_opt import NoamOpt
from transformer.noam_opt import get_std_opt

# GPUs to use
devices = [0]  # Or use [0, 1] etc for multiple GPUs

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'


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
        out = greedy_decode(model, src, src_mask, max_len=MAX_LEN, start_symbol=TGT.vocab.stoi[BOS_WORD])
        # print('Translation:', end='\t')
        print('Out size (0)', out.size(0), 'Out size (1)', out.size(1))
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
                tgt.append(sym)
            print()

            translate.append(translate_str)
            tgt.append([tgt_str])

    print('Translate arr:', translate)
    print('Target arr:', tgt)


SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

if True:
    # Load spacy stuff
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    # Maximum Sentence Length
    MAX_LEN = 100

    # Load IWSLT Data ---> German to English Translation
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
                                                 vars(x)['trg']) <= MAX_LEN)
    # Frequency of words in the vocabulary
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    print("Size of source vocabulary:", len(SRC.vocab))
    print("Size of target vocabulary:", len(TGT.vocab))

    # Number of encoder & decoder blocks
    n = 6

    # Size of hidden dimension for all layers
    d_model = 512

    # Size of dimension for feed forward part
    ff_dim = 2048

    # Number of heads
    h = 8

    # Dropout
    dropout = 0.1

    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    model = make_model(len(SRC.vocab), len(TGT.vocab), n=n, d_model=d_model, d_ff=ff_dim, h=h, dropout=dropout)
    print("Model made with n:", n, "hidden_dim:", d_model, "feed forward dim:", ff_dim, "heads:", h,
          "dropout:", dropout)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    # UNCOMMENT WHEN RUNNING ON RESEARCH MACHINES - run on GPU
    # model.cuda()

    # Used by original authors, hurts perplexity but improves BLEU score
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

    # UNCOMMENT WHEN RUNNING ON RESEARCH MACHINES - run on GPU
    # criterion.cuda()

    BATCH_SIZE = 3000  # Was 12000, but I only have 12 GB RAM on my single GPU.

    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    #model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
    #                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # Use standard optimizer -- As used in the paper
    model_opt = get_std_opt(model)

    # Set EPOCH
    epoch_range = 1

    for epoch in range(epoch_range):
        print("=" * 80)
        print("Epoch ", epoch + 1)
        print("=" * 80)
        print("Training...")
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))

        print("Validation...")
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
        print(loss)
        run_validation_bleu_score(model, SRC, TGT, valid_iter)
else:
    model = torch.load('iwslt.pt')