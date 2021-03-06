#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date: 2018-12-02 20:54
import spacy
import time
import torch
import torch.nn as nn
import numpy as np
import distiller
import distiller.apputils as apputils
import math
import os
#import condensa
#from condensa.schemes import Compose, Prune, Quantize

from copy import deepcopy
from torchtext import data, datasets
from transformer.model import make_model
# from transformer.binary_model import make_model
from transformer.greedy import greedy_decode
from transformer.label_smoothing import LabelSmoothing
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.my_iterator import MyIterator, rebatch
from transformer.noam_opt import NoamOpt
from transformer.noam_opt import get_std_opt
from transformer.arguments import init_config
from transformer.metrics import evaluate_bleu
from distiller.quantization import PostTrainLinearQuantizer
from distiller.quantization import QuantAwareTrainRangeLinearQuantizer
from distiller.data_loggers import TensorBoardLogger, PythonLogger
from distiller.data_loggers import collect_quant_stats

# GPUs to use
devices = [0]  # Or use [0, 1] etc for multiple GPUs

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'

max_src_in_batch = 25000
max_tgt_in_batch = 25000

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


# def run_epoch(data_iter, model, loss_compute, args, SRC=None, TGT=None, valid_iter=None, is_valid=False):
def run_epoch(data_iter, model, loss_compute, args, epoch, steps_per_epoch, compression_scheduler=None, SRC=None, TGT=None, valid_iter=None, is_valid=False):
    """
    Standard Training and Logging Function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # IF PRUNING
        if compression_scheduler:
           compression_scheduler.on_minibatch_begin(epoch, minibatch_id=i, minibatches_per_epoch=steps_per_epoch)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

        # IF PRUNING
        loss = loss_compute(out, batch.trg_y, batch.ntokens, i, epoch, steps_per_epoch, compression_scheduler)
        #loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print('Iteration: %d Loss %f Tokens per Sec: %f' % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

        if i % args.valid_every == 1 and valid_iter is not None:
            model.eval()
            run_validation_bleu_score(model.module, SRC, TGT, valid_iter)

        if is_valid:
            run_validation_bleu_score(model.module, SRC, TGT, valid_iter)

        # IF PRUNING
        if compression_scheduler:
           compression_scheduler.on_minibatch_end(epoch, minibatch_id=i, minibatches_per_epoch=steps_per_epoch)

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
        out = greedy_decode(model, src, src_mask, max_len=100, start_symbol=TGT.vocab.stoi[BOS_WORD])
        for k in range(out.size(0)):
            translate_str = []
            for i in range(1, out.size(1)):
                sym = TGT.vocab.itos[out[k, i]]
                if sym == EOS_WORD:
                    break
                translate_str.append(sym)
            tgt_str = []
            for j in range(1, batch.trg.size(0)):
                sym = TGT.vocab.itos[batch.trg.data[j, k]]
                if sym == EOS_WORD:
                    break
                tgt_str.append(sym)

            translate.append(translate_str)
            tgt.append(tgt_str)

    # Essential for sacrebleu calculations
    translation_sentences = [" ".join(x) for x in translate]
    target_sentences = [" ".join(x) for x in tgt]
    bleu_validation = evaluate_bleu(translation_sentences, target_sentences)
    print('Validation BLEU Score', bleu_validation)
    return bleu_validation


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

    # model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
    #                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # Use standard optimizer -- As used in the paper
    model_opt = get_std_opt(model)

    # PRUNING CODE
    if args.summary:
        df = distiller.weights_sparsity_tbl_summary(model, False)
        print(df)
        exit(0)

    msglogger = apputils.config_pylogger('logging.conf', None)
    tflogger = TensorBoardLogger(msglogger.logdir)
    tflogger.log_gradients = True
    pylogger = PythonLogger(msglogger)

    source = args.compress

    if args.compress:
        compression_scheduler = distiller.config.file_config(model_par.module, None, args.compress)

    print(model_par.module)

    best_bleu = 0
    best_epoch = 0

    steps_per_epoch = math.ceil(len(train_iter.data()) / 60)

    for epoch in range(args.epoch):
        print("=" * 80)
        print("Epoch ", epoch + 1)
        print("=" * 80)
        print("Training...")
        model_par.train()

        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # IF PRUNING
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt), args, epoch,
                  steps_per_epoch, compression_scheduler, SRC, TGT, valid_iter, is_valid=False)

        # run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
        #           MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt), args,
        #           SRC, TGT, valid_iter, is_valid=False)

        print("Validation...")
        model_par.eval()

        # IF PRUNING
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None), args, epoch,
                         steps_per_epoch, compression_scheduler, SRC, TGT, valid_iter, is_valid=True)

        # loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
        #                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None), args,
        #                  SRC, TGT, valid_iter, is_valid=True)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch)

        print('Validation loss:', loss)
        print('Validation perplexity: ', np.exp(loss))
        bleu_score = run_validation_bleu_score(model, SRC, TGT, valid_iter)

        if best_bleu < bleu_score:
            best_bleu = bleu_score
            model_file = args.save_to + args.exp_name + 'validation.bin'
            print('Saving model without optimizer [%s]' % model_file)
            torch.save(model_par.module.state_dict(), model_file)
            best_epoch = epoch

        model_file = args.save_to + args.exp_name + 'latest.bin'
        print('Saving latest model without optimizer [%s]' % model_file)
        torch.save(model_par.module.state_dict(), model_file)

    print('The best epoch was:', best_epoch)


class SimpleLossCompute(object):
    """
    A simple loss compute and train function.
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


def test(args):
    # TODO: Add testing configurations
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
                                                    filter_pred=lambda x: len(
                                                        vars(x)['src']) <= args.max_length and len(
                                                        vars(x)['trg']) <= args.max_length)

    # Frequency of words in the vocabulary
    SRC.build_vocab(train.src, min_freq=args.min_freq)
    TGT.build_vocab(train.trg, min_freq=args.min_freq)

    print('Running test...')
    print("Size of source vocabulary:", len(SRC.vocab))
    print("Size of target vocabulary:", len(TGT.vocab))

    model = make_model(len(SRC.vocab), len(TGT.vocab), n=args.num_blocks, d_model=args.hidden_dim, d_ff=args.ff_dim,
                       h=args.num_heads, dropout=args.dropout)
    print("Model made with n:", args.num_blocks, "hidden_dim:", args.hidden_dim, "feed forward dim:", args.ff_dim,
          "heads:", args.num_heads, "dropout:", args.dropout)

    if args.load_model:
        print("Loading model from [%s]" % args.load_model)
        model.load_state_dict(torch.load(args.load_model))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    # feed_forward = []
    # attn = []
    # embed = []
    # sublayer = []
    # generator = []
    # for name, param in model.named_parameters():
    #     if name.__contains__("feed_forward"):
    #         feed_forward.append(np.prod(param.size()))
    #     if name.__contains__("attn"):
    #         attn.append(np.prod(param.size()))
    #     if name.__contains__("embed"):
    #         embed.append(np.prod(param.size()))
    #     if name.__contains__("sublayer"):
    #         sublayer.append(np.prod(param.size()))
    #     if name.__contains__("generator"):
    #         generator.append(np.prod(param.size()))

    feed_forward = []
    # attn = []
    # embed = []
    # sublayer = []
    # generator = []
    for name, param in model.named_parameters():
        if name.__contains__("embed") or name.__contains__("generator"):
            feed_forward.append(np.prod(param.size()))

    print("Num parameters:", np.sum(feed_forward))
    # print("Num parameters in original attn layer", np.sum(attn))
    # print("Num parameters in original embedding layer", np.sum(embed))
    # print("Num parameters in original sublayer", np.sum(sublayer))
    # print("Num parameters in original generator layer", np.sum(generator))

    # pad_idx = TGT.vocab.stoi[BLANK_WORD]
    # criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

    # UNCOMMENT WHEN RUNNING ON RESEARCH MACHINES - run on GPU
    model.cuda()

    test_iter = MyIterator(test, batch_size=args.batch_size, device=0, repeat=False,
                           sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    # ## Post-Linear Quantization Code
    # overrides_yaml = """
    # encoder.layers.*.self_attn.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # encoder.layers.*.feed_forward.*:
    #     bits_activations: 8
    #     bits_weights: 8
    #     bits_bias: 8
    # encoder.layers.*.sublayer.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # encoder.norm.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # decoder.layers.*.self_attn.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # decoder.layers.*.feed_forward.*:
    #     bits_activations: 8
    #     bits_weights: 8
    #     bits_bias: 8
    # decoder.layers.*.src_attn.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # decoder.layers.*.sublayer.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # decoder.norm.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # src_embed.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # tgt_embed.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # generator.*:
    #     bits_activations: null
    #     bits_weights: null
    #     bits_bias: null
    # """

    # CREATE STATS FILE
    # distiller.utils.assign_layer_fq_names(model)
    # stats_file = './acts_quantization_stats.yaml'
    #
    # if not os.path.isfile(stats_file):
    #     def eval_for_stats(model):
    #         valid_iter = MyIterator(val, batch_size=args.batch_size, device=0, repeat=False,
    #                                 sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn,
    #                                 train=False,
    #                                 sort=False)
    #         model.eval()
    #         run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
    #                          MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None), args,
    #                          SRC, TGT, valid_iter, is_valid=True)

        # collect_quant_stats(distiller.utils.make_non_parallel_copy(model), eval_for_stats, save_dir='.')

    # overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
    # quantizer = PostTrainLinearQuantizer(deepcopy(model), mode="ASYMMETRIC_UNSIGNED", overrides=overrides)

    # Post-Linear Quantization block
    # dummy_input = (torch.ones(130, 10).to(dtype=torch.long),
    #                torch.ones(130, 22).to(dtype=torch.long),
    #                torch.ones(130, 1, 10).to(dtype=torch.long),
    #                torch.ones(130, 22, 22).to(dtype=torch.long))
    # quantizer.prepare_model(dummy_input)
    # model = quantizer.model

    model.eval()
    print(model)

    translate = []
    tgt = []

    start_infer_time = time.time()

    for k, batch in enumerate(test_iter):
        src_orig = batch.src.transpose(0, 1).cuda()
        trg_orig = batch.trg.transpose(0, 1)
        for m in range(0, len(src_orig), 1):
            src = src_orig[m:(m + 1)].cuda()
            trg = trg_orig[m:(m + 1)]
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask,
                                max_len=100, start_symbol=TGT.vocab.stoi["<s>"])
            translate_str = []
            for i in range(0, out.size(0)):
                for j in range(1, out.size(1)):
                    sym = TGT.vocab.itos[out[i, j]]
                    if sym == "</s>":
                        break
                    translate_str.append(sym)
            tgt_str = []
            for i in range(trg.size(0)):
                for j in range(1, trg.size(1)):
                    sym = TGT.vocab.itos[trg[i, j]]
                    if sym == "</s>":
                        break
                    tgt_str.append(sym)

            translate.append(translate_str)
            tgt.append(tgt_str)

    print("Time for inference: ", time.time() - start_infer_time)

    # Essential for sacrebleu calculations
    translation_sentences = [" ".join(x) for x in translate]
    target_sentences = [" ".join(x) for x in tgt]

    bleu_validation = evaluate_bleu(translation_sentences, target_sentences)
    print('Test BLEU Score:', bleu_validation)

    # Save quantized model!
    # model_file = args.save_to + args.exp_name + '.bin'
    # print('Saving latest model without optimizer [%s]' % model_file)
    # torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    args = init_config()
    print(args)

    # Seed the RNG
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
