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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        if args.task == 'jit':
            source_ids, tag, target_ids, _ = batch
        elif args.data_type == 's1' or args.data_type == 's2':
            source_ids, tag, target_ids = batch
        else:
            source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            elif args.data_type == 's1' or args.data_type == 's2':
                outputs = model(input_ids=source_ids, tag_ids=tag, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    pred_scores = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        if args.data_type == 's1' or args.data_type == 's2':
            tag = batch[1].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            
            elif args.data_type == 's1' or args.data_type == 's2':
                preds = model.generate(source_ids,
                                       tag_ids=tag,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       output_scores=True,
                                       return_dict_in_generate=True,
                                       max_length=args.max_target_length)
                top_preds = list(preds.sequences.cpu().numpy())
                scores = list(preds.sequences_scores.cpu().numpy())
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       output_scores=True,
                                       return_dict_in_generate=True,
                                       max_length=args.max_target_length)
                top_preds = list(preds.sequences.cpu().numpy())
                scores = list(preds.sequences_scores.cpu().numpy())
            pred_ids.extend(top_preds)
            pred_scores.extend(scores)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))
    score_fn = os.path.join(args.res_dir, "test_{}.score".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2, open(score_fn, 'w') as f3:
            for pred_nl, gold, score in zip(pred_nls, eval_examples, pred_scores):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                    f3.write(str(score) + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    if args.test_file is not None:
        args.test_filename = args.test_file
    
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                if args.task == 'jit':
                    source_ids, tag, target_ids, _ = batch
                elif args.data_type == 's1' or args.data_type == 's2':
                    source_ids, tag, target_ids = batch
                else:
                    source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                elif args.data_type == 's1' or args.data_type == 's2':
                    outputs = model(input_ids=source_ids, tag_ids=tag, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        fa.write(
                            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-bleu']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)
    
    if args.do_retrieval:
        model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin')))
        model.eval()

        train_tensor = None
        if os.path.exists(os.path.join(args.output_dir, 'train_tensor.pt')):
            logger.info("load train from cache")
            train_tensor = torch.load(os.path.join(args.output_dir, 'train_tensor.pt'))
        else:
            _, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train', only_src=True, is_sample=False)
            sampler = SequentialSampler(train_data)
            dataloader = DataLoader(train_data, sampler=sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
            for batch in tqdm(dataloader, total=len(dataloader)):
                source_ids = batch[0].to(args.device)
                tag = batch[1].to(args.device)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                with torch.no_grad():
                    preds = model.retrieval(source_ids, tag, source_mask)
                    # preds = torch.mean(preds['last_hidden_state'], dim=1) #[bs, dim]
                    preds = torch.max(preds['last_hidden_state'], dim=1).values
                    preds = F.normalize(preds, p=2, dim=1)
                    if train_tensor is not None:
                        train_tensor = torch.cat([train_tensor, preds], 0)
                    else:
                        train_tensor = preds
            torch.save(train_tensor, os.path.join(args.output_dir, 'train_tensor.pt'))
        logger.info("train tensor: {}".format(train_tensor.shape))

        if args.retrieval_file == "valid":
            eval_tensor = None
            if os.path.exists(os.path.join(args.output_dir, 'valid_tensor.pt')):
                logger.info("load valid from cache")
                eval_tensor = torch.load(os.path.join(args.output_dir, 'valid_tensor.pt'))
            else:
                _, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev', only_src=True, is_sample=False)
                sampler = SequentialSampler(eval_data)
                dataloader = DataLoader(eval_data, sampler=sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
                for batch in tqdm(dataloader, total=len(dataloader)):
                    source_ids = batch[0].to(args.device)
                    tag = batch[1].to(args.device)
                    source_mask = source_ids.ne(tokenizer.pad_token_id)
                    with torch.no_grad():
                        preds = model.retrieval(source_ids, tag, source_mask)
                        # preds = torch.mean(preds['last_hidden_state'], dim=1) #[bs, dim]
                        preds = torch.max(preds['last_hidden_state'], dim=1).values
                        preds = F.normalize(preds, p=2, dim=1)
                        if eval_tensor is not None:
                            eval_tensor = torch.cat([eval_tensor, preds], 0)
                        else:
                            eval_tensor = preds
                torch.save(eval_tensor, os.path.join(args.output_dir, 'valid_tensor.pt'))
            logger.info("valid tensor: {}".format(eval_tensor.shape))

        if args.retrieval_file == "test":
            test_tensor = None
            if os.path.exists(os.path.join(args.output_dir, 'test_tensor.pt')):
                logger.info("load test from cache")
                test_tensor = torch.load(os.path.join(args.output_dir, 'test_tensor.pt'))
            else:
                _, test_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test', only_src=True, is_sample=False)
                sampler = SequentialSampler(test_data)
                dataloader = DataLoader(test_data, sampler=sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
                for batch in tqdm(dataloader, total=len(dataloader)):
                    source_ids = batch[0].to(args.device)
                    tag = batch[1].to(args.device)
                    source_mask = source_ids.ne(tokenizer.pad_token_id)
                    with torch.no_grad():
                        preds = model.retrieval(source_ids, tag, source_mask)
                        # preds = torch.mean(preds['last_hidden_state'], dim=1) #[bs, dim]
                        preds = torch.max(preds['last_hidden_state'], dim=1).values #[bs, dim]
                        preds = F.normalize(preds, p=2, dim=1)
                        if test_tensor is not None:
                            test_tensor = torch.cat([test_tensor, preds], 0)
                        else:
                            test_tensor = preds
                torch.save(test_tensor, os.path.join(args.output_dir, 'test_tensor.pt'))
            logger.info("test tensor: {}".format(test_tensor.shape))

        train_diff, eval_diff, test_diff, train_msg = [], [], [], []
        with open(args.train_filename, 'r') as f:
            for line in f.readlines():
                train_diff.append(json.loads(line)["original_string"])
                train_msg.append(json.loads(line)["docstring"])
        with open(args.dev_filename, 'r') as f:
            for line in f.readlines():
                eval_diff.append(json.loads(line)["original_string"])
        with open(args.test_filename, 'r') as f:
            for line in f.readlines():
                test_diff.append(json.loads(line)["original_string"])

        n_best = 10
        if args.retrieval_file == "test":
            scores = torch.mm(test_tensor, train_tensor.T)
            scores = scores.cpu().numpy()
            with open(os.path.join(args.output_dir, 'test.output'), 'w') as f1, open(os.path.join(args.output_dir, 'test.score'), 'w') as f2:
                for i, score in tqdm(enumerate(scores), total=len(scores)):
                    sort_ids = np.argsort(score, axis=-1, kind='quicksort', order=None)[::-1]
                    # for j in range(len(sort_ids)):
                    #     if sort_ids[j] < 0.9:
                    #         break
                    idxs = sort_ids[:n_best]
                    bleu4 = []
                    candidate = test_diff[i].split()
                    for idx in idxs:
                        reference = [train_diff[idx].split()]
                        bleu4.append(sentence_bleu(reference, candidate))
                    idx = idxs[np.argmax(bleu4)]
                    f1.write(train_msg[idx].strip() + '\n')
                    f2.write('{}\n'.format(max(bleu4)))

        if args.retrieval_file == "valid":
            scores = torch.mm(eval_tensor, train_tensor.T)
            scores = scores.cpu().numpy()
            with open(os.path.join(args.output_dir, 'valid.output'), 'w') as f1, open(os.path.join(args.output_dir, 'valid.score'), 'w') as f2:
                for i, score in tqdm(enumerate(scores), total=len(scores)):
                    sort_ids = np.argsort(score, axis=-1, kind='quicksort', order=None)[::-1]
                    # for j in range(len(sort_ids)):
                    #     if sort_ids[j] < 0.9:
                    #         break
                    idxs = sort_ids[:n_best]
                    bleu4 = []
                    candidate = eval_diff[i].split()
                    for idx in idxs:
                        reference = [train_diff[idx].split()]
                        bleu4.append(sentence_bleu(reference, candidate))
                    idx = idxs[np.argmax(bleu4)]
                    f1.write(train_msg[idx].strip() + '\n')
                    f2.write('{}\n'.format(max(bleu4)))
    
    if args.do_jit:
        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 0
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, tag, target_ids, gold = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)
                pred = model(input_ids=source_ids, tag_ids=tag, attention_mask=source_mask,
                             tgt_ids=target_ids, tgt_mask=target_mask)
                
                loss = gold * torch.log(pred) + 0.1 * ((1 - gold) * torch.log(1 - pred))
                loss = torch.neg(torch.mean(loss))
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))


            # Eval model with dev dataset
            if 'dev_loss' in dev_dataset:
                eval_examples, eval_data = dev_dataset['dev_loss']
            else:
                eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                dev_dataset['dev_loss'] = eval_examples, eval_data
            
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                            num_workers=4, pin_memory=True)
            # Start evaluating model
            logger.info("  " + "***** Running ppl evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            total_loss = 0
            n_total = 0
            n_tp, n_fp, n_fn = 0, 0, 0
            n_pred, n_label = [], []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval"):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, tag, target_ids, gold = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                with torch.no_grad():
                    pred = model(input_ids=source_ids, tag_ids=tag, attention_mask=source_mask,
                                    tgt_ids=target_ids, tgt_mask=target_mask)
            
                    loss = gold * torch.log(pred) + 0.1 * ((1 - gold) * torch.log(1 - pred))
                    loss = torch.neg(torch.mean(loss))
                    zero = torch.zeros_like(pred)
                    one = torch.ones_like(pred)
                    hyp = torch.where(pred > 0.5, one, zero)
                    tp = (hyp * gold).sum().item()
                    fp = (hyp * (1 - gold)).sum().item()
                    fn = ((1 - hyp) * gold).sum().item()
                    total_loss += loss.item()
                    n_total += len(gold)
                    n_tp += tp
                    n_fp += fp
                    n_fn += fn
                    n_pred += pred.detach().cpu().numpy().tolist()
                    n_label += gold.detach().cpu().numpy().tolist()
            loss_per_word = total_loss/n_total
            eps = 1e-7
            p = n_tp / (n_tp + n_fp + eps)
            r = n_tp / (n_tp + n_fn + eps)
            f = 2 * p * r / (p + r + eps)
            auc = roc_auc_score(y_true=n_label, y_score=n_pred)
            print('  - (Validating)   ppl: {ppl: 8.5f}, f1: {f1:3.3f}, '\
                'precision: {precision:3.3f}, recall: {recall:3.3f}, auc: {auc:3.3f},'.format(
                    ppl=math.exp(min(loss_per_word, 100)), f1=100*f, precision=100*p, recall=100*r, auc=auc*100))
            
            if auc > best_ppl:
                not_loss_dec_cnt = 0
                logger.info("  Best auc:%s", auc)
                logger.info("  " + "*" * 20)
                fa.write("[%d] Best auc changed into %.4f\n" % (cur_epoch, auc))
                best_ppl = auc

                # Save best checkpoint for best auc
                output_dir = os.path.join(args.output_dir, 'checkpoint-best-auc')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if args.always_save_model:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best auc model into %s", output_model_file)

        logger.info("Finish training and take %s", get_elapse_time(t0))


    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
