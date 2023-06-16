# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 21:47
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: main_CPPIF_transformer.py

import math
import sys
import os
import torch
import config_CPPIF_transformer, data_loader_transformer, util_metric
import CPPIF_transformer
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# seed setting
# SEED = 1
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# flag is an indicator for checking whether this record has binding sites information
def boost_mask_BCE_loss(input_mask, flag):
    def conditional_BCE(y_true, y_pred, cri_nonReduce):
        seq_len = input_mask.shape[1]
        loss = flag.unsqueeze(-1).repeat(1, seq_len).view(-1) * cri_nonReduce(y_true, y_pred) * input_mask.view(-1)
        return torch.sum(loss) / torch.sum(input_mask)
    return conditional_BCE

def periodic_test(test_iter, model, criterion, cri_nonReduce, cri_nonReduce_weight, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    print('test current performance')
    if config.model_mode == 1:
        test_metric, test_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, cri_nonReduce, cri_nonReduce_weight, config)
        print('[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = test_metric.numpy()
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3], '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8], '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
        print('#' * 60 + 'Over' + '#' * 60)

        step_test_interval.append(sum_epoch)
        test_acc_record.append(test_metric[0])
        test_loss_record.append(test_loss)

        return test_metric, test_loss, test_repres_list, test_label_list
    else:
        test_metric, test_loss, avg_bi_loss, avg_pep_loss, avg_prot_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data, test_metric_site, test_roc_data_site, test_prc_data_site, \
        = model_eval(test_iter,
                        model,
                        criterion,
                        cri_nonReduce, cri_nonReduce_weight,
                        config)
        print(
            '[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\tAP,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = test_metric.numpy()
        AP_bi = test_prc_data[-1]
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % AP_bi, '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8],
              '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
        print('-' * 60 + 'Over' + '-' * 60)
        print(
            '[ACC2,\t\tPrecision2,\t\tSensitivity2,\tSpecificity2,\t\tF12,\t\tAUC2,\t\tAP2,\t\t\tMCC2,\t\t TP2,    \t\tFP2,\t\t\tTN2, \t\t\tFN2]')
        plmt2 = test_metric_site.numpy()
        AP_pep = test_prc_data_site[-1]
        print('%.5g\t\t' % plmt2[0], '%.5g\t\t' % plmt2[1], '%.5g\t\t' % plmt2[2], '%.5g\t\t' % plmt2[3],
              '%.5g\t' % plmt2[4],
              '%.5g\t\t' % plmt2[5], '%.5g\t\t' % AP_pep,'%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7], '  %.5g\t\t' % plmt2[8],
              '  %.5g\t\t' % plmt2[9], ' %.5g\t\t' % plmt2[10])
        print('-' * 60 + 'Over' + '-' * 60)
        # print(
        #     '[ACC3,\t\tPrecision3,\t\tSensitivity3,\tSpecificity3,\t\tF13,\t\tAUC3,\t\tAP3,\t\t\tMCC3,\t\t TP3,    \t\tFP3,\t\t\tTN3, \t\t\tFN3]')
        # plmt2 = test_metric_prot_site.numpy()
        # AP_prot = test_prc_data_prot_site[-1]
        # print('%.5g\t\t' % plmt2[0], '%.5g\t\t' % plmt2[1], '%.5g\t\t' % plmt2[2], '%.5g\t\t' % plmt2[3],
        #       '%.5g\t' % plmt2[4],
        #       '%.5g\t\t' % plmt2[5], '%.5g\t\t' % AP_prot, '%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7], '  %.5g\t\t' % plmt2[8],
        #       '  %.5g\t\t' % plmt2[9], ' %.5g\t\t' % plmt2[10])
        # print('#' * 60 + 'Over' + '#' * 60)

        # step_test_interval.append(sum_epoch)
        # test_acc_record.append(test_metric[0])
        # test_loss_record.append(test_loss)

        return test_metric, test_loss, avg_bi_loss, avg_pep_loss, avg_prot_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data, test_metric_site, test_roc_data_site, test_prc_data_site


def periodic_valid(valid_iter, model, criterion, cri_nonReduce, cri_nonReduce_weight, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)
    if config.model_mode == 1:
        valid_metric, valid_loss, valid_repres_list, valid_label_list, \
        valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, cri_nonReduce, cri_nonReduce_weight, config)

        print('validation current performance')
        print('[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = valid_metric.numpy()
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8],
              '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
        print('#' * 60 + 'Over' + '#' * 60)

        step_valid_interval.append(sum_epoch)
        valid_acc_record.append(valid_metric[0])
        valid_loss_record.append(valid_loss)

        return valid_metric, valid_loss, valid_repres_list, valid_label_list
    else:
        valid_metric, valid_loss, avg_bi_loss, avg_pep_loss, avg_prot_loss, valid_repres_list, valid_label_list, valid_roc_data, valid_prc_data, \
        valid_metric_site, valid_roc_data_site, valid_prc_data_site, \
        valid_metric_prot_site, valid_roc_data_prot_site, valid_prc_prot_data_site = model_eval(
            valid_iter, model, criterion, cri_nonReduce, cri_nonReduce_weight, config)
        print('validation current performance')
        print('[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\tAP,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = valid_metric.numpy()
        AP_bi = valid_prc_data[-1]
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % AP_bi,'%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8],
              '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
        print('-' * 60 + 'Over' + '-' * 60)
        print(
            '[ACC2,\t\tPrecision2,\t\tSensitivity2,\tSpecificity2,\t\tF12,\t\tAUC2,\t\tAP2,\t\t\tMCC2,\t\t TP2,    \t\tFP2,\t\t\tTN2, \t\t\tFN2]')
        plmt2 = valid_metric_site.numpy()
        AP_pep = valid_prc_data_site[-1]
        print('%.5g\t\t' % plmt2[0], '%.5g\t\t' % plmt2[1], '%.5g\t\t' % plmt2[2], '%.5g\t\t' % plmt2[3],
              '%.5g\t' % plmt2[4],
              '%.5g\t\t' % plmt2[5], '%.5g\t\t' % AP_pep,'%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7], '  %.5g\t\t' % plmt2[8],
              '  %.5g\t\t' % plmt2[9], ' %.5g\t\t' % plmt2[10])
        print('-' * 60 + 'Over' + '-' * 60)
        print(
            '[ACC3,\t\tPrecision3,\t\tSensitivity3,\tSpecificity3,\t\tF13,\t\tAUC3,\t\tAP3,\t\t\tMCC3,\t\t TP3,    \t\tFP3,\t\t\tTN3, \t\t\tFN3]')
        plmt2 = valid_metric_prot_site.numpy()
        AP_prot = valid_prc_prot_data_site[-1]
        print('%.5g\t\t' % plmt2[0], '%.5g\t\t' % plmt2[1], '%.5g\t\t' % plmt2[2], '%.5g\t\t' % plmt2[3],
              '%.5g\t' % plmt2[4],
              '%.5g\t\t' % plmt2[5], '%.5g\t\t' % AP_prot,'%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7], '  %.5g\t\t' % plmt2[8],
              '  %.5g\t\t' % plmt2[9], ' %.5g\t\t' % plmt2[10])
        print('#' * 60 + 'Over' + '#' * 60)

        # step_valid_interval.append(sum_epoch)
        # valid_acc_record.append(valid_metric[0])
        # valid_loss_record.append(valid_loss)

        return valid_metric, valid_loss, valid_repres_list, valid_label_list, valid_metric_site, valid_metric_prot_site


def train_model(train_iter, valid_iter, test_iter, model, optimizer, criterion, cri_nonReduce, cri_nonReduce_weight, config, iter_k):
    # writer = SummaryWriter()
    best_acc = 0
    best_performance = 0
    loss_print = 0
    loss_bi_print = 0
    loss_pep_print = 0
    loss_prot_print = 0
    device = torch.device("cuda" if config.cuda else "cpu")
    # binary prediction
    label_print = torch.empty([0], device=device)
    pred_prob_print = torch.empty([0], device=device)
    loss_init = None
    loss_last = None
    loss_last2 = None
    for epoch in range(1, config.epoch + 1):
        steps = 0
        repres_list = []
        label_list = []
        model.train()
        for batch in train_iter:
            steps += 1
            X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_seqs, prot_seqs, \
            X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs = batch
            # X_pep = X_pep.cpu()
            # X_p = X_p.cpu()
            # X_SS_pep = X_SS_pep.cpu()
            # X_SS_p = X_SS_p.cpu()
            # X_2_pep = X_2_pep.cpu()
            # X_2_p = X_2_p.cpu()
            # X_dense_pep = X_dense_pep.cpu()
            # X_dense_p = X_dense_p.cpu()
            # X_pep_mask = X_pep_mask.cpu()
            # X_bs_flag = X_bs_flag.cpu()
            # X_bs = X_bs.cpu()
            # labels = labels.cpu()
            # pred_binary_label = model.binary_predict(prot_seqs, pep_seqs)
            # pred_binary_label, pred_prot_site, pred_pep_site = model.predict(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, prot_seqs, pep_seqs, X_prot_mask, X_pep_mask)
            pred_binary_label, pred_pep_site = model(prot_seqs, pep_seqs, X_prot_mask, X_pep_mask)

            # binary interaction prediction loss
            labels = labels.view(-1)
            pred_label = pred_binary_label
            loss_bi = criterion(pred_label, labels)

            # peptide binding site prediction loss
            condition_CE = boost_mask_BCE_loss(X_pep_mask, X_bs_flag)
            pred_site_label = pred_pep_site
            pred_site_label = pred_site_label.view(-1, 2)
            X_bs = X_bs.view(-1)
            loss_pep_site = condition_CE(pred_site_label, X_bs, cri_nonReduce)

            # protein binding site prediction loss
            # condition_CE_prot = boost_mask_BCE_loss(X_prot_mask, X_prot_bs_flag)
            # pred_prot_site_label = pred_prot_site
            # pred_prot_site_label = pred_prot_site_label.view(-1, 2)
            # X_prot_bs = X_prot_bs.view(-1)
            # loss_prot_site = condition_CE_prot(pred_prot_site_label, X_prot_bs, cri_nonReduce_weight)

            # multi task learning
            # if loss_init is None:
            #     # loss_init = [loss_bi.detach(), loss_pep_site.detach(), loss_prot_site.detach()]
            #     loss_init = [1.0, 1.0, 1.0]
            #     # print("loss_init: ", loss_init)
            #     # loss_last2 = [loss_bi.detach(), loss_pep_site.detach(), loss_prot_site.detach()]
            #     loss_last2 = torch.tensor([1.0, 1.0, 1.0])
            #     loss_t = (loss_bi / loss_init[0]
            # 			  + loss_pep_site / loss_init[1]
            # 			  + loss_prot_site / loss_init[2]) / 3
            #
            # elif loss_last is None:
            #     # print("loss_bi1: ", loss_bi, " loss_pep_site1: ", loss_pep_site, " loss_prot_site1: ", loss_prot_site)
            #     # loss_last = [loss_bi.detach(), loss_pep_site.detach(), loss_prot_site.detach()]
            #     loss_last = torch.tensor([1.0, 1.0, 1.0])
            #     loss_t = (loss_bi / loss_init[0]
            # 			  + loss_pep_site / loss_init[1]
            # 			  + loss_prot_site / loss_init[2]) / 3
            # else:
            #     # print("loss_bi: ", loss_bi, " loss_pep_site: ", loss_pep_site, " loss_prot_site: ", loss_prot_site)
            #     if loss_last[0] != 0 and loss_last2[0] != 0:
            #         w0 = (loss_last[0] / loss_last2[0]).exp()
            #     else:
            #         w0 = math.exp(1.0)
            #     if loss_last[1] != 0 and loss_last2[1] != 0:
            #         w1 = (loss_last[1] / loss_last2[1]).exp()
            #     else:
            #         w1 = math.exp(1.0)
            #     if loss_last[2] != 0 and loss_last2[2] != 0:
            #         w2 = (loss_last[2] / loss_last2[2]).exp()
            #     else:
            #         w2 = math.exp(1.0)
            #     loss_t = loss_bi / loss_init[0] * w0 / (w0 + w1 + w2) \
            # 			 + loss_pep_site / loss_init[1] * w1 / (w0 + w1 + w2) \
            # 			 + loss_prot_site / loss_init[2] * w2 / (w0 + w1 + w2)
            #
            #
            # loss = loss_t
            # loss = loss_bi + loss_pep_site + loss_prot_site
            loss = loss_bi + loss_pep_site
            loss = loss / config.accum_steps
            loss.backward()

            loss_print += loss
            loss_bi_print += loss_bi / config.accum_steps
            loss_pep_print += loss_pep_site / config.accum_steps
            # loss_prot_print += loss_prot_site / config.accum_steps
            pred_prob_print = torch.cat([pred_prob_print, pred_label], dim=0)
            label_print = torch.cat([label_print, labels])

            # gradient accumulation
            if steps % config.accum_steps == 0 or steps == len(train_iter):

                # multi task learning
                # loss_last2 = loss_last
                # loss_last = [loss_bi_print.detach(), loss_pep_print.detach(), loss_prot_print.detach()]

                optimizer.step()
                optimizer.zero_grad()

                # tensorboard
                # writer.add_scalar('train_loss_bi', loss_bi_print.item(), (epoch-1)*(len(train_iter)//config.accum_steps + 1) + steps//config.accum_steps)
                # writer.add_scalar('train_loss_pep', loss_pep_print.item(), (epoch-1)*(len(train_iter)//config.accum_steps + 1) + steps//config.accum_steps)
                # writer.add_scalar('train_loss_prot', loss_prot_print.item(), (epoch-1)*(len(train_iter)//config.accum_steps + 1) + steps//config.accum_steps)

                '''Periodic Train Log'''
                if steps % config.interval_log == 0:
                    corre = (torch.max(pred_prob_print, 1)[1] == label_print).int()
                    corrects = corre.sum()
                    the_batch_size = len(label_print)
                    train_acc = 100.0 * corrects / the_batch_size
                    # sys.stdout.write(
                    #     '\rEpoch[{}] Batch[{}] - loss: {:.6f} loss_bi: {:.6f} loss_pep: {:.6f} loss_prot: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                    #                                                                         loss_print, loss_bi_print, loss_pep_print, loss_prot_print,
                    #                                                                         train_acc,
                    #                                                                         corrects,
                    #                                                                         the_batch_size))
                    sys.stdout.write(
                        '\rEpoch[{}] Batch[{}] - loss: {:.6f} loss_bi: {:.6f} loss_pep: {:.6f} | ACC: {:.4f}%({}/{})'.format(
                            epoch, steps,
                            loss_print, loss_bi_print, loss_pep_print,
                            train_acc,
                            corrects,
                            the_batch_size))
                    print()

                    # step_log_interval.append(steps)
                    # train_acc_record.append(train_acc)
                    # train_loss_record.append(loss)

                loss_print = 0
                loss_bi_print = 0
                loss_pep_print = 0
                # loss_prot_print = 0
                label_print = torch.empty([0], device=device)
                pred_prob_print = torch.empty([0], device=device)

                # '''Periodic Validation'''
                # if valid_iter and steps % config.interval_log == 0:
                #
                #     valid_metric, valid_loss, valid_repres_list, valid_label_list, valid_metric_site, valid_metric_prot_site = periodic_valid(
                #         valid_iter,
                #         model,
                #         criterion, cri_nonReduce, cri_nonReduce_weight,
                #         config,
                #         epoch)
                    # valid_acc = valid_metric[0]
                    # if valid_acc > best_acc:
                    #     best_acc = valid_acc
                    #     best_performance = valid_metric

                # '''Periodic Test'''
                # if test_iter and steps % config.interval_log == 0:
                #     test_metric, test_loss, test_repres_list, test_label_list, test_metric_site, test_metric_prot_site = periodic_test(
                #         test_iter,
                #         model,
                #         criterion, cri_nonReduce, cri_nonReduce_weight,
                #         config,
                #         epoch)

        # sum_epoch = iter_k * config.epoch + epoch
        sum_epoch = epoch

        '''Periodic Validation'''
        # if valid_iter and sum_epoch % config.interval_valid == 0:
        #
        #     valid_metric, valid_loss, valid_repres_list, valid_label_list, valid_metric_site, valid_metric_prot_site = periodic_valid(valid_iter,
        #                                                                                        model,
        #                                                                                        criterion, cri_nonReduce, cri_nonReduce_weight,
        #                                                                                        config,
        #                                                                                        sum_epoch)
        #     valid_acc = valid_metric[0]
        #     if valid_acc > best_acc:
        #         best_acc = valid_acc
        #         best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:
            test_metric, test_loss, avg_bi_loss, avg_pep_loss, avg_prot_loss, repres_list, label_list, test_roc_data, test_prc_data, test_metric_site, test_roc_data_site, test_prc_data_site, \
             = periodic_test(test_iter,
                              model,
                              criterion, cri_nonReduce, cri_nonReduce_weight,
                              config,
                              sum_epoch)

            test_auc = test_metric[5]
            if test_auc > best_acc:
                best_acc = test_auc
                best_performance = test_metric
                best_metric_site = test_metric_site
                best_prc_data = test_prc_data
                best_prc_data_site = test_prc_data_site
                if test_metric_site[5] > 0.715:
                    # torch.save({"test_roc_data": test_roc_data, "model": model.state_dict()}, f'Dataset1_AUC:{test_auc},MCC:{best_mcc}.pl')
                    torch.save(model.state_dict(), f'Trans_bi_AUC:{test_auc},pep_AUC:{test_metric_site[5]}.pl')

            # tensorboard
            AUC_bi = test_metric.numpy()[5]
            AP_bi = test_prc_data[-1]
            AUC_pep = test_metric_site.numpy()[5]
            AP_pep = test_prc_data_site[-1]
            # AUC_prot = test_metric_prot_site.numpy()[5]
            # AP_prot = test_prc_data_prot_site[-1]

            # writer.add_scalar('test_loss', test_loss.item(), sum_epoch)
            # writer.add_scalar('test_loss_bi', avg_bi_loss.item(), sum_epoch)
            # writer.add_scalar('test_loss_pep', avg_pep_loss.item(), sum_epoch)
            # writer.add_scalar('test_loss_prot', avg_prot_loss.item(), sum_epoch)
            # writer.add_scalar('test_AUC_bi', AUC_bi, sum_epoch)
            # writer.add_scalar('test_AUPRC_bi', AP_bi, sum_epoch)
            # writer.add_scalar('test_AUC_pep', AUC_pep, sum_epoch)
            # writer.add_scalar('test_AUPRC_pep', AP_pep, sum_epoch)
            # writer.add_scalar('test_AUC_prot', AUC_prot, sum_epoch)
            # writer.add_scalar('test_AUPRC_prot', AP_prot, sum_epoch)

            '''Periodic Save'''
            # save the model if specific conditions are met
            # test_acc = test_metric[0]
            # if test_acc > best_acc:
            #     best_acc = test_acc
            #     best_performance = test_metric
            #     if config.save_best and best_acc > config.threshold:
            #         save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)
            #
            # test_label_list = [x + 2 for x in test_label_list]
            # repres_list.extend(test_repres_list)
            # label_list.extend(test_label_list)

            '''feature dimension reduction'''
            # if sum_epoch % 1 == 0 or epoch == 1:
            #     dimension_reduction(repres_list, label_list, epoch)

            '''reduction feature visualization'''
            # if sum_epoch % 5 == 0 or epoch == 1 or (epoch % 2 == 0 and epoch <= 10):
            #     penultimate_feature_visulization(repres_list, label_list, epoch)
            #
            # time_test_end = time.time()
            # print('inference time:', time_test_end - time_test_start, 'seconds')

    return best_performance, best_prc_data, best_metric_site, best_prc_data_site

def model_eval(data_iter, model, criterion, cri_nonReduce, cri_nonReduce_weight, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    # binary prediction
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    # peptide binding site prediction
    label_pred_site = torch.empty([0], device=device)
    label_real_site = torch.empty([0], device=device)
    pred_prob_site = torch.empty([0], device=device)
    # protein binding site prediction
    label_pred_prot_site = torch.empty([0], device=device)
    label_real_prot_site = torch.empty([0], device=device)
    pred_prob_prot_site = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, iter_size_site, corrects_site, avg_loss, avg_bi_loss, avg_prot_loss, avg_pep_loss, iter_size_prot_site, corrects_prot_site = 0, 0, 0, 0, 0, 0, 0,0,0,0
    repres_list = []
    label_list = []
    AUC_pep_list = []
    MCC_pep_list = []
    pad_pep_len = config.pad_pep_len
    pad_seq_len = config.pad_prot_len
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            # peptide binding site prediction
            pep_pred_site = torch.empty([0], device=device)
            pep_real_site = torch.empty([0], device=device)
            pep_prob_site = torch.empty([0], device=device)
            if config.model_mode == 1:
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, labels = batch
                pred_pos_label = model.binary_forward(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p)
                labels = labels.view(-1)
                pred_label = torch.cat([1-pred_pos_label, pred_pos_label], 1)
                loss = criterion(pred_label, labels)
            else:
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_seqs, prot_seqs, \
                X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs = batch
                # X_pep = X_pep.cpu()
                # X_p = X_p.cpu()
                # X_SS_pep = X_SS_pep.cpu()
                # X_SS_p = X_SS_p.cpu()
                # X_2_pep = X_2_pep.cpu()
                # X_2_p = X_2_p.cpu()
                # X_dense_pep = X_dense_pep.cpu()
                # X_dense_p = X_dense_p.cpu()
                # X_pep_mask = X_pep_mask.cpu()
                # X_bs_flag = X_bs_flag.cpu()
                # X_bs = X_bs.cpu()
                # labels = labels.cpu()
                # pred_binary_label = model.binary_predict(prot_seqs, pep_seqs)
                # pred_binary_label, pred_prot_site, pred_pep_site = model.predict(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p,
                #                                                    X_dense_pep, X_dense_p, prot_seqs, pep_seqs, X_prot_mask, X_pep_mask)
                pred_binary_label, pred_pep_site = model(prot_seqs, pep_seqs, X_prot_mask, X_pep_mask)

                # binary interaction prediction loss
                labels = labels.view(-1)
                pred_label = pred_binary_label
                loss_bi = criterion(pred_label, labels)

                # peptide binding site prediction loss
                condition_CE = boost_mask_BCE_loss(X_pep_mask, X_bs_flag)
                pred_site_label = pred_pep_site
                pred_site_label = pred_site_label.view(-1, 2)
                X_bs = X_bs.view(-1)
                loss_pep_site = condition_CE(pred_site_label, X_bs, cri_nonReduce)

                # protein binding site prediction loss
                # condition_CE_prot = boost_mask_BCE_loss(X_prot_mask, X_prot_bs_flag)
                # pred_prot_site_label = pred_prot_site
                # pred_prot_site_label = pred_prot_site_label.view(-1, 2)
                # X_prot_bs = X_prot_bs.view(-1)
                # loss_prot_site = condition_CE_prot(pred_prot_site_label, X_prot_bs, cri_nonReduce_weight)

                # the final loss function contains three parts
                # loss = loss_bi + loss_pep_site + loss_prot_site
                loss = loss_bi + loss_pep_site

                # peptide binding site prediction evaluation
                pred_prob_all = F.softmax(pred_site_label, dim=1)
                pred_pos_site_label = pred_prob_all[:, 1]
                p_class_site = torch.max(pred_prob_all, 1)[1]
                for i, f in enumerate(X_pep_mask.view(-1)):
                    if X_bs_flag[i//pad_pep_len] == 1 and f == 1:
                        corre_site = (p_class_site[i] == X_bs[i]).int()
                        corrects_site += corre_site.sum()
                        iter_size_site += 1

                        pep_pred_site = torch.cat([pep_pred_site, p_class_site[i].view(-1).float()])
                        pep_real_site = torch.cat([pep_real_site, X_bs[i].view(-1).float()])
                        pep_prob_site = torch.cat([pep_prob_site, pred_pos_site_label.view(-1)[i].view(-1)])

                        label_pred_site = torch.cat([label_pred_site, p_class_site[i].view(-1).float()])
                        label_real_site = torch.cat([label_real_site, X_bs[i].view(-1).float()])
                        pred_prob_site = torch.cat([pred_prob_site, pred_pos_site_label.view(-1)[i].view(-1)])

                # calculate each peptide's AUC and MCC
                if X_bs_flag[0] == 1:
                    metric_pep, roc_data_pep, prc_data_pep = util_metric.caculate_metric(pep_pred_site,
                                                                                         pep_real_site,
                                                                                         pep_prob_site)
                    auc_pep = metric_pep[5].cpu().detach().numpy()
                    mcc_pep = metric_pep[6].cpu().detach().numpy()
                else:
                    auc_pep = -5
                    mcc_pep = -5
                # print('AUC', auc_pep)
                # print('MCC', mcc_pep)
                AUC_pep_list.append(auc_pep)
                MCC_pep_list.append(mcc_pep)

                # protein binding site prediction evaluation
                # pred_prot_prob_all = F.softmax(pred_prot_site_label, dim=1)
                # pred_pos_site_label = pred_prot_prob_all[:, 1]
                # p_class_site = torch.max(pred_prot_prob_all, 1)[1]
                # for i, f in enumerate(X_prot_mask.view(-1)):
                #     if X_prot_bs_flag[i // pad_seq_len] == 1 and f == 1:
                #         corre_site = (p_class_site[i] == X_prot_bs[i]).int()
                #         corrects_prot_site += corre_site.sum()
                #         iter_size_prot_site += 1
                #
                #         label_pred_prot_site = torch.cat([label_pred_prot_site, p_class_site[i].view(-1).float()])
                #         label_real_prot_site = torch.cat([label_real_prot_site, X_prot_bs[i].view(-1).float()])
                #         pred_prob_prot_site = torch.cat([pred_prob_prot_site, pred_pos_site_label.view(-1)[i].view(-1)])

            loss = loss.float()
            avg_loss += loss
            avg_bi_loss += loss_bi
            # avg_prot_loss += loss_prot_site
            avg_pep_loss += loss_pep_site

            # binary interaction prediction evaluation
            pred_prob_all = F.softmax(pred_label, dim=1)
            pred_pos_label = pred_prob_all[:, 1]
            p_class = torch.max(pred_prob_all, 1)[1]
            corre = (p_class == labels).int()
            corrects += corre.sum()
            iter_size += labels.size(0)
            label_pred = torch.cat([label_pred, p_class.float()])
            label_real = torch.cat([label_real, labels.float()])
            pred_prob = torch.cat([pred_prob, pred_pos_label.view(-1)])


    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= len(data_iter)
    avg_bi_loss /= len(data_iter)
    avg_prot_loss /= len(data_iter)
    avg_pep_loss /= len(data_iter)
    # accuracy = 100.0 * corrects / iter_size
    accuracy = metric[0]
    if config.model_mode == 1 :
        print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                    100*accuracy,
                                                                    corrects,
                                                                    iter_size))
        return metric, avg_loss, repres_list, label_list, roc_data, prc_data
    else:
        metric_site, roc_data_site, prc_data_site = util_metric.caculate_metric(label_pred_site, label_real_site,
                                                                              pred_prob_site)
        accuracy_site = metric_site[0]

        # save each peptide's AUC, MCC
        torch.save(AUC_pep_list, 'trans_each_pep_AUC_list')
        torch.save(MCC_pep_list, 'trans_each_pep_MCC_list')

        # metric_prot_site, roc_data_prot_site, prc_data_prot_site = util_metric.caculate_metric(label_pred_prot_site, label_real_prot_site,
        #                                                                         pred_prob_prot_site)
        # accuracy_prot_site = metric_prot_site[0]
        print('Evaluation - loss: {:.6f}  loss_bi: {:.6f}  loss_pep: {:.6f}  bi_ACC: {:.4f}%({}/{})  peptide site ACC: {:.4f}%({}/{})'.format(avg_loss, avg_bi_loss, avg_pep_loss,
                                                                      100 * accuracy,
                                                                      corrects,
                                                                      iter_size,
                                                                       100 * accuracy_site,
                                                                       corrects_site,
                                                                       iter_size_site,
                                                                        # 100 * accuracy_prot_site,
                                                                        # corrects_prot_site,
                                                                        # iter_size_prot_site
                                                                       ))
        return metric, avg_loss, avg_bi_loss, avg_pep_loss, avg_prot_loss, repres_list, label_list, roc_data, prc_data, metric_site, roc_data_site, prc_data_site

# warm up
class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                    self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False

def k_fold_CV(train_loader_list, valid_loader_list, test_loader_list, config, k):
    valid_performance_list = []

    iter_k = 0
    print('=' * 50, 'iter_k={}'.format(k + 1), '=' * 50)

    # Cross validation on training set
    train_iter = train_loader_list[iter_k]
    valid_iter = valid_loader_list[iter_k]
    test_iter = test_loader_list[iter_k]

    print('len(train_iter)', len(train_iter))
    print('len(valid_iter)', len(valid_iter))
    print('len(test_iter)', len(test_iter))
    print('----------Data Loader Over----------')

    model = CPPIF_transformer.Transformer(config)

    if config.cuda:
        model.cuda()

    model_dict = torch.load('Trans_bi_AUC:0.6677045131714876,pep_AUC:0.7167055066570338.pl')
    model.load_state_dict(model_dict)

    # bert_params = []
    # other_params = []

    # set different lr for BERT and other models
    # for name, para in model.named_parameters():
    #     # print(name)
    #     if para.requires_grad:
    #         # if "BERT" in name or "binary_classification" in name or "prot_bert_linear" in name or "pep_bert_linear" in name:
    #         if "BERT" in name:
    #             bert_params += [para]
    #         else:
    #             other_params += [para]
    # params = [
    #     {"params": bert_params, "lr": 1e-5},
    #     {"params": other_params, "lr": 5e-4},
    # ]
    # optimizer = torch.optim.AdamW(params)

    # warm up
    # tot_updates = 15000 // config.batch_size // config.accum_steps
    # warmup_updates = tot_updates // 2
    # lr = 1e-4
    # end_lr = 1e-5
    # power = 1.0
    # scheduler = PolynomialDecayLR(optimizer, warmup_updates, tot_updates, lr, end_lr, power)

    # optimize all parameters in same lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,)  # weight_decay=config.reg

    # balanced cross entropy loss
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 5])).to(config.device)
    criterion = torch.nn.CrossEntropyLoss()
    cri_nonReduce = torch.nn.CrossEntropyLoss(reduction='none')
    cri_nonReduce_weight = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 10]), reduction='none').to(config.device)
    # model.train()

    # print('=' * 50 + 'Start Training' + '=' * 50)
    # valid_metric, test_prc_data, valid_metric_site, test_prc_data_site = train_model(train_iter, valid_iter, test_iter,
    #                                                                                  model, optimizer, criterion,
    #                                                                                  cri_nonReduce, cri_nonReduce_weight, config, iter_k)
    # print('=' * 50 + 'Train Finished' + '=' * 50)

    print('=' * 40 + 'Best Performance iter_k={}'.format(k + 1), '=' * 40)
    if config.model_mode == 1:
        valid_metric, valid_loss, valid_repres_list, valid_label_list, \
        valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, cri_nonReduce, config)
        print(
            '[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = valid_metric.numpy()
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8],
              '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
    else:
        test_metric, test_loss, avg_bi_loss, avg_pep_loss, avg_prot_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data, test_metric_site, test_roc_data_site, test_prc_data_site = model_eval(
            test_iter,
            model,
            criterion,
            cri_nonReduce, cri_nonReduce_weight,
            config)
        print(
            '[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\tAP,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = test_metric.numpy()
        AP_bi = test_prc_data[-1]
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % AP_bi, '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7],
              '  %.5g\t\t' % plmt[8],
              '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
        print('-' * 60 + 'binary prediction over start binding sites predicting' + '-' * 60)
        print(
            '[ACC2,\t\tPrecision2,\t\tSensitivity2,\tSpecificity2,\t\tF12,\t\tAUC2,\t\tAP2,\t\t\tMCC2,\t\t TP2,    \t\tFP2,\t\t\tTN2, \t\t\tFN2]')
        plmt2 = test_metric_site.numpy()
        AP_pep = test_prc_data_site[-1]
        print('%.5g\t\t' % plmt2[0], '%.5g\t\t' % plmt2[1], '%.5g\t\t' % plmt2[2], '%.5g\t\t' % plmt2[3],
              '%.5g\t' % plmt2[4],
              '%.5g\t\t' % plmt2[5], '%.5g\t\t' % AP_pep, '%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7],
              '  %.5g\t\t' % plmt2[8],
              '  %.5g\t\t' % plmt2[9], ' %.5g\t\t' % plmt2[10])
    print('=' * 40 + 'Cross Validation Over' + '=' * 40)

    # valid_performance_list.append(valid_performance)

    '''draw figure'''
    # draw_figure_CV(config, config.learn_name + '_k[{}]'.format(iter_k + 1))

    '''reset plot data'''
    global step_log_interval, train_acc_record, train_loss_record, \
        step_valid_interval, valid_acc_record, valid_loss_record
    step_log_interval = []
    train_acc_record = []
    train_loss_record = []
    step_valid_interval = []
    valid_acc_record = []
    valid_loss_record = []

    return model, valid_performance_list

if __name__ == '__main__':
    '''load configuration'''
    config =config_CPPIF_transformer.get_train_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    # for i in range(config.k_fold):
    for i in [0]:
        k = i
        train_loader_list, valid_loader_list, test_loader_list = data_loader_transformer.load_data(config, k)
        print('=' * 20, 'load data over', '=' * 20)

        '''draw preparation'''
        step_log_interval = []
        train_acc_record = []
        train_loss_record = []
        step_valid_interval = []
        valid_acc_record = []
        valid_loss_record = []
        step_test_interval = []
        test_acc_record = []
        test_loss_record = []

        '''train procedure'''
        valid_performance = 0
        best_performance = 0
        last_test_metric = 0

        # k cross validation
        k_fold_CV(train_loader_list, valid_loader_list, test_loader_list, config, k)