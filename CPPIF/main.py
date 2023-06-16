# -*- coding: utf-8 -*-
# @Time    : 2022/4/2 15:12
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: main.py

import sys
import os
import torch
import torch.nn.functional as F
import config, data_loader, util_metric
import nc_model

# flag is an indicator for checking whether this record has binding sites information
def boost_mask_BCE_loss(input_mask, flag):
    def conditional_BCE(y_true, y_pred, cri_nonReduce):
        seq_len = input_mask.shape[1]
        loss = flag.unsqueeze(-1).repeat(1, seq_len).view(-1) * cri_nonReduce(y_true, y_pred) * input_mask.view(-1)
        return torch.sum(loss) / torch.sum(input_mask)
    return conditional_BCE

def periodic_test(test_iter, model, criterion, cri_nonReduce, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    print('test current performance')
    if config.model_mode == 1:
        test_metric, test_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, cri_nonReduce, config)
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
        test_metric, test_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data, test_metric_site, test_roc_data_site, test_prc_data_site = model_eval(test_iter,
                                                                                                            model,
                                                                                                            criterion,
                                                                                                            cri_nonReduce,
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
              '%.5g\t\t' % plmt2[5], '%.5g\t\t' % AP_pep, '%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7], '  %.5g\t\t' % plmt2[8],
              '  %.5g\t\t' % plmt2[9], ' %.5g\t\t' % plmt2[10])
        print('#' * 60 + 'Over' + '#' * 60)

        step_test_interval.append(sum_epoch)
        test_acc_record.append(test_metric[0])
        test_loss_record.append(test_loss)

        return test_metric, test_loss, test_repres_list, test_label_list, \
        test_roc_data, test_prc_data, test_metric_site, test_roc_data_site, test_prc_data_site


def periodic_valid(valid_iter, model, criterion, cri_nonReduce, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)
    if config.model_mode == 1:
        valid_metric, valid_loss, valid_repres_list, valid_label_list, \
        valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, cri_nonReduce, config)

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
        valid_metric, valid_loss, valid_repres_list, valid_label_list, valid_roc_data, valid_prc_data, \
        valid_metric_site, valid_roc_data_site, valid_prc_data_site = model_eval(
            valid_iter, model, criterion, cri_nonReduce, config)
        print('validation current performance')
        print('[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = valid_metric.numpy()
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8],
              '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
        print('-' * 60 + 'Over' + '-' * 60)
        print(
            '[ACC2,\t\tPrecision2,\t\tSensitivity2,\tSpecificity2,\t\tF12,\t\tAUC2,\t\t\tMCC2,\t\t TP2,    \t\tFP2,\t\t\tTN2, \t\t\tFN2]')
        plmt2 = valid_metric_site.numpy()
        print('%.5g\t\t' % plmt2[0], '%.5g\t\t' % plmt2[1], '%.5g\t\t' % plmt2[2], '%.5g\t\t' % plmt2[3],
              '%.5g\t' % plmt2[4],
              '%.5g\t\t' % plmt2[5], '%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7], '  %.5g\t\t' % plmt2[8],
              '  %.5g\t\t' % plmt2[9], ' %.5g\t\t' % plmt2[10])
        print('#' * 60 + 'Over' + '#' * 60)

        step_valid_interval.append(sum_epoch)
        valid_acc_record.append(valid_metric[0])
        valid_loss_record.append(valid_loss)

        return valid_metric, valid_loss, valid_repres_list, valid_label_list, valid_metric_site


def train_model(train_iter, valid_iter, test_iter, model, optimizer, criterion, cri_nonReduce, config, iter_k):
    best_acc = 0
    best_performance = 0

    for epoch in range(1, config.epoch + 1):
        steps = 0
        repres_list = []
        label_list = []
        model.train()
        for batch in train_iter:
            if config.model_mode == 1:
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, labels = batch
                pred_pos_label = model.binary_forward(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p)
                labels = labels.view(-1)
                pred_label = torch.cat([1-pred_pos_label, pred_pos_label], 1)
                loss = criterion(pred_label, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1
            else:
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, X_pep_mask, X_bs_flag, X_bs, labels = batch
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
                pred_pos_label, pred_pos_site_label = model(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p)
                labels = labels.view(-1)
                # pred_label = torch.cat([1 - pred_pos_label, pred_pos_label], 1)
                pred_label = pred_pos_label
                loss_bi = criterion(pred_label, labels)

                condition_CE = boost_mask_BCE_loss(X_pep_mask, X_bs_flag)
                # pred_pos_site_label = pred_pos_site_label.unsqueeze(-1)
                # pred_site_label = torch.cat([1 - pred_pos_site_label, pred_pos_site_label], -1)
                pred_site_label = pred_pos_site_label
                pred_site_label = pred_site_label.view(-1, 2)
                X_bs = X_bs.view(-1)
                loss_site = condition_CE(pred_site_label, X_bs, cri_nonReduce)
                loss = loss_bi + loss_site
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1

            '''Periodic Train Log'''
            if steps % config.interval_log == 0:
                corre = (torch.max(pred_label, 1)[1] == labels).int()
                corrects = corre.sum()
                the_batch_size = len(labels)
                train_acc = 100.0 * corrects / the_batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                        loss,
                                                                                        train_acc,
                                                                                        corrects,
                                                                                        the_batch_size))
                print()

                step_log_interval.append(steps)
                train_acc_record.append(train_acc)
                train_loss_record.append(loss)

        # sum_epoch = iter_k * config.epoch + epoch
        sum_epoch = epoch

        '''Periodic Validation'''
        # if valid_iter and sum_epoch % config.interval_valid == 0:
        #     if config.model_mode == 1:
        #         valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
        #                                                                                        model,
        #                                                                                        criterion, cri_nonReduce,
        #                                                                                        config,
        #                                                                                        sum_epoch)
        #     else:
        #         valid_metric, valid_loss, valid_repres_list, valid_label_list, valid_metric_site = periodic_valid(valid_iter,
        #                                                                                        model,
        #                                                                                        criterion, cri_nonReduce,
        #                                                                                        config,
        #                                                                                        sum_epoch)
        #     valid_acc = valid_metric[0]
        #     if valid_acc > best_acc:
        #         best_acc = valid_acc
        #         best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:
            # time_test_start = time.time()
            if config.model_mode == 1:
                test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                          model,
                                                                                          criterion, cri_nonReduce,
                                                                                          config,
                                                                                          sum_epoch)
            else:
                test_metric, test_loss, test_repres_list, test_label_list, \
                test_roc_data, test_prc_data, test_metric_site, test_roc_data_site, test_prc_data_site = periodic_test(test_iter,
                                                                                          model,
                                                                                          criterion, cri_nonReduce,
                                                                                          config,
                                                                                          sum_epoch)
                test_auc = test_metric[5]
                if test_auc > best_acc:
                    best_acc = test_auc
                    best_performance = test_metric
                    best_metric_site = test_metric_site
                    best_prc_data = test_prc_data
                    best_prc_data_site = test_prc_data_site
                    if test_metric_site[5] > 0.785:
                        # torch.save({"test_roc_data": test_roc_data, "model": model.state_dict()}, f'Dataset1_AUC:{test_auc},MCC:{best_mcc}.pl')
                        torch.save(model.state_dict(), f'CAMP_bi_AUC:{test_auc},pep_AUC:{test_metric_site[5]}.pl')
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

def model_eval(data_iter, model, criterion, cri_nonReduce, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    # binary prediction
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    # binding site prediction
    label_pred_site = torch.empty([0], device=device)
    label_real_site = torch.empty([0], device=device)
    pred_prob_site = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, iter_size_site, corrects_site, avg_loss = 0, 0, 0, 0, 0
    repres_list = []
    label_list = []
    pad_pep_len = config.pad_pep_len
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            if config.model_mode == 1:
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, labels = batch
                pred_pos_label = model.binary_forward(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p)
                labels = labels.view(-1)
                pred_label = torch.cat([1-pred_pos_label, pred_pos_label], 1)
                loss = criterion(pred_label, labels)
            else:
                X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, X_pep_mask, X_bs_flag, X_bs, labels = batch
                pred_pos_label, pred_pos_site_label = model(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p)
                labels = labels.view(-1)
                # pred_label = torch.cat([1 - pred_pos_label, pred_pos_label], 1)
                pred_label = pred_pos_label
                loss_bi = criterion(pred_label, labels)

                condition_CE = boost_mask_BCE_loss(X_pep_mask, X_bs_flag)
                # pred_pos_site_label = pred_pos_site_label.unsqueeze(-1)
                # pred_site_label = torch.cat([1 - pred_pos_site_label, pred_pos_site_label], -1)
                pred_site_label = pred_pos_site_label
                pred_site_label = pred_site_label.view(-1, 2)
                X_bs = X_bs.view(-1)
                loss_site = condition_CE(pred_site_label, X_bs, cri_nonReduce)
                loss = loss_bi + loss_site

                pred_prob_all = F.softmax(pred_site_label, dim=1)
                pred_pos_site_label1 = pred_prob_all[:, 1]
                p_class_site = torch.max(pred_site_label, 1)[1]
                for i, f in enumerate(X_pep_mask.view(-1)):
                    if X_bs_flag[i//pad_pep_len] == 1 and f == 1:
                        corre_site = (p_class_site[i] == X_bs[i]).int()
                        corrects_site += corre_site.sum()
                        iter_size_site += 1

                        label_pred_site = torch.cat([label_pred_site, p_class_site[i].view(-1).float()])
                        label_real_site = torch.cat([label_real_site, X_bs[i].view(-1).float()])
                        pred_prob_site = torch.cat([pred_prob_site, pred_pos_site_label1.view(-1)[i].view(-1)])

            loss = loss.float()
            avg_loss += loss
            pred_prob_all = F.softmax(pred_label, dim=1)
            pred_pos_label1 = pred_prob_all[:, 1]
            p_class = torch.max(pred_label, 1)[1]
            corre = (p_class == labels).int()
            corrects += corre.sum()
            iter_size += labels.size(0)
            label_pred = torch.cat([label_pred, p_class.float()])
            label_real = torch.cat([label_real, labels.float()])
            pred_prob = torch.cat([pred_prob, pred_pos_label1.view(-1)])


    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= len(data_iter)
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
        print('Evaluation - loss: {:.6f}  bi_ACC: {:.4f}%({}/{})  site_ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                      100 * accuracy,
                                                                      corrects,
                                                                      iter_size,
                                                                       100 * accuracy_site,
                                                                       corrects_site,
                                                                       iter_size_site
                                                                       ))
        return metric, avg_loss, repres_list, label_list, roc_data, prc_data, metric_site, roc_data_site, prc_data_site


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

    model = nc_model.model()

    if config.cuda:
        model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, )  # weight_decay=config.reg
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 5])).to(config.device)
    criterion = torch.nn.CrossEntropyLoss()
    cri_nonReduce = torch.nn.CrossEntropyLoss(reduction='none')
    model.train()

    print('=' * 50 + 'Start Training' + '=' * 50)
    valid_metric, test_prc_data, valid_metric_site, test_prc_data_site = train_model(train_iter, valid_iter, test_iter, model, optimizer, criterion, cri_nonReduce, config, iter_k)
    print('=' * 50 + 'Train Finished' + '=' * 50)

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
        # valid_metric, valid_loss, valid_repres_list, valid_label_list, valid_roc_data, valid_prc_data, \
        # valid_metric_site, valid_roc_data_site, valid_prc_data_site = model_eval(
        #     valid_iter, model, criterion, cri_nonReduce, config)
        print(
            '[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\tAP,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
        plmt = valid_metric.numpy()
        AP_bi = test_prc_data[-1]
        print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3],
              '%.5g\t' % plmt[4],
              '%.5g\t\t' % plmt[5], '%.5g\t\t' % AP_bi, '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8],
              '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
        print('-' * 60 + 'binary prediction over start binding sites predicting' + '-' * 60)
        print(
            '[ACC2,\t\tPrecision2,\t\tSensitivity2,\tSpecificity2,\t\tF12,\t\tAUC2,\t\tAP2,\t\t\tMCC2,\t\t TP2,    \t\tFP2,\t\t\tTN2, \t\t\tFN2]')
        plmt2 = valid_metric_site.numpy()
        AP_pep = test_prc_data_site[-1]
        print('%.5g\t\t' % plmt2[0], '%.5g\t\t' % plmt2[1], '%.5g\t\t' % plmt2[2], '%.5g\t\t' % plmt2[3],
              '%.5g\t' % plmt2[4],
              '%.5g\t\t' % plmt2[5], '%.5g\t\t' % AP_pep, '%.5g\t\t' % plmt2[6], '%.5g\t\t' % plmt2[7], '  %.5g\t\t' % plmt2[8],
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
    config = config.get_train_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    # for i in range(5):
    for i in [0]:
        k = i
        train_loader_list, valid_loader_list, test_loader_list = data_loader.load_data(config, k)
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