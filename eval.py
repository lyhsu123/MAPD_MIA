import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
def ROC(label, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point
def get_max_accuracy(y_true, probs, thresholds=None):
    if thresholds is None:
        fpr, tpr, thresholds = roc_curve(y_true, probs)
    accuracy_scores = []
    precision_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_true,[1 if m > thresh else 0 for m in probs]))
        precision_scores.append(precision_score(y_true, [1 if m > thresh else 0 for m in probs]))
    accuracies = np.array(accuracy_scores)
    precisions = np.array(precision_scores)
    max_accuracy = accuracies.max()
    max_precision = precisions.max()
    max_accuracy_threshold = thresholds[accuracies.argmax()]
    max_precision_threshold = thresholds[precisions.argmax()]
    return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold
def get_threshold(source_m, source_stats, target_m, target_stats):
    acc_source, t, prec_source, tprec = get_max_accuracy(source_m, source_stats)
    acc_test, t_test, prec_test, _ = get_max_accuracy(target_m, target_stats)
    acc_test_t, _, _, _ = get_max_accuracy(target_m, target_stats, thresholds=[t])
    _, _, prec_test_t, _ = get_max_accuracy(target_m, target_stats, thresholds=[tprec])
    print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test, acc_test_t, t))
    print("prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test, prec_test_t, tprec))
    return acc_test_t, prec_test_t, t, tprec, t_test
def validation_metrics(memberset, nonmemberset, mode='XGB', params=None):
    '''
    memberset和nonmemberset都是已经已训练得到的成员性质；
    该函数：1.把memeberset和nonmemberset分为trainset和testset，两个set都含有member和non_member,
              其中testset中含有等比例的member和non_member，non_member中含有一部分预测错误的样本
            2.输出为balanced_accuracy, precision, recall, F1-score
    '''
    y_mem = np.ones(memberset.shape[0])
    y_nonmem = np.zeros(nonmemberset.shape[0])
    X_train_mem, X_test_mem, y_train_mem, y_test_mem = train_test_split(memberset, y_mem, test_size=0.1, random_state=39)
    X_train_nonmem, X_test_nonmem, y_train_nonmem, y_test_nonmem = train_test_split(nonmemberset, y_nonmem, test_size=0.1, random_state=39)
    X_train = np.vstack([X_train_mem, X_train_nonmem])
    X_test = np.vstack([X_test_mem, X_test_nonmem])
    y_train = np.hstack([y_train_mem, y_train_nonmem])
    y_test = np.hstack([y_test_mem, y_test_nonmem])
    
    y_pred_train = np.zeros(len(y_train))
    y_pred_test = np.zeros(len(y_test))
    random_state = random.randint(0, 200)

    if mode == 'XGB':
        if params is None:
            params = {
            'max_depth' : [5, 10, 16],
            'min_child_weight' : [1, 3, 5],
            'scale_pos_weight' : [1.0, 3.0, 5.0],
            'learning_rate' : [0.0001, 0.001, 0.005],
            'n_estimators' : [100, 200, 500]
            }
            classifier = XGBClassifier(early_stopping_rounds=20)
            classifier = GridSearchCV(classifier, params, n_jobs=-1, scoring='f1', cv=5)
            classifier.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            print("the best params are: ", classifier.best_params_)
            print("the best (f1) score is: ", classifier.best_score_)
            print()
            classifier = classifier.best_estimator_
        else:
            classifier = XGBClassifier(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                              max_depth=params['max_depth'], min_child_weight=params['min_child_weight'], scale_pos_weight=params['scale_pos_weight'])
            classifier.fit(X_train, y_train, verbose=False)
        p_train = classifier.predict(X_train)
        p_test = classifier.predict(X_test)
        print(classifier.get_xgb_params())
    elif mode == 'threshold':
        X_train = np.hstack([X_train[0], X_train[1]])
        X_test = np.hstack([X_test[0], X_test[1]])
#         fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(y_train, X_train)
        acc_test_t, prec_test_t, t, tprec, t_test = get_threshold(y_train, X_train, y_test, X_test)
        p_train = np.where(X_train >= t, 1, 0)
        p_test = np.where(X_test >= t, 1, 0)
#         p_train = np.where(X_train >= tprec, 1, 0)
#         p_test = np.where(X_test >= tprec, 1, 0)
    print("Distance record: acc_train:{}, banlanced_acc_test:{}, precision:{}, recall:{}, f1_score:{}".
          format(accuracy_score(y_train, p_train), accuracy_score(y_test, p_test),
                  precision_score(y_test, p_test, pos_label=1),
                  recall_score(y_test, p_test, pos_label=1),
                  f1_score(y_test, p_test, pos_label=1)))
    return p_train, p_test, y_train, y_test