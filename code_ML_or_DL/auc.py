import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# 计算auc，要掌握的是第一个函数，也就是get_auc(labels, preds, n_bins)
# from:https://blog.csdn.net/lieyingkub99/article/details/81266664
# https://zhuanlan.zhihu.com/p/411010918?ivk_sa=1024320u


def get_auc(labels, preds, n_bins=100):
    # 基于排序的公式，注意这个函数等价于下面的calcAUC_byProb()，只不过这个实现方式更高效，所以用这个，下面的calcAUC_byProb()函数可以帮助理解
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins  # 每个桶的宽度等于总分值（1.0）除以桶的数量
    # 把label中每个值放入对应的桶
    for i in range(len(labels)):
        bin_idx = int(preds[i] / bin_width)   # 找到对应的桶的下标
        if labels[i] == 1:
            pos_histogram[bin_idx] += 1
        else:
            neg_histogram[bin_idx] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins): # 这个计算公式等价于下面的54-62行
        satisfied_pair += (
            pos_histogram[i] * accumulated_neg
            + pos_histogram[i] * neg_histogram[i] * 0.5
        )
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)


def calcAUC_byProb(labels, probs): # 这个函数用来帮助理解计算公式
    N = 0           # 正样本数量
    P = 0           # 负样本数量
    neg_prob = []   # 负样本的预测值
    pos_prob = []   # 正样本的预测值
    for index, label in enumerate(labels):
        if label == 1:
            # 正样本数++
            P += 1
            # 把其对应的预测值加到“正样本预测值”列表中
            pos_prob.append(probs[index])
        else:
            # 负样本数++
            N += 1
            # 把其对应的预测值加到“负样本预测值”列表中
            neg_prob.append(probs[index])
    number = 0
    # 遍历正负样本间的两两组合
    for pos in pos_prob:
        for neg in neg_prob:
            # 如果正样本预测值>负样本预测值，正序对数+1
            if (pos > neg):
                number += 1
            # 如果正样本预测值==负样本预测值，算0.5个正序对
            elif (pos == neg):
                number += 0.5
    return number / (N * P)

if __name__ == "__main__":

    y = np.array(
        [
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1
        ]
    )
    pred = np.array(
        [0.9321, 0.8432213, 0.3542132, 0.1212, 0.82143, 0.9231231, 0.9621, 0.7321123,0.8923,0.8623]
    )

    # pred = np.array([0.9, 0.8, 0.3, 0.1,0.4,0.9,0.66,0.7])

    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print("-----sklearn:", auc(fpr, tpr))
    print("-----py脚本:", get_auc(y, pred))
