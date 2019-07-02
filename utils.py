import math
import heapq
import numpy as np
import os

num_users, num_items = 0, 0


def get_pairwise_train_dataset(path='data/ml1m_train.dat'):
    global num_users, num_items
    print('loading pair-wise data from flie %s...' % path)
    user_input, item_i_input, item_j_input = [], [], []
    with open(path, 'r') as f:
        for line in f:
            arr = line.split(' ')
            u, i, j = int(arr[0]), int(arr[1]), int(arr[2])
            user_input.append(u)
            item_i_input.append(i)
            item_j_input.append(j)
            if u > num_users:
                num_users = u
            if i > num_items or j > num_items:
                num_items = max(i, j)
    return num_users, num_items, user_input, item_i_input, item_j_input


def get_test_data(path='data/ml1m_test_ratings.lsvm'):
    print('loading test data from file %s...' % path)
    testRatings = dict()
    testItems = dict()
    with open(path, 'r') as f:
        for u, line in enumerate(f):
            testRatings[u], testItems[u] = list(), list()
            for item_rating in line.strip().split(' '):
                item, rating = int(item_rating.split(':')[0]), int(float(item_rating.split(':')[1]))
                testItems[u].append(item)
                testRatings[u].append(rating)

    return testItems, testRatings


_model = None
_testItems = None
_testRatings = None
_K = None
_item_rating_dict = None

def evaluate_model(model, testItems, testRatings, K):
    global _model
    global _testItems
    global _testRatings
    global _K
    _model = model
    _testItems = testItems
    _testRatings = testRatings
    _K = K

    metrics = np.array([0. for _ in range(6)])
    for user in _testItems.keys():
            metrics += eval_one_rating(user)
    return metrics / len(_testItems)


def eval_one_rating(user):
    global _item_rating_dict
    ratings = _testRatings[user]
    items = _testItems[user]
    item_rating_dict = dict()
    for item, rating in zip(items, ratings):
        item_rating_dict[item] = rating
    _item_rating_dict = item_rating_dict
    k_largest_items = heapq.nlargest(_K, item_rating_dict, key=item_rating_dict.get)
    
    # Get prediction scores
    users = np.full(len(items), user, dtype='int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)    
    item_prediction_dict = dict()
    for item, prediction in zip(items, predictions):
        item_prediction_dict[item] = prediction
    sorted_item = heapq.nlargest(len(item_rating_dict), item_prediction_dict, key=item_prediction_dict.get)
    top_labels = [1 if item in k_largest_items else 0 for item in sorted_item]

    # Evaluate top rank list
    hr = getHitRatio(top_labels[:_K])
    p = getPrecision(top_labels[:_K])
    ndcg_bin = getNDCG_bin(top_labels[:_K])
    auc = getAUC(top_labels)
    map = getMAP(top_labels)
    mrr = getMRR(top_labels)
    return np.array([hr, p, ndcg_bin, auc, map, mrr])


def getHitRatio(labels):
    return 1 if 1 in labels else 0


def getPrecision(labels):
    return sum(labels) / len(labels)


def getNDCG_bin(labels):
    dcg, max_dcg = 0, 0
    for i, label in enumerate(labels):
        dcg += label / math.log2(i + 2)
        max_dcg += 1 / math.log2(i + 2)
    return dcg / max_dcg


def getAUC(labels):
    global _K
    if len(labels) <= _K:
        return 1

    auc = 0
    for i, label in enumerate(labels[::-1]):
        auc += label * (i + 1)

    return (auc - _K * (_K + 1) / 2) / (_K * (len(labels) - _K))


def getMAP(labels):
    global _K
    MAP = 0
    for i, label in enumerate(labels):
        MAP += label * getPrecision(labels[:i + 1])
    return MAP / _K


def getMRR(labels):
    global _K
    mrr = 0
    for i, label in enumerate(labels):
        mrr += label * (1 / (i + 1))
    return mrr / _K
