import numpy as np
from sklearn.metrics import roc_auc_score

def generate_score(name_score, test_labels, predictions, predictions_proba):
    '''
    Generate score
    @params:
        - name_score: 'accuracy', 'auc', 'gacc', 'f1', 'p20', 'npv', 'ppv'
        - test_labels
        - predictions
        - predictions_proba
    @return: score result
    '''
    classes, count_class = np.unique(test_labels, return_counts = True)
    class_min = classes[0] if count_class[0] < count_class[1] else classes[1]
    class_maj = classes[0] if count_class[0] > count_class[1] else classes[1]

    if name_score == 'accuracy':
        tp = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_min)))
        tn = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_maj)))
        fp = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_maj)))
        fn = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_min)))
        p = tp + fn
        n = fp + tn
        if tp+tn != 0:
            return (tp+tn)/(p+n)
        else:
            return 0

    elif name_score == 'auc':
        return roc_auc_score(test_labels, predictions_proba[:, 1])

    elif name_score == 'gacc':
        tp = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_min)))
        tn = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_maj)))
        fp = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_maj)))
        fn = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_min)))
        p = tp + fn
        n = fp + tn
        if tp != 0:
            return np.sqrt(tp/p*tn/n)
        else:
            return 0

    elif name_score == 'f1':
        tp = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_min)))
        tn = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_maj)))
        fp = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_maj)))
        fn = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_min)))
        p = tp + fn
        n = fp + tn
        if tp != 0:
            return 2*tp / (2*tp + fp + fn)
        else:
            return 0

    elif name_score == 'p20':
        test_labels_20, preds = zip(*sorted(zip(test_labels, predictions_proba[:, 1]), key=lambda x: -x[1]))
        return np.sum(np.array(test_labels_20)[:int(0.2*len(np.array(test_labels_20)))] == class_maj)/int(0.2*len(np.array(test_labels_20)))
    
    elif name_score == 'ppv':
        tp = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_min)))
        tn = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_maj)))
        fp = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_maj)))
        fn = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_min)))
        if tp != 0:
            return tp/(tp+fp)
        else:
            return 0
    
    elif name_score == 'npv':
        tp = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_min)))
        tn = np.sum(np.logical_and(np.equal(test_labels, predictions), (test_labels == class_maj)))
        fp = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_maj)))
        fn = np.sum(np.logical_and(np.logical_not(np.equal(test_labels, predictions)), (test_labels == class_min)))
        if tn != 0:
            return tn/(tn+fn)
        else:
            return 0