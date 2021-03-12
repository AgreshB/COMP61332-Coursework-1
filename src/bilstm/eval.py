import numpy as np

# generate a confusion matrix using actual and predicted labels
def get_confusion_matrix(y_actual, y_pred, size):
    matrix = np.zeros((size, size) , dtype=np.dtype(np.int32))
    for i in range(0, len(y_actual)):
        if y_actual[i] == y_pred[i]:
            matrix[y_actual[i] - 1][y_actual[i] - 1] += 1
        else:
            matrix[y_actual[i] - 1][y_pred[i] - 1] += 1
    return matrix

# caluclate micro f1 score using confusion matrix
def get_micro_f1(confusion_matrix):
    matrix = np.array(confusion_matrix)
    tp, fp, fn = [], [], []

    for i in range(0, np.size(matrix, 0)):
        tp.append(matrix[i][i])
        fp.append(np.sum(matrix[:][i]) - matrix[i][i])
        fn.append(np.sum(matrix[i][:]) - matrix[i][i])

    tp_sum = np.sum(np.array(tp))
    fp_sum = np.sum(np.array(fp))
    fn_sum = np.sum(np.array(fn))

    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    return 2 * precision * recall / (precision + recall)

# calculate macro f1 using confusion matrix
def get_macro_f1(confusion_matrix):
    matrix = np.array(confusion_matrix)
    precision, recall = [], []
    f1 = []

    for i in range(0, np.size(matrix,0)):
        tp=matrix[i][i]

        fp = np.sum(matrix[:][i]) - matrix[i][i]
        fn = np.sum(matrix[i][:]) - matrix[i][i]

        p, r = 0, 0

        if tp != 0 or fp != 0:
            p = tp / (tp + fp)
            precision.append(tp / (tp + fp))

        if tp != 0 or fn != 0:
            r = tp / (tp + fn)
            recall.append(tp / (tp + fn))

        if p != 0 or r != 0:
            f1.append(2 * p * r / (p + r))
        else:
            f1.append(0)

    precision_avg = np.mean(np.array(precision))
    recall_avg = np.mean(np.array(recall))

    return 2 * precision_avg * recall_avg / (precision_avg + recall_avg), f1