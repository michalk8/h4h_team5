import numpy as K
eps = 1e-15

def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + eps)


def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + eps)


def f1_m(y_true, y_pred):
       precision_ = precision(y_true, y_pred)
       recall_ = recall(y_true, y_pred)
       return 2*((precision_ * recall_) / (precision_ + recall_ + eps))
