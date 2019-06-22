import numpy as np 
import pandas as pd 

def onehot_labels(labels):
    """
    One-hot encoder to encode labels into multi-columns.
    ======================================================================
    param labels: 
        a list or np.array, where each element of labels is the label of each sample.
    ======================================================================
    return:
        np.array, in column i, 0 and 1 represent whether the label is belong to class i.
    ======================================================================
    example:
        labels = [0,1,2,2,1]
        onehot_labels = onehot(labels)
        onehot_labels
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [0., 0., 1.],
               [0., 1., 0.]])

    """
    num_of_class = len(np.unique(labels))
    output = np.zeros([len(labels), num_of_class])
    for i, label in enumerate(list(np.unique(labels))):
        output[labels == label, i] = 1
    return output