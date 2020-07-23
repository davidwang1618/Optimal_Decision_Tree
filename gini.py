import numpy as np
from collections import Counter


def frequency(data: np.ndarray, attribute: int) -> np.array:
    '''
    calculate frequency for each values in an attribute.
    
    input:
        data: data we want to deal with.
        attribute: index of the attribute we want to calculate frequency.
    
    return: empirical frequency
    '''
    assert isinstance(data, np.ndarray), 'data should be a np array.'
    if data.shape[0] == 0: return np.array([])
    counter = Counter(data[:, attribute])
    return np.array([count/data.shape[0] for count in counter.values()])


def gini_impurity(data: np.ndarray, attribute: int) -> float:
    '''
    calculate gini impurity of an attribute.
    reference: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    
    input:
        data: data we want to deal with.
        attribute: index of the attribute we want to calculate gini coefficient.
    
    return: gini impurity
    '''
    return 1 - np.sum(np.square(frequency(data, attribute)))