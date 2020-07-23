import numpy as np

from numpy.random import randint, random
from typing import Dict, Union, Tuple
from gini import gini_impurity


class OC1:
    '''
    OC1 oblique decision tree.
    reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.6068&rep=rep1&type=pdf
    '''
    
    def __init__(self, metric=gini_impurity):
        '''
        constructor:
        store impurity metric function;
        create stag_prob which will be used in self.__perturb().
        '''
        self.metric = metric
        self.stagment = 1
        self.tree = {}  # use dict to store a tree.
    
    
    def fit(self, data: np.ndarray, perturbation: int=20) -> None:
        '''
        fit OC1 to training data.
        input:
            data: training set, note that data[:, -1] is the label column.
            perturbation: number of times of perturbation when constructing each node.
        return: None, but store the OC1 tree in self.tree
        '''
        self.tree = self.__create_OC1(data)
    
    
    def predict(self, data: np.ndarray) -> np.array:
        '''
        predict a set of data.
        input: data we want to predict.
            note that data has to have a label column as the last column,
            but the values of labels are not used.
        return: predicted labels
        '''
        return np.array([self.__predict_single(obs) for obs in data])
        
    
    def get_depth(self) -> int:
        '''
        return the depth of self.tree
        if there is only one node in self.tree, its depth is 0 instead of 1.
        '''
        def dfs_compute_depth(tree):
            '''
            helper function: use DFS to get tree depth.
            '''
            if not isinstance(tree, dict): return 0
            return 1 + max(dfs_compute_depth(tree['left']), dfs_compute_depth(tree['right']))
        
        return dfs_compute_depth(self.tree)    
                
                
    def __create_OC1(self, data: np.ndarray, perturbation: int=20):
        '''
        techinical details of self.fit().
        input:
            data: training set with the labels as the last column
            perturbation: number of times of perturbation when constructing each node.
        return: OC1 tree in dict form.
            - tree['split_vec'] is the splitting vector for this node
            - tree['left'], tree['right'] are the left and right subtrees represented in dict
        '''
        # deal with corner cases: no data and leaf.
        if data.shape[0] == 0: return None
        
        is_pure, leaf = self.__is_pure(data)
        if is_pure: return leaf
        
        # get best split for each attribute.
        # splits is a 3d np.ndarray with
        # d1: attribute index
        # d2: best split point of this attribute
        # d3: impurity after best split on this attribute
        splits = self.__get_all_best_splits(data)
        
        # for all these attributes, we choose our initial split as 
        # the one which can yield the lowest impurity.
        # split_info = (split point we choose, impurity after this split)
        index, split_info = min(enumerate(splits), key=lambda x: x[1][1])
        
        # for oblique split, we need a split vector.
        split_vec = np.zeros((data.shape[1], ))
        split_vec[-1] = -split_info[0]
        split_vec[index] = 1
        
        # use the initial split to split data and save the impurity.
        # this is our benchmark.
        left, right = self.__split_data(data, split_vec)
        impurity = self.metric(left, -1) + self.metric(right, -1)
        
        for _ in range(perturbation):
            attr_perturbed = randint(0, len(split_vec)-1)
            impurity, split_vec = self.__perturb(data, split_vec, attr_perturbed, impurity)
        
        # construct left & right subtrees recursively.
        tree = {'split_vec': split_vec}
        left_data, right_data = self.__split_data(data, split_vec)
        left_tree = self.__create_OC1(left_data)
        tree['left'] = left_tree
        right_tree = self.__create_OC1(right_data)
        tree['right'] = right_tree
        
        return tree
    
    
    def __predict_single(self, observation: np.ndarray) -> Union[float, int]:
        '''
        predict the label of a single observation.
        '''
        OC1 = self.tree
        while isinstance(OC1, dict):
            split_vec = OC1['split_vec']
            go_left = self.__split_func(observation, split_vec) < 0
            if go_left: OC1 = OC1['left']
            else: OC1 = OC1['right']
        
        return OC1  
    
    
    def __is_pure(self, data: np.ndarray) -> Tuple[bool, Union[int, float]]:
        '''
        check if data is pure in terms of labels.
        note that the label is column (-1) of data.
        
        input: data to be checked
        return: (if data is pure, label of data if it is pure)
        '''
        y = data[:, -1].reshape(-1)
        return (all(y == y[0]), y[0])
    
    
    def __get_split_pts(self, data: np.ndarray, attribute: int) -> np.array:
        '''
        get the split points of an attribute, which are just the middle points of data[:, attribute]
        input:
            data, attribute: attribute & data to compute middle points on
        return: middle points of data[:, attribute]
        '''
        attribute_vals = np.sort(data[:, attribute])
        return np.convolve(attribute_vals, np.array([0.5, 0.5]))
    
    
    def __get_best_split_pt(self, data: np.ndarray, attribute: int) -> Tuple[float, float]:
        '''
        get the best split point of an attribute.
        input:
            data, attribute: attribute & data to consider
        return: (best split point, impurity after best split)
        '''
        split_pts = self.__get_split_pts(data, attribute)
        metric_vals = {}
        for split_pt in split_pts:
            in_left_tree = data[:, attribute] < split_pt
            left_tree, right_tree = data[in_left_tree], data[~in_left_tree]
            metric_vals[split_pt] = self.metric(left_tree, -1) + self.metric(right_tree, -1)
            
        return min(metric_vals.items(), key=lambda x: x[1])
    
    
    def __get_all_best_splits(self, data: np.ndarray) -> np.ndarray:
        '''
        get best split points for all attributes.
        input: data we will consider.
        return: 3-d np.ndarray with
            d1: attribute index
            d2: best split point of this attribute
            d3: impurity after best split on this attribute
        '''
        return np.array([self.__get_best_split_pt(data, i) for i in range(data.shape[1]-1)])
    
    
    def __split_data(self, data: np.ndarray, split_vec: np.array) -> Tuple[np.ndarray, np.ndarray]:
        '''
        split data given a split vector.
        input: data & split vector we shall use
        return: (data in left subtree, data in right subtree)
        '''
        left, right = np.zeros(data.shape), np.zeros(data.shape)
        idx_left, idx_right = 0, 0
        for observation in data:
            if self.__split_func(observation, split_vec) > 0:
                right[idx_right] = observation
                idx_right += 1
            else:
                left[idx_left] = observation
                idx_left += 1
        
        left = left[~np.all(left==0, axis=1)]
        right = right[~np.all(right==0, axis=1)]
        return left, right     
    
    
    def __split_func(self, obervation: np.ndarray, split_vec: np.array) -> float:
        '''
        calculate the value of our splitting function on a specific observation.
        input: observation & split vector in this case
        return: value of splitting function
        '''
        return np.sum(np.multiply(obervation.reshape(-1)[:-1], split_vec[:-1])) + split_vec[-1]
    
    
    def __calculate_U(self, observation: np.ndarray, split_vec: np.array, attribute: int) -> float:
        '''
        Compute U_j (U for a specific attribute and observation) using Eq(1) in the paper.
        input: observation, split vector & attribute we shall use
        return: value of U_j in Eq(1) in the paper.
        '''
        am = split_vec[attribute]
        numerator = am * observation[attribute] - self.__split_func(observation, split_vec)
        return numerator / observation[attribute]
    
    
    def __perturb(self, data: np.ndarray, split_vec: np.array, attribute: int, prev_impurity: float) -> Tuple[float, np.array]:
        '''
        perturb a specific attribute following Fig 1 in the paper.
        input:
            data: sample data we use
            split_vec: current split vector
            attribute: attribute index we shall perturb
            prev_impurity: previous impurity we have reached
        return: (best impurity, best split vector) after perturbation
        '''
        Us = np.array(sorted([self.__calculate_U(obs, split_vec, attribute) for obs in data])).reshape(-1, 1)
        possible_splits = self.__get_split_pts(Us, 0)
        
        # find the best split am1
        am_dict = {}
        for split_val in possible_splits:
            new_split_vec = np.array(split_vec)
            new_split_vec[attribute] = split_val
            left, right = self.__split_data(data, new_split_vec)
            impurity = self.metric(left, -1) + self.metric(right, -1)
            am_dict[split_val] = (impurity, new_split_vec)
        
        best_new_impurity, best_new_split_vec = min(am_dict.values(), key=lambda x: x[0])
        
        # if the new split is better, return the new split information.
        if best_new_impurity < prev_impurity:
            self.stagment = 1
            return best_new_impurity, best_new_split_vec
        
        # if the new and old ones are equally good, use exp(-stag_prob) to accept the change.
        elif best_new_impurity == prev_impurity:
            if random() > np.exp(-self.stagment):
                self.stagment += 1
                return best_new_impurity, best_new_split_vec
        
        # else return the old split information.
        return prev_impurity, split_vec