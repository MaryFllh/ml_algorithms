import numpy as np

from collections import Counter


class Node:
    def __init__(
        self, feature_ind=None, threshold=None, left=None, right=None, value=None
    ) -> None:
        self.feature_ind = feature_ind
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree:
    def __init__(
        self, included_features_num=None, max_depth=100, min_samples_for_split=2
    ) -> None:
        """
        There are a number of different stopping criteria in creating a decision tree:
            - The maximum predefined tree depth has been reached, i.e. max_depth
            - There are fewer than a predefined number of samples in a node, i.e. min_samples_for_split
            - All the samples in a node have the same label
        The user-defined criteria are used to instantiate the class.

        Args:
            included_features_num(int): number of features to include in building the tree, allows for
                a fewer number of features to be used rather than using all the features in the input
            max_depth(int): maximum tree depth before stopping
            min_samples_for_split(int): the number of samples in each node should be more than this number
        """
        self.included_features_num = included_features_num
        self.max_depth = max_depth
        self.min_samples_for_split = min_samples_for_split
        self.root = None

    def fit(self, X, y):
        self.samples_num, self.input_features_num = X.shape
        self.features_num = (
            self.input_features_num
            if not self.included_features_num
            else min(self.included_features_num, self.input_features_num)
        )
        self.root = self._create_decision_tree(X, y)

    def _create_decision_tree(self, X, y, depth=0):
        """
        Creates the tree by recursively finding the best feature and the best threshold to split on

        Args:
            X(array): input features
            y(array): true labels
            depth(int): the current depth of the tree

        Returns:
            root(Node): root of the tree
        """
        samples_num, features_num = X.shape
        if (
            depth >= self.max_depth
            or samples_num < self.min_samples_for_split
            or len(np.unique(y)) == 1
        ):
            # stopping criteria is satisfied, return node with most common label as its value
            terminal_node_value = self._find_most_common_label(y)
            return Node(value=terminal_node_value)

        # otherwise, keep splitting on the best feature's best value
        feature_inds = np.random.choice(features_num, self.features_num, replace=False)
        best_feature_ind, best_thresh = self._find_best_split(X, y, feature_inds)

        # create children
        left_child_inds, right_child_inds = self._create_children(X[:, best_feature_ind], best_thresh)
        left_child = self._create_decision_tree(X[left_child_inds, :], y[left_child_inds], depth+1)
        right_child = self._create_decision_tree(X[right_child_inds, :], y[right_child_inds], depth+1)
        
        return Node(best_feature_ind, best_thresh, left_child, right_child)

    def _find_most_common_label(self, y):
        label_counts = Counter(y)
        return label_counts.most_common(1)[0][0]

    def _find_best_split(self, X, y, feature_inds):
        """
        Given the input feature arrays and their corresponding labels,
        iterates through each feature and for each feature iterates through
        all possible values, and computes the information gain from splitting
        on that feature's value. If the gain is higher than previous splits,
        the split is made on this feature and threshold (value). This continues
        until the feature with the highest information gain is obtained.

        Args:
            X(array): input features
            y(array): labels corresponding to the input
            features_inds(array): the indices of the input features and labels
                we are interested in

        Returns:
            best_feature_ind(int): index of the feature with highest information gain
            best_threshold(int): the value that the best_feature should be split on
        """
        best_info_gain = -1
        best_feature_ind, best_threshold = None, None

        for ind in feature_inds:
            X_subset = X[:, ind]
            thresholds = np.unique(X_subset)
            for threshold in thresholds:
                info_gain = self._compute_information_gain(X_subset, y, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_threshold = threshold
                    best_feature_ind = ind

        return best_feature_ind, best_threshold

    def _compute_information_gain(self, X, y, threshold):
        """
        Computes the information gain from splitting X on threshold
        which equals the entropy of X minus the weighted entropy of
        its children. The children are the samples with smaller or bigger
        than the threshold.

        Args:
            X(array): input features
            y(array): corresponding labels
            threshold(float): the value X is split on

        Returns:
            information_gain(float): the information gain from making the split
        """
        parent_entropy = self._compute_entropy(y)

        # split to left and right based on threshold
        left_child_inds, right_child_inds = self._create_children(X, threshold)
            
        if len(left_child_inds) == 0 or len(right_child_inds) == 0:
            # there is no change in distribution made by the split
            return 0

        # compute weight of each split
        left_weight = len(left_child_inds) / len(y)
        right_weight = len(right_child_inds) / len(y)
        
        # compute child entropy
        left_entropy = left_weight * self._compute_entropy(y[left_child_inds])
        right_entropy = right_weight * self._compute_entropy(y[right_child_inds]) 
        child_entropy = left_entropy + right_entropy
        
        return parent_entropy - child_entropy

    def _compute_entropy(self, y):
        """
        Computes the entropy based on the labels y. Entropy is calculated as
        the summation of the probability of occuring times the log2 of that
        probability over all possible values. - sum (p * log2 p)

        Args:
            y(array): label of each data point

        Returns:
            entropy(float): the entropy of the input
        """
        probs = np.bincount(y) / len(y)
        entropy = -sum([p * np.log2(p) for p in probs if p > 0])
        return entropy
        
    def _create_children(self, X, threshold):
        """
        Given input features and the threshold to be split on,
        splits the input to left and right children.

        Args:
            X(array): input feature that needs to be split based on its values
            threshold(float): threshold that determines the split

        Returns:
            left_child_inds(array): indices corresponding to the subset of the 
                input that has values smaller/equal than threshold
            right_child_inds(array): indices corresponding to a subset of the
                input that has values larger than threshold
        """
        left_child_inds = np.argwhere(X <= threshold).flatten()
        right_child_inds = np.argwhere(X > threshold).flatten()
        return left_child_inds, right_child_inds

    def predict(self, X):
        return [self._traverse_tree(Xi, self.root) for Xi in X] 
    
    def _traverse_tree(self, X, root):
        if root.is_leaf_node():
            return root.value

        if X[root.feature_ind] <= root.threshold:
            # go down left child
            return self._traverse_tree(X, root.left)
        else:
            # go down right child
            return self._traverse_tree(X, root.right)