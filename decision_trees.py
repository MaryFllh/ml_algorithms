import numpy as np

from collections import Counter


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, value=None
    ) -> None:
        self.feature = feature
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
            return Node(terminal_node_value)

        # otherwise, keep splitting on the best feature's best value

        # find best splits on the number of desired features
        feature_inds = np.random.choice(samples_num, self.features_num, replace=False)
        best_feature, best_thresh = self._find_best_split(X, y, feature_inds)

        # create children

    def _find_most_common_label(y):
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

        Args:
            X(array): input features
            y(array): corresponding labels
            threshold(float): the value X is split on

        Returns:
            information_gain(float): the information gain from making the split
        """
        pass

    def _compute_entropy(self, X):
        pass

    def predit(self, X):
        pass
