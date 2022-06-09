"""
KNN算法实现
"""
import math
import numpy as np
from collections import Counter


class KNN(object):
    def __init__(self, data: np.ndarray, instance: np.ndarray, k=5):
        """
        初始化knn参数
        Args:
            data: 包括特征和标签的ndarray
            instance: 感兴趣实例，不包含标签
            k: 邻居数
        """
        self.data = data
        self.instance = instance
        self.k = k
        self.distances = []
        self.result = []

    def fit(self):
        for row in self.data:
            euclidean_distance = np.sqrt(np.sum((row[:-1] - self.instance) ** 2))
            self.distances.append([euclidean_distance, row[-1]])
        self.distances = sorted(self.distances, key=lambda x: x[0])
        top_nearest = self.distances[:self.k]
        top_class = [i[1] for i in top_nearest]
        res_class = Counter(top_class).most_common(1)[0][0]
        for row in self.data:
            if row[-1] == res_class:
                self.result.append(row[:-1])
