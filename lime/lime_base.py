"""
Contains abstract functionality for learning locally linear sparse model.
"""
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn.tree
from sklearn.linear_model import Ridge, lars_path, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""

    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.random_seed = random_state

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)  # 距离取指作为样本权重
        # print(weights.shape)
        # print(neighborhood_labels)
        labels_column = neighborhood_labels[:, label]  # 只取一个类别的预测概率
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        if model_regressor != 'DecisionTreeClassifier' and model_regressor != 'DecisionTreeRegression' and model_regressor != 'ensemble':
            # 下面代码只适合分类问题了
            labels_column = np.round(labels_column)
            if model_regressor is None:
                model_regressor = LogisticRegression(fit_intercept=True,
                                                     random_state=self.random_state)
            easy_model = model_regressor
            print(f'label={label}')
            batch_size = 1000
            rounds = 30
            results = {'LC': [], 'RS': [], 'BT': [], 'ET': [], 'MS': []}
            strategies = {'LC': self._lc, 'RS': self._rs, 'BT': self._bt, 'ET': self._et, 'MS': self._ms}
            best_model = easy_model
            best_prec = 0

            # for strategy_name in ['LC']:
            # for strategy_name in ['RS']:
            # for strategy_name in ['ET']:
            for strategy_name in ['MS']:
                # for strategy_name in ['LC', 'RS', 'ET', 'MS']:
                anno_batch = np.concatenate([self._first_rs(neighborhood_data, batch_size), np.array([0])])
                used_index = anno_batch.flatten()
                # anno_batch = np.array([0])
                x_train = neighborhood_data[anno_batch][:, used_features]
                y_train = labels_column[anno_batch]
                # weight_array = anno_batch.flatten()
                for i in range(rounds):
                    # print(anno_batch)
                    easy_model.fit(x_train, y_train, sample_weight=weights[used_index])
                    # prec = easy_model.score(x_train, y_train, sample_weight=weights[used_index])
                    prec = easy_model.score(neighborhood_data[:, used_features], labels_column, sample_weight=weights)
                    results[strategy_name].append(prec)
                    if prec > best_prec:
                        best_model = easy_model
                        best_prec = prec
                    # proba = easy_model.predict_proba(neighborhood_data[[row for row in range(neighborhood_data.shape[0])
                    #                                                     if row not in used_index.tolist()]][:, used_features])
                    proba = easy_model.predict_proba(neighborhood_data[:, used_features])
                    strategy = strategies[strategy_name]
                    # print(proba)
                    anno_batch = strategy(proba, batch_size)
                    used_index = np.concatenate([used_index, anno_batch]).flatten()
                    # print(anno_batch)
                    x_train = np.concatenate([x_train, neighborhood_data[anno_batch][:, used_features]])
                    y_train = np.concatenate([y_train, labels_column[anno_batch]])
                    # weight_array = np.concatenate([weight_array, anno_batch]).flatten()
            # compare_model = model_regressor
            # compare_model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)
            # compare_score = compare_model.score(neighborhood_data[:, used_features], labels_column,
            #                                     sample_weight=weights)
            #
            # # 不同查询方式的性能比较
            # plt.figure()
            # l1 = plt.plot(range(1, len(results['LC']) + 1), results['LC'], color='red', label='LC')
            # l2 = plt.plot(range(1, len(results['RS']) + 1), results['RS'], color='blue', label='RS')
            # l3 = plt.plot(range(1, len(results['ET']) + 1), results['ET'], color='green', label='ET')
            # l3 = plt.plot(range(1, len(results['MS']) + 1), results['MS'], color='yellow', label='MS')
            # l4 = plt.axhline(y=compare_score, color='black', label='origin')
            # plt.legend(labels=['LC', 'RS', 'ET', 'MS', 'origin'])
            # plt.show()

            # easy_model.fit(neighborhood_data[:, used_features],
            #                labels_column, sample_weight=weights)  # 每个样本单独赋予权重
            prediction_score = best_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=weights)  # score为决定系数R^2,其实这里就是通过黑盒模型的预测值和岭回归的预测值进行计算

            local_pred = best_model.predict(neighborhood_data[0, used_features].reshape(1, -1))  # 对感兴趣实例的岭回归预测值

            if self.verbose:
                print('Intercept', best_model.intercept_)
                print('Prediction_local', local_pred)
                print('Right:', neighborhood_labels[0, label])
            # print(easy_model.coef_)
            # print(sorted(zip(used_features, easy_model.coef_[0]),
            #              key=lambda x: np.abs(x[1]), reverse=True))
            return (best_model.intercept_,  # 多项式中的独立项，可以理解为kx+b中的b
                    sorted(zip(used_features, best_model.coef_[0]),
                           key=lambda x: np.abs(x[1]), reverse=True),  # 局部回归模型中的特征权重，按照权重绝对值进行排序
                    prediction_score, local_pred)
        elif model_regressor == 'DecisionTreeClassifier':
            tree_model = DecisionTreeClassifier(random_state=self.random_state)
            labels_column = np.round(labels_column)
            tree_model.fit(neighborhood_data[:, used_features],
                           labels_column, sample_weight=weights)
            prediction_score = tree_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=weights)
            local_pred = tree_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
            if self.verbose:
                print('Intercept', tree_model.feature_importances_)
                print('Prediction_local', local_pred)
                print('Right:', neighborhood_labels[0, label])
            return (None,
                    sorted(zip(used_features, tree_model.feature_importances_),
                           key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred
                    )
        elif model_regressor == 'DecisionTreeRegression':
            tree_model = DecisionTreeRegressor(random_state=self.random_state)
            tree_model.fit(neighborhood_data[:, used_features],
                           labels_column, sample_weight=weights)
            prediction_score = tree_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=weights)
            local_pred = tree_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
            if self.verbose:
                print('Intercept', tree_model.feature_importances_)
                print('Prediction_local', local_pred)
                print('Right:', neighborhood_labels[0, label])
            return (None,
                    sorted(zip(used_features, tree_model.feature_importances_),
                           key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred
                    )
        elif model_regressor == 'ensemble':
            labels_column = np.round(labels_column)
            easy_model1 = LogisticRegression(fit_intercept=True,
                                             random_state=self.random_state)
            easy_model2 = DecisionTreeClassifier(random_state=self.random_state)

            batch_size = 1000
            rounds = 30
            results = {'VE': []}
            strategies = {'VE': self._ve}
            best_model = [easy_model1, easy_model2]
            best_prec = 0

            for strategy_name in ['VE']:
                anno_batch = np.concatenate([self._first_rs(neighborhood_data, batch_size), np.array([0])])
                used_index = anno_batch.flatten()
                # anno_batch = np.array([0])
                x_train = neighborhood_data[anno_batch][:, used_features]
                y_train = labels_column[anno_batch]
                # weight_array = anno_batch.flatten()
                for i in range(rounds):
                    # print(anno_batch)
                    easy_model1.fit(x_train, y_train, sample_weight=weights[used_index])
                    easy_model2.fit(x_train, y_train, sample_weight=weights[used_index])

                    proba1 = easy_model1.predict_proba(neighborhood_data[:, used_features])
                    proba2 = easy_model2.predict_proba(neighborhood_data[:, used_features])
                    proba = (proba1 + proba2) / 2
                    pred = np.argmax(proba, axis=1).flatten()
                    prec = sklearn.metrics.accuracy_score(pred, labels_column)
                    results[strategy_name].append(prec)
                    if prec > best_prec:
                        best_model = [easy_model1, easy_model2]
                        best_prec = prec
                    pred1 = np.argmax(proba1, axis=1)
                    pred2 = np.argmax(proba2, axis=1)
                    # print(pred1.reshape(-1, 1))
                    pred_all = np.concatenate((pred1.reshape(-1, 1), pred2.reshape(-1, 1)), axis=1)
                    strategy = strategies[strategy_name]
                    # print(proba)
                    anno_batch = strategy(pred_all, batch_size)
                    used_index = np.concatenate([used_index, anno_batch]).flatten()
                    # print(anno_batch)
                    x_train = np.concatenate([x_train, neighborhood_data[anno_batch][:, used_features]])
                    y_train = np.concatenate([y_train, labels_column[anno_batch]])
                    # weight_array = np.concatenate([weight_array, anno_batch]).flatten()

            prediction_score = best_prec
            local_pred1 = best_model[0].predict_proba(neighborhood_data[0, used_features].reshape(1, -1))
            local_pred2 = best_model[1].predict_proba(neighborhood_data[0, used_features].reshape(1, -1))
            local_pred = np.argmax((local_pred1 + local_pred2) / 2, axis=1).tolist()[0]
            # print(local_pred)
            coef1 = sklearn.preprocessing.MinMaxScaler().fit_transform(np.abs(best_model[0].coef_[0]).reshape(-1, 1))
            coef2 = sklearn.preprocessing.MinMaxScaler().fit_transform(
                np.array(best_model[1].feature_importances_).reshape(-1, 1))
            coef = (coef1 + coef2) / 2
            if self.verbose:
                # print('Intercept', best_model.intercept_)
                print('Prediction_local', local_pred)
                print('Right:', neighborhood_labels[0, label])
            # print(easy_model.coef_)
            # print(sorted(zip(used_features, easy_model.coef_[0]),
            #              key=lambda x: np.abs(x[1]), reverse=True))
            return (None,  # 多项式中的独立项，可以理解为kx+b中的b
                    sorted(zip(used_features, coef),
                           key=lambda x: np.abs(x[1]), reverse=True),  # 局部回归模型中的特征权重，按照权重绝对值进行排序
                    prediction_score, local_pred)

    def _first_rs(self, proba, batch_size):
        np.random.seed(self.random_seed)
        res = np.random.choice(range(proba.shape[0]), batch_size, replace=False)
        return res

    def _rs(self, proba, batch_size):
        '''
        随机查询
        Args:
            proba:
            batch_size:

        Returns:

        '''
        return np.random.choice(range(proba.shape[0]), batch_size, replace=False)

    def _lc(self, proba, batch_size):
        '''
        查询不确定性最高的样本
        Args:
            proba:
            batch_size:

        Returns:

        '''
        # print(np.argsort(np.max(proba, axis=1))[:batch_size])
        return np.argsort(np.max(proba, axis=1))[:batch_size]

    def _bt(self, proba, batch_size):
        '''
        查询类别间预测概率最接近的样本
        Args:
            proba:
            batch_size:

        Returns:

        '''
        sorted_proba = np.sort(proba, axis=1)
        return np.argsort(np.abs(sorted_proba[:, -1] - sorted_proba[:, -2]))[:batch_size]

    def _et(self, proba, batch_size):
        e = (-proba * np.log2(proba)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:batch_size]
        return selection

    def _ms(self, proba, batch_size):
        rev = np.sort(proba, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:batch_size]
        return selection

    def _ve(self, pred, batch_size, class_count=2):
        pred = np.array(pred)
        # print(pred.shape)
        counts = np.zeros((pred.shape[0], class_count))
        counts[:, 1] = np.sum(pred, axis=1)
        counts[:, 0] = pred.shape[1] - counts[:, 1]
        values = np.zeros((pred.shape[0], 1))
        # print(counts)
        values[:, 0] = - ((counts[:, 0] / pred.shape[1]) * np.log2(counts[:, 0] / pred.shape[1]))
        values[:, 0] = values[:, 0] - ((counts[:, 1] / pred.shape[1]) * np.log2(counts[:, 1] / pred.shape[1]))
        values[np.isnan(values)] = 0
        # print(values)
        selection = np.argsort(values.flatten())[:batch_size]
        # print(selection)
        return selection
