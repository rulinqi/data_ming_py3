# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def run_main():
    print('模型评估与选择')

def fit_prob():
    """
    过拟合与欠拟合
    :return:
    """
    # 加载数据
    # 数字数据集: 任务是预测一张图片中的数字是什么
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(X)
    print(y)
    print(X[0])

    # 1.“刚刚好”
    # gamma=0.001
    train_sizes, train_scores, val_scores = learning_curve(
        SVC(gamma=0.001), X, y, cv=10, scoring='accuracy',
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
    )

    # 在10折的交叉验证数据上进行平均
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    # 绘制学习曲线
    plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='training')
    plt.plot(train_sizes, val_scores_mean, '*-', color='g', label='cross validation')

    plt.xlabel('training sample size')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

    # 2.过拟合
    # gamma=0.1
    train_sizes, train_scores, val_scores = learning_curve(
        SVC(gamma=0.1), X, y, cv=10, scoring='accuracy',
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
    )

    # 在10折的交叉验证数据上进行平均
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    # 绘制学习曲线
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='training')
    plt.plot(train_sizes, val_scores_mean, '*-', color='g', label='cross validation')

    plt.xlabel('training sample size')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

def val_curve():
    """
    验证曲线, 参数的选取
    :return:
    """
    # 加载数据
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(X.shape)
    print(y)

    param_range = np.arange(1, 11) / 2000.
    # param_range = np.logspace(-6.5, -3, 10)
    print(param_range)

    train_scores, val_scores = validation_curve(
        SVC(), X, y, param_name='gamma', param_range=param_range,
        cv=5, scoring='accuracy')

    # 在5折的交叉验证数据上进行平均
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    # 绘制学习曲线
    plt.plot(param_range, train_scores_mean, 'o-', color='b', label='training')
    plt.plot(param_range, val_scores_mean, '*-', color='g', label='cross validation')

    plt.xlabel('gamma')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

def param_tuning():
    """
    交叉验证及参数调整
    :return:
    """
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

    # 设置参数调整的范围及配置
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    svm_model = svm.SVC()

    # 将超参数配置及模型放入GridSearch中进行自动搜索
    clf = GridSearchCV(svm_model, param_grid, cv=5)
    clf.fit(X_train, y_train)

    # 获取选择的最优模型
    best_model = clf.best_estimator_

    # 查看选择的最优超参数配置
    print(clf.best_params_)

    # 预测
    y_pred = best_model.predict(X_test)
    print('accuracy', accuracy_score(y_test, y_pred))

def feat_selection():
    """
    特征选择
    :return:
    """
    # 1. 去除方差小的特征
    # 6个样本，3维的特征向量
    X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]

    # 根据方差保留80%的向量
    # 计算公式：var_thresh = p(1-p)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(X)

    # 2. 基于单变量统计特征选择
    iris = load_iris()
    X, y = iris.data, iris.target
    print('原始特征：')
    print(X.shape)
    print(X[:5, :])
    # 使用卡方分布选择2个维度的变量
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    print('选取的特征：')
    print(X_new.shape)
    print(X_new[:5, :])

    # 3. 基于模型的特征选择
    iris = load_iris()
    X, y = iris.data, iris.target
    print('原始特征：')
    print(X.shape)
    print(X[:5, :])

    clf = RandomForestClassifier()
    clf = clf.fit(X, y)
    print('特征得分：')
    print(clf.feature_importances_)

    # 基于随机森林选择特征
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print('选取的特征：')
    print(X_new.shape)
    print(X_new[:5, :])

if __name__ == '__main__':
    run_main()
