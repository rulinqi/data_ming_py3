# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def run_main():
    print()

def scikit_ml():
    """
    通过scikit-learn认识机器学习
    :return:
    """
    # 1.加载示例数据集
    iris = datasets.load_iris()
    digits = datasets.load_digits()

    # 查看数据集
    # iris
    print(iris.data[:5])
    print(iris.data)
    print(iris.data.shape)
    print(iris.feature_names)
    print(iris.target_names)
    print(iris.target)

    # digits
    print(digits.data[0])
    print(digits.data.shape)
    print(digits.images)
    print(digits.target_names)
    print(digits.target)

    # 2.在训练集上训练模型
    # 手动划分训练集、测试集
    n_test = 100  # 测试样本个数
    train_X = digits.data[:-n_test, :]
    train_y = digits.target[:-n_test]

    test_X = digits.data[-n_test:, :]
    y_true = digits.target[-n_test:]

    # 选择SVM模型
    svm_model = svm.SVC(gamma=0.001, C=100.)
    # svm_model = svm.SVC(gamma=100., C=1.)
    # 训练模型
    svm_model.fit(train_X, train_y)

    # 选择LR模型
    lr_model = LogisticRegression()
    # 训练模型
    lr_model.fit(train_X, train_y)

    # 3.在测试集上测试模型
    y_pred_svm = svm_model.predict(test_X)
    y_pred_lr = lr_model.predict(test_X)

    # 查看结果
    # print '预测标签：', y_pred
    # print '真实标签：', y_true
    print('SVM结果：', accuracy_score(y_true, y_pred_svm))
    print('LR结果：', accuracy_score(y_true, y_pred_lr))

    # 4.保存模型
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    # 重新加载模型进行预测
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    random_samples_index = np.random.randint(0, 1796, 5)
    print(random_samples_index)
    random_samples = digits.data[random_samples_index, :]
    random_targets = digits.target[random_samples_index]

    random_predict = model.predict(random_samples)

    print(random_predict)
    print(random_targets)

def scikit_tutorial():
    """
    scikit-learn入门
    :return:
    """
    # 1.准备数据集
    X = np.random.randint(0, 100, (10, 4))
    y = np.random.randint(0, 3, 10)
    y.sort()
    print('样本：')
    print(X)
    print('标签：', y)

    # 分割训练集、测试集
    # random_state确保每次随机分割得到相同的结果  random_state：是随机数的种子
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3., random_state=7)
    print('训练集：')
    print(X_train)
    print(y_train)
    print('测试集：')
    print(X_test)
    print(y_test)

    # 特征归一化
    x1 = np.random.randint(0, 1000, 5).reshape(5, 1)
    x2 = np.random.randint(0, 10, 5).reshape(5, 1)
    x3 = np.random.randint(0, 100000, 5).reshape(5, 1)
    print(x1)
    print(np.random.randint(0, 1000, (5, 1)))
    X = np.concatenate([x1, x2, x3], axis=1)
    print(X)
    print(preprocessing.scale(X))

    # 生成分类数据进行验证scale的必要性
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                               random_state=25, n_clusters_per_class=1, scale=100)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    # 注释掉以下这句表示不进行特征归一化
    # X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3., random_state=7)
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_train, y_train)
    svm_classifier.score(X_test, y_test)

    # 2.训练模型
    # 回归模型
    boston_data = datasets.load_boston()
    X = boston_data.data
    y = boston_data.target
    print('样本：')
    print(X[:5, :])
    print('标签：')
    print(y[:5])

    # 选择线性回顾模型
    lr_model = LinearRegression()
    # 分割训练集、测试集
    X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=1/3,random_state=7)
    # 训练模型
    lr_model.fit(X_train, y_train)
    # 返回参数
    lr_model.get_params()
    lr_model.score(X_train, y_train)
    lr_model.score(X_test, y_test)

    # 3.交叉验证
    # K最近邻分类
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3., random_state=10)

    k_range = range(1, 31)
    cv_scores = []
    for n in k_range:
        knn = KNeighborsClassifier(n)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')  # 分类问题使用
        # scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='neg_mean_squared_error') # 回归问题使用
        cv_scores.append(scores.mean())

    plt.plot(k_range, cv_scores)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
    # 选择最优的K
    best_knn = KNeighborsClassifier(n_neighbors=5)
    best_knn.fit(X_train, y_train)
    print(best_knn.score(X_test, y_test))
    print(best_knn.predict(X_test))


def scikit_pca():
    """
    主成分分析
    :return:
    """
    digits = datasets.load_digits()
    X_digits, y_digits = digits.data, digits.target

    n_row, n_col = 2, 5

    def plot_digits(images, y, max_n=10):
        """
            显示手写数字的图像
        """
        # 设置图像尺寸
        fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        i = 0
        while i < max_n and i < images.shape[0]:
            p = fig.add_subplot(n_row, n_col, i + 1, xticks=[], yticks=[])
            p.imshow(images[i], cmap=plt.cm.bone, interpolation='nearest')
            # 添加标签
            p.text(0, -1, str(y[i]))
            i = i + 1

    plot_digits(digits.images, digits.target, max_n=10)

    def plot_pca_scatter(X_pca):
        """
            主成分显示
        """
        colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
        for i in range(len(colors)):
            # 只显示前两个主成分在二维坐标系中
            # 如果想显示前三个主成分，可以放在三维坐标系中。有兴趣的可以自己尝试下
            px = X_pca[:, 0][y_digits == i]
            py = X_pca[:, 1][y_digits == i]
            plt.scatter(px, py, c=colors[i])
        plt.legend(digits.target_names)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

        n_components = 10  # 取前10个主成分
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_digits)
        plot_pca_scatter()

        def print_pca_components(images, n_col, n_row):
            plt.figure(figsize=(2. * n_col, 2.26 * n_row))
            for i, comp in enumerate(images):
                plt.subplot(n_row, n_col, i + 1)
                plt.imshow(comp.reshape((8, 8)), interpolation='nearest')
                plt.text(0, -1, str(i + 1) + '-component')
                plt.xticks(())
                plt.yticks(())

        print_pca_components(pca.components_[:n_components], n_col, n_row)

if __name__ == '__main__':
    run_main()