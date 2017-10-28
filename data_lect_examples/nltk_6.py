# -*- encoding: utf-8 -*-

import nltk
from nltk.corpus import brown # 需要下载brown语料库   引用布朗大学的语料库
import jieba
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk import FreqDist
from nltk.text import TextCollection

def run_main():
    print()

def nltk():
    """
    nltk,文本处理简介
    :return:
    """
    # 1.语料库
    # 查看语料库包含的类别
    print(brown.categories())
    # 查看brown语料库
    print('共有{}个句子'.format(len(brown.sents())))
    print('共有{}个单词'.format(len(brown.words())))

    # 2.分词
    sentence = "Python is a widely used high-level programming language for general-purpose programming."
    tokens = nltk.word_tokenize(sentence)  # 需要下载punkt分词模型
    print(tokens)

    # 3.结巴分词
    seg_list = jieba.cut("欢迎来到小象学院", cut_all=True)
    print("全模式: " + "/ ".join(seg_list))  # 全模式

    seg_list = jieba.cut("欢迎来到小象学院", cut_all=False)
    print("精确模式: " + "/ ".join(seg_list))  # 精确模式

    # 4.词形处理
    # 词干提取(stemming)
    # PorterStemmer
    from nltk.stem.porter import PorterStemmer

    porter_stemmer = PorterStemmer()
    print(porter_stemmer.stem('looked'))
    print(porter_stemmer.stem('looking'))

    # SnowballStemmer
    from nltk.stem import SnowballStemmer

    snowball_stemmer = SnowballStemmer('english')
    print(snowball_stemmer.stem('looked'))
    print(snowball_stemmer.stem('looking'))

    # LancasterStemmer
    from nltk.stem.lancaster import LancasterStemmer

    lancaster_stemmer = LancasterStemmer()
    print(lancaster_stemmer.stem('looked'))
    print(lancaster_stemmer.stem('looking'))

    # 词形归并(lemmatization)
    from nltk.stem import WordNetLemmatizer  # 需要下载wordnet语料库

    wordnet_lematizer = WordNetLemmatizer()
    print(wordnet_lematizer.lemmatize('cats'))
    print(wordnet_lematizer.lemmatize('boxes'))
    print(wordnet_lematizer.lemmatize('are'))
    print(wordnet_lematizer.lemmatize('went'))

    # 指明词性可以更准确地进行lemma
    # lemmatize 默认为名词
    print(wordnet_lematizer.lemmatize('are', pos='v'))
    print(wordnet_lematizer.lemmatize('went', pos='v'))

    #词性标注 (Part-Of-Speech)
    words = nltk.word_tokenize('Python is a widely used programming language.')
    print(nltk.pos_tag(words))  # 需要下载 averaged_perceptron_tagger

    # 去除停用词
    from nltk.corpus import stopwords  # 需要下载stopwords

    filtered_words = [word for word in words if word not in stopwords.words('english')]
    print('原始词：', words)
    print('去除停用词后：', filtered_words)

    # 5.典型的文本预处理流程
    # 原始文本
    raw_text = 'Life is like a box of chocolates. You never know what you\'re gonna get.'

    # 分词
    raw_words = nltk.word_tokenize(raw_text)

    # 词形归一化
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]

    # 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    print('原始文本：', raw_text)
    print('预处理结果：', filtered_words)

def sentiment_analysis():
    """
    情感分析
    :return:
    """
    # 使用机器学习实现
    text1 = 'I like the movie so much!'
    text2 = 'That is a good movie.'
    text3 = 'This is a great one.'
    text4 = 'That is a really bad movie.'
    text5 = 'This is a terrible movie.'

    def proc_text(text):
        """
            预处处理文本
        """
        # 分词
        raw_words = nltk.word_tokenize(text)

        # 词形归一化
        wordnet_lematizer = WordNetLemmatizer()
        words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]

        # 去除停用词
        filtered_words = [word for word in words if word not in stopwords.words('english')]

        # True 表示该词在文本中，为了使用nltk中的分类器
        return {word: True for word in filtered_words}

    # 构造训练样本
    train_data = [[proc_text(text1), 1],
                  [proc_text(text2), 1],
                  [proc_text(text3), 1],
                  [proc_text(text4), 0],
                  [proc_text(text5), 0]]

    # 训练模型
    nb_model = NaiveBayesClassifier.train(train_data)

    # 测试模型
    text6 = 'That is a bad one.'
    print(nb_model.classify(proc_text(text5)))

def text_similarity():
    """
    文本相似度
    :return:
    """
    text1 = 'I like the movie so much '
    text2 = 'That is a good movie '
    text3 = 'This is a great one '
    text4 = 'That is a really bad movie '
    text5 = 'This is a terribl  e movie'

    text = text1 + text2 + text3 + text4 + text5
    words = nltk.word_tokenize(text)
    freq_dist = FreqDist(words)
    print(freq_dist['is'])

    # 取出常用的n=5个单词
    n = 5
    # 构造“常用单词列表”
    most_common_words = freq_dist.most_common(n)
    print(most_common_words)

    def lookup_pos(most_common_words):
        """
            查找常用单词的位置
        """
        result = {}
        pos = 0
        for word in most_common_words:
            result[word[0]] = pos
            pos += 1
        return result

    # 记录位置
    std_pos_dict = lookup_pos(most_common_words)
    print(std_pos_dict)

    # 新文本
    new_text = 'That one is a good movie. This is so good!'

    # 初始化向量
    freq_vec = [0] * n

    # 分词
    new_words = nltk.word_tokenize(new_text)

    # 在“常用单词列表”上计算词频
    for new_word in new_words:
        if new_word in list(std_pos_dict.keys()):
            freq_vec[std_pos_dict[new_word]] += 1

    print(freq_vec)

def text_classification():
    """
    文本分类
    :return:
    """
    text1 = 'I like the movie so much '
    text2 = 'That is a good movie '
    text3 = 'This is a great one '
    text4 = 'That is a really bad movie '
    text5 = 'This is a terrible movie'

    # 构建TextCollection对象
    tc = TextCollection([text1, text2, text3,
                         text4, text5])
    new_text = 'That one is a good movie. This is so good!'
    word = 'That'
    tf_idf_val = tc.tf_idf(word, new_text)
    print('{}的TF-IDF值为：{}'.format(word, tf_idf_val))


if __name__ == '__main__':
    run_main()