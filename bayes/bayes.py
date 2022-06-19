# coding=UTF-8
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from random import shuffle
import jieba


def makeStopWord():
    with open('../data/stopword.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    stopWord = []
    for line in lines:
        words = jieba.lcut(line, cut_all=False)
        for word in words:
            stopWord.append(word)
    return stopWord


def getWords(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.read()
    lines = lines.split('\n')
    splitsen = ['  '.join(jieba.lcut(line, cut_all=False)) for line in lines]
    return splitsen


def convert2Data(posArray, negArray):
    randIt = []
    data = []
    labels = []
    for i in range(len(posArray)):
        randIt.append([posArray[i], 1])
    for i in range(len(negArray)):
        randIt.append([negArray[i], 0])
    shuffle(randIt)
    for i in range(len(randIt)):
        data.append(randIt[i][0])
        labels.append(randIt[i][1])
    return data, labels


def makedata(pospath, negpath):
    pos = getWords(pospath)
    neg = getWords(negpath)
    data, label = convert2Data(pos, neg)

    return data, label


def main():

    traindata, trainlabel = makedata('../data/dataset/Pos-train.txt',
                                     '../data/dataset/Neg-train.txt')
    testdata, testlabel = makedata('../data/dataset/Pos-test.txt',
                                   '../data/dataset/Neg-test.txt')
    stpwordpth = '../data/stopword.txt'
    with open(stpwordpth, 'rb') as fp:
        stopword = fp.read().decode('utf-8')  # 停用词提取
    stpwrdlst = stopword.splitlines()

    vectorizer = CountVectorizer(stop_words=stpwrdlst)
    x = vectorizer.fit_transform(traindata).toarray()
    y = np.array(trainlabel)

    gnb = GaussianNB()

    scores = cross_val_score(gnb, x, y, cv=5, scoring='f1_weighted')  # accuracy be default
    print(scores)
    print("Means: ", scores.mean())

    gnb.fit(x, y)
    x_test = vectorizer.transform(testdata).toarray()

    y_test = gnb.predict(x_test)
    testlabel = np.array(testlabel)
    eq = np.sum(y_test == testlabel)
    acc = eq / len(testlabel)
    print("In test data,the accuracy is:{:.2f}".format(acc))
    print(len(y_test), len(testlabel))
    print(y_test)
    print(type(y_test))
    print(testlabel)
    print(type(testlabel))


if __name__ == '__main__':
    main()
