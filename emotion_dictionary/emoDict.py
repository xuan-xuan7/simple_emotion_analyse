import zhon.hanzi
import re
import jieba
import os


# 读取文件，文件读取函数
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        # 返回list类型数据
        text = text.split('\n')
    return text


# 将数据写入文件中
def write_data(filename, data):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w', encoding='utf-8') as f:
        for line in data:
            for a in line:
                f.write(str(a))
                f.write('\t')
            f.write('\n')
    f.close()


# 文本分句
def cut_sentence(text):
    sentence_list = re.findall(zhon.hanzi.sentence, text)
    return sentence_list


# 文本分词
def tokenize(sentence):
    word_list = [w for w in jieba.cut(sentence)]
    return word_list


# 去停用词函数
def del_stopwords(words):
    # 读取停用词表
    stopwords = read_file("../data/hit_stopwords.txt")
    # 去除停用词后的句子
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words


# 获取六种权值的词，根据要求返回list
def weighted_value(request):
    result_dict = []
    if request == "one":
        result_dict = read_file("../data/dictionary/most.txt")
    elif request == "two":
        result_dict = read_file("../data/dictionary/very.txt")
    elif request == "three":
        result_dict = read_file("../data/dictionary/more.txt")
    elif request == "four":
        result_dict = read_file("../data/dictionary/ish.txt")
    elif request == "five":
        result_dict = read_file("../data/dictionary/insufficiently.txt")
    elif request == "six":
        result_dict = read_file("../data/dictionary/inverse.txt")
    elif request == 'posdict':
        result_dict = read_file("../data/dictionary/pos_all_dict.txt")
    elif request == 'negdict':
        result_dict = read_file("../data/dictionary/neg_all_dict.txt")
    else:
        pass
    return result_dict


print("reading sentiment dict .......")
# 读取情感词典
posdict = weighted_value('posdict')
negdict = weighted_value('negdict')
# 读取程度副词词典
# 权值为2
mostdict = weighted_value('one')
# 权值为1.75
verydict = weighted_value('two')
# 权值为1.50
moredict = weighted_value('three')
# 权值为1.25
ishdict = weighted_value('four')
# 权值为0.25
insufficientdict = weighted_value('five')
# 权值为-1
inversedict = weighted_value('six')


# 程度副词处理，对不同的程度副词给予不同的权重
def match_adverb(word, sentiment_value):
    # 最高级权重为
    if word in mostdict:
        sentiment_value *= 8
    # 比较级权重
    elif word in verydict:
        sentiment_value *= 6
    # 比较级权重
    elif word in moredict:
        sentiment_value *= 4
    # 轻微程度词权重
    elif word in ishdict:
        sentiment_value *= 2
    # 相对程度词权重
    elif word in insufficientdict:
        sentiment_value *= 0.5
    # 否定词权重
    elif word in inversedict:
        sentiment_value *= -1
    else:
        sentiment_value *= 1
    return sentiment_value


# 打分
def single_sentiment_score(text_sent):
    sentiment_scores = []
    # 对单条分句
    sentences = cut_sentence(text_sent)
    # print(sentences)
    for sent in sentences:
        words = tokenize(sent)
        seg_words = del_stopwords(words)
        # i，s 记录情感词和程度词出现的位置
        i = 0  # 记录扫描到的词位子
        s = 0  # 记录情感词的位置
        poscount = 0  # 记录积极情感词数目
        negcount = 0  # 记录消极情感词数目
        # 逐个查找情感词
        for word in seg_words:
            # 如果为积极词
            if word in posdict:
                poscount += 1  # 情感词数目加1
                # 在情感词前面寻找程度副词
                for w in seg_words[s:i]:
                    poscount = match_adverb(w, poscount)
                s = i + 1  # 记录情感词位置
            # 如果是消极情感词
            elif word in negdict:
                negcount += 1
                for w in seg_words[s:i]:
                    negcount = match_adverb(w, negcount)
                s = i + 1
            # 如果结尾为感叹号或者问号，表示句子结束，并且倒序查找感叹号前的情感词，权重+4
            elif word == '!' or word == '！' or word == '?' or word == '？':
                for w2 in seg_words[::-1]:
                    # 如果为积极词，poscount+2
                    if w2 in posdict:
                        poscount += 4
                        break
                    # 如果是消极词，negcount+2
                    elif w2 in negdict:
                        negcount += 4
                        break
            i += 1  # 定位情感词的位置
        # 计算情感值
        sentiment_score = poscount - negcount
        sentiment_scores.append(sentiment_score)
        # 查看每一句的情感值
        # print('分句分值：',sentiment_score)
    sentiment_sum = 0
    for s in sentiment_scores:
        sentiment_sum += s
    return sentiment_sum


# 返回一个列表，列表中元素为（分值，微博）元组
def run_score(contents):
    scores_list = []
    for content in contents:
        if content != '':
            score = single_sentiment_score(content)
            scores_list.append((score, content))
    return scores_list


# 计算百分比
def format_percentage(a, b):
    p = 100 * a / b
    if p == 0.0:
        q = '0%'
    else:
        q = f'%.2f%%' % p
    return q


# 主程序
if __name__ == '__main__':
    print('Processing........')
    # 测试
    pos_train_data_path = '../data/dataset/Pos-train.txt'
    neg_train_data_path = '../data/dataset/Neg-train.txt'
    pos_test_data_path = '../data/dataset/Pos-test.txt'
    neg_test_data_path = '../data/dataset/Neg-test.txt'
    sentences = read_file(pos_train_data_path)
    scores_pos_train = run_score(sentences)
    sentences = read_file(pos_test_data_path)
    scores_pos_test = run_score(sentences)
    sentences = read_file(neg_train_data_path)
    scores_neg_train = run_score(sentences)
    sentences = read_file(neg_test_data_path)
    scores_neg_test = run_score(sentences)
    write_data('../dictionary_result/pos_train.txt', scores_pos_train)
    write_data('../dictionary_result/neg_train.txt', scores_neg_train)
    write_data('../dictionary_result/pos_test.txt', scores_pos_test)
    write_data('../dictionary_result/neg_test.txt', scores_neg_test)
    pos_count = 0
    neg_count = 0
    pos_right = 0
    neg_right = 0
    for score in scores_pos_train:
        if score[0] >= 0:
            pos_right += 1
        pos_count += 1
    for score in scores_pos_test:
        if score[0] >= 0:
            pos_right += 1
        pos_count += 1
    for score in scores_neg_train:
        if score[0] <= 0:
            neg_right += 1
        neg_count += 1
    for score in scores_neg_test:
        if score[0] <= 0:
            neg_right += 1
        neg_count += 1
    pos_accurate = format_percentage(pos_right, pos_count)
    neg_accurate = format_percentage(neg_right, neg_count)
    all_accurate = format_percentage(pos_right + neg_right, pos_count + neg_count)
    print("positive评价的正确率为：" + pos_accurate)
    print("negative评价的正确率为：" + neg_accurate)
    print("全部评价的正确率为：" + all_accurate)
    print('succeed.......')
