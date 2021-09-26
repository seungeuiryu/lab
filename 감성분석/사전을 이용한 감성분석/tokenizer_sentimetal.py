from konlpy.tag import Mecab
import score as ss
import tokenizer_sentence as tt
import movie_review_stopword_code as ms
import pandas as pd

# 기타 함수

f = open('/Users/xiu0327/lab/2021_07_17/step2_tokenizer/dictionary_gam/부사.txt', 'r', encoding='utf-8')
data_file = open('/Users/xiu0327/lab/감성분석/data/38766_output.txt', 'r', encoding='utf-8')
data_text = data_file.read()
data = data_text.split('\n')
busa_text = f.read()
busa = busa_text.split('\n')

movie_sentiment_dic = pd.read_csv('/Users/xiu0327/lab/2021_07_17/dictionary_tokenizer.csv', encoding='utf-8')

def BusaCheck(word):
    if word in busa:
        return True
    else:
        return False

def vector_sum(array):
    sum = 0
    for num in array:
        sum+=int(num[1])
    return sum
def vector_probability(result):
    neg_count = 0
    pos_count = 0
    neu_count = 0
    for i in result:
        if int(i[1]) < 0:
            neg_count += 1
        if int(i[1]) > 0:
            pos_count += 1
        if int(i[1]) == 0:
            neu_count += 1
    all_count = neg_count + pos_count + neu_count
    if(all_count != 0):
        neg_pro = neg_count/all_count
        pos_pro = pos_count/all_count
        neu_pro = neu_count/all_count
    return [neg_pro, pos_pro, neu_pro]

# step1. 형태소 분석

def get_tokenizer(corpus_fname, output_fname, pos=False):
    tokenizer = Mecab()
    result_tokens = []
    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            if pos:
                tokens = tokenizer.pos(line)
                tokens = [morph + "/" + tag for morph, tag in tokens]
            else:
                tokens = tokenizer.morphs(line)
            result_tokens.append(tokens)
            for token in tokens:
                f2.write(token+", ")
            f2.write('\n')
    print("형태소 분석 완료. " + output_fname + "을 확인해주세요.\n")
    return result_tokens

# step2. 불용어 제거

def get_stopword(default_stopword, user_stopword, output_fname, step1):
    dic1_file = open(default_stopword, 'r', encoding='utf-8')
    dic2_file = open(user_stopword, 'r', encoding='utf-8')

    dic1 = dic1_file.read()
    dic2 = dic2_file.read()

    dic1 = dic1.split('\n')
    dic2 = dic2.split('\n')

    dic = dic1+dic2

    result_stopword = []

    for line in step1:
        result_tmp = []
        for word in line:
            if word not in dic:
                result_tmp.append(word)
        result_stopword.append(result_tmp)


    output_file = open(output_fname, 'w', encoding='utf-8')

    for line in result_stopword:
        for word in line:
            output_file.write(word + ", ")
        output_file.write('\n')
    print("불용어 제거가 완료되었습니다. "+output_fname+"을 확인해주세요.")
    return result_stopword

# step3. 사전을 이용하여 극성점수 부여

def get_score(step2, output_fname):
    output_file = open(output_fname, 'w', encoding='utf-8')
    score_class = ss.KnuSL
    result_score = []
    sen_idx = 0
    for line in step2:
        result = []
        for word in line:
            tmp_score = score_class.data_list(word)
            if tmp_score == 'None':
                for i in enumerate(movie_sentiment_dic):
                    if word == i[0]:
                        tmp_score = str(i[0])
                if tmp_score == 'None':
                    tmp_score = "0"
            result.append([word, tmp_score])

        result_score.append(result)

        for i in range(len(result)):
            if BusaCheck(result[i][0]):
                try:
                    result[i + 1][1] = str(int(result[i + 1][1]) * int(result[i][1]))
                    result[i][1] = "0"
                except:
                    print("가중치를 부여할 단어가 없음. 부사가 문장의 마지막 단어.\n")


        if vector_sum(result) < 0:
            output_file.write("<" + data[sen_idx] + ">" + "는 부정입니다\n")
            prob = vector_probability(result)
            output_file.write("부정: "+str(prob[0])+", 긍정: "+str(prob[1])+", 중립: "+str(prob[2])+'\n')
            for detail in result:
                output_file.write("["+detail[0]+", "+detail[1]+"] ")
            output_file.write('\n')
        elif vector_sum(result) > 0:
            output_file.write("<" + data[sen_idx] + ">" + "는 긍정입니다\n")
            prob = vector_probability(result)
            output_file.write("부정: "+str(prob[0])+", 긍정: "+str(prob[1])+", 중립: "+str(prob[2])+'\n')
            for detail in result:
                output_file.write("["+detail[0]+", "+detail[1]+"] ")
            output_file.write('\n')
        else:
            output_file.write("<" + data[sen_idx] + ">" + "는 중립입니다\n")
            prob = vector_probability(result)
            output_file.write("부정: "+str(prob[0])+", 긍정: "+str(prob[1])+", 중립: "+str(prob[2])+'\n')
            for detail in result:
                output_file.write("["+detail[0]+", "+detail[1]+"] ")
            output_file.write('\n')

        sen_idx+=1

    print("극성 점수 계산이 완료되었습니다. " + output_fname + "을 확인해주세요.")
    return result_score














step1 = get_tokenizer('/Users/xiu0327/lab/감성분석/data/38766_output.txt','tokenizer_형태소분석 파일.txt')
#step2 = get_stopword('/Users/xiu0327/lab/2021_07_17/step2_tokenizer/stopwords-ko/stopwords-ko.txt', '/Users/xiu0327/lab/2021_07_17/step2_tokenizer/movie_review_stopword.txt', 'tokenizer_불용어제거 파일.txt', step1)
step3 = get_score(step1, 'tokenizer_극성 점수 계산.txt')

