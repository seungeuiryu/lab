from konlpy.tag import Mecab,Okt
import re

def using_tokenizer(sen):
    stop_word = ['있', '많', '었', 'ㅎ']
    tokenizer = Okt()
    tag = ['NNG', 'NNP', 'VA', 'IC']
    # 일반명사, 고유명사, 형용사, 감탄사
    token = tokenizer.pos(sen)
    for word in token:
        if word[0] in stop_word:
            token.remove(word)
    tmp = []
    for w in token:
        if w[1] in tag:
            tmp.append(w[0])
    return tmp

def not_using_tokenizer(sen):
    token = sen.replace('.', '').split(' ')
    return token

inputFile = open('트립어드바이저_해운대_리뷰.txt', 'r', encoding='utf-8')
outputFile1 = open('Okt_사용_O.txt', 'w', encoding='utf-8')
outputFile2 = open('mecab_사용_X.txt', 'w', encoding='utf-8')


with open('트립어드바이저_해운대_리뷰.txt', 'r', encoding='utf-8') as f:
    for line in f:
        arr1 = using_tokenizer(line)
        arr2 = not_using_tokenizer(line)

        outputFile1.write('@'.join(arr1)+'\n')
        outputFile2.write('@'.join(arr2))

