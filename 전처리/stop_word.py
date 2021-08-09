from nltk.tokenize import word_tokenize
import re

EMAIL_PATTERN = re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.UNICODE)
NEWS_PATTERN1 = re.compile("\(.*\)|\s-\s.*", re.UNICODE)
NEWS_PATTERN2 = re.compile("\[.*\]|\s-\s.*", re.UNICODE)

file = open('processed_news.txt', 'r', encoding='utf-8')
stop_word = open('stop_word.txt', 'r', encoding='utf-8')
output = open('stopword_news.txt', 'w', encoding='utf-8')

stop_words = stop_word.read()
stop_words = stop_words.split('\n')

line = file.readline()


while line:
    _, sentence = line.strip().split('\t')
    sentence = sentence.replace('   ', ' ')
    sentence = sentence.replace('  ', ' ')
    sentence_arr = sentence.split(' ')
    sentence_result = []
    for sent in sentence_arr:
        sent = re.sub(EMAIL_PATTERN, '', sent)
        sent = re.sub(URL_PATTERN, '', sent)
        sent = re.sub(NEWS_PATTERN1, '', sent)
        sent = re.sub(NEWS_PATTERN2, '', sent)
        if sent == '':
            continue
        else:
            sentence_result.append(sent)

    sentence = ' '.join(sentence_result)
    word_tokens = word_tokenize(sentence)

    result = []
    for w in word_tokens:
        if w not in stop_words:
            result.append(w)

    stopword_sent = ' '.join(result)
    output.writelines(stopword_sent + '\n')

    line = file.readline()