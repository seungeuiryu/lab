import csv
import re
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

place = '해운대'

file = open(f'data.csv', 'r', encoding='utf-8', newline='')
reviews = [row['review'].lower() for row in csv.DictReader(file)
           if row['language'] == '영어' and row['place'] == place]

sents = [sent_tokenize(rv) for rv in reviews]
sents = sum(sents, [])

lemmatizer = WordNetLemmatizer()


def tokenize(text):
    text = re.sub('[^a-zA-Z0-9 ]', '', text).strip()

    # tokenize
    wds = [wd for wd in word_tokenize(text)
           if wd not in stopwords.words('english')]

    # tag pos
    pos = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS',
           'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    wds_pos = [wd for wd in pos_tag(wds) if wd[1] in pos]

    # lemmatize
    wds_pos_lem = []
    for wd in wds_pos:
        try:
            wd = lemmatizer.lemmatize(wd[0], wd[1][0].lower())
        except:
            wd = wd[0]
        wds_pos_lem.append(wd)

    return wds_pos_lem


rv_processed = [tokenize(rv) for rv in reviews]
file = open('review_token.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(file)
writer.writerows([rv for rv in rv_processed if rv])

st_processed = [tokenize(sent) for sent in sents]
file = open('sentence_token.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(file)
writer.writerows([st for st in st_processed if st])

