import matplotlib.pyplot as plt
from konlpy.tag import Mecab
import json
import os
from pprint import pprint
from matplotlib import font_manager, rc
import nltk
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
        # 헤더 제외하고 fata 저장
    return data

train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

m = Mecab('C:/mecab/mecab-ko-dic')

def tokenize(doc):
    # flatten = 정규화 , stem = 근어 ?
    return ['/'.join(t) for t in m.pos(doc, flatten=False, join=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json', encoding='utf-8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', encoding='utf-8') as f:
        test_docs = json.load(f)

else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # json 파일로 저장
    with open('train_docs.json', 'w', encoding='utf-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent='\t')
    with open('test_docs.json', 'w', encoding='utf-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent='\t')

tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')
font_fname = 'C:/Windows/Fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

#plt.figure(figsize=(20,10))
#text.plot(50)
# plt.show()
# 그래프 확인 코드
print('완료')
selected_words = [f[0] for f in text.vocab().most_common(1000)]
print('완료')

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

print('완료')

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=10, batch_size=512)
result = model.evaluate(x_test, y_test)

save_file = 'C:/Users/xiu03/lab/donga/감성분석/gam_model'
model.save(save_file)
