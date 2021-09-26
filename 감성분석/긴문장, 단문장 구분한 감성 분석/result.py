import keras.models
import numpy as np
import pandas as pd
import tokenizer


model = keras.models.load_model('C:/Users/xiu03/lab/donga/감성분석/gam_model')

def predict_pos_neg(review):
    token = tokenizer.tokenize(review)
    tf = tokenizer.term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        return ("[{}] : 확률 = {:.2f}% 결과 = 긍정\n".format(review, score*100))
    else:
        return ("[{}] : 확률 = {:.2f}% 결과 = 부정\n".format(review, score * 100))


data_csv = pd.read_csv('184318_data.csv', encoding='utf-8')
data = data_csv['document']

output = open('movie_output.txt','w', encoding='utf-8')
for sentence in data:
    output.write(predict_pos_neg(sentence))
