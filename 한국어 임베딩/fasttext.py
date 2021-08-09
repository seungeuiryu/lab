from gensim.models import FastText
from tqdm import tqdm

corpus_fname = './data/corpus_mecab.txt'
model_fname = './data/fasttext'

print('corpus 생성')
#말뭉치 생성

corpus = [sent.strip().split(" ") for sent in tqdm(open(corpus_fname, 'r', encoding='utf-8').readlines())]

print("학습 중")



model = FastText(corpus, vector_size=100, workers=4, sg=1, min_count=6, word_ngrams=1)

model.save(model_fname)
# https://projector.tensorflow.org/ 에서 시각화 하기 위해 모델을 따로 저장
model.wv.save_word2vec_format(model_fname + "_vis")
print('완료')
