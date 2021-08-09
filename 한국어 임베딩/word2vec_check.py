from gensim.models import KeyedVectors

# 모델을 로딩하여 가장 유사한 단어를 출력

loaded_model = KeyedVectors.load_word2vec_format("./data/word2vec")
# 모델 로드
print(loaded_model.vectors.shape)
print(loaded_model.most_similar("최민식", topn=5))
print(loaded_model.most_similar("남대문", topn=5))
print(loaded_model.similarity("헐크", '아이언맨'))
print(loaded_model.most_similar(positive=['어벤져스', '아이언맨'], negative=['스파이더맨'], topn=1))
