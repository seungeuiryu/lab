from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
corpus_fname = './data/corpus_mecab.txt'
model_fname = './data/word2vec'


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

print('corpus 생성')
corpus = [sent.strip().split(" ") for sent in tqdm(open(corpus_fname, 'r', encoding='utf-8').readlines())]

print("학습 중")
model = Word2Vec(
    corpus,
    vector_size=100,
    workers=4,
    sg=1,
    compute_loss=True,
    min_count=5,
    callbacks=[callback()])
model.wv.save_word2vec_format(model_fname)
print('완료')
