import torch
from torchtext.legacy import data
from torchtext import datasets
from konlpy.tag import Mecab
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pandas as pd

mecab = Mecab('C:/mecab/mecab-ko-dic')

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize=mecab.morphs, preprocessing=generate_bigrams)
# 토크나이저 : mecab , 전처리 : bi_gram
LABEL = data.LabelField(dtype=torch.float)
# field 형식 = {csv 컬럼명 : (데이터 컬럼명, field 이름)}

fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
# dictionary 형식은 {csv컬럼명 : (데이터 컬럼명, Field이름)}

train_data, test_data = data.TabularDataset.splits(
    path='C:/Users/xiu03/lab/korean-pytorch-sentiment-analysis/data',
    train='train_data.csv',
    test='test_data.csv',
    format='csv',
    fields=fields,
)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors = 'fasttext.simple.300d',
                 unk_init = torch.Tensor.normal_
                 )

LABEL.build_vocab(train_data)

print(len(TEXT.vocab))
# 단어 개수
print(TEXT.vocab.itos[:5])
# 단어장에 있는 단어를 5개만 출력
print(LABEL.vocab.stoi)
# 라벨의 데이터 집합과 그와 매핑된 고유 정수 출력
print(vars(train_data.examples[15]))
# 사전화된 훈련 데이터가 어떻게 되어있는지 예시로 출력

# 데이터 생성자 만들기 (단어를 수치화 한다는 뜻)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)

print(next(iter(train_iterator)).text)
# 수치화된 텐서를 출력

print(TEXT.vocab.itos[2533], TEXT.vocab.itos[54], TEXT.vocab.itos[2647])
print(TEXT.vocab.itos[14207], TEXT.vocab.itos[14207], TEXT.vocab.itos[556])


def print_shape(name, data):
    print(f'{name} has shape {data.shape}')


print(nn.functional.avg_pool2d)
txt = torch.rand(2, 5, 10)
print(txt.shape, F.avg_pool2d(txt, (5, 1)).shape)
# (5x1) 필터로 평균 구하는 과정

txt = torch.tensor(
    [[[1, 2, 3, 4], [4, 5, 6, 7]]], dtype=torch.float
)
print(txt.shape, "\n", txt)

print(F.avg_pool2d(txt, (2, 1)).shape, F.avg_pool2d(txt, (2, 1)))
# (2x1) 필터로 평균 구하는 과정
print(F.avg_pool2d(txt, (2, 2)).shape, F.avg_pool2d(txt, (2, 2)))
# (2x2) 필터로 평균 구하는 과정


class FastText(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):

        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        res = self.fc(pooled)
        return res



INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'모델의 파라미터 수는 {count_parameters(model):,} 개 입니다.')

pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
print(model.embedding.weight.data.shape)
print(model.embedding.weight.data.copy_(pretrained_embeddings)) # copy_ 메서드는 인수를 현재 모델의 웨이트에 복사함)
# 앞서 학습된 festtext 모델의 단어를 임베딩 레이어에 복사

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)  # output_dim = 1
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    print(f'Epoch: {epoch + 6:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

def predict_sentiment(model, sentence): #예측 함수
    model.eval()
    tokenized = [tok for tok in mecab.morphs(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1) # 배치
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

review_data = pd.read_csv('C:/Users/xiu03/lab/184318_data.csv', encoding='utf-8')
sentence = review_data['document']

output = open('기계학습_감정분석 결과2.txt', 'w', encoding='utf-8')

print('결과값 도출 시작')

for sen in sentence:
    output.write("평가 문장 : "+sen+'\n')
    output.write("결과 : "+str(predict_sentiment(model, sen))+'\n')
