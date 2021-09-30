import torch
import random
import numpy as np
from torchtext.legacy import data
from transformers import BertModel
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from tqdm import tqdm

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#랜덤시드 고정

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#bert 라이브러리에서 제공하는 토크나이저 사용

tokens = tokenizer.tokenize('우리 사이엔 낮은 담이 있어 서로의 진심을 안을 수가 없어요')
print(tokens)

indexes = tokenizer.convert_tokens_to_ids(tokens)
print(indexes)

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = tokenizer.max_model_input_sizes['bert-base-multilingual-cased']
print(max_input_length)

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

TEXT = data.Field(batch_first = True,
                 use_vocab = False,
                 tokenize = tokenize_and_cut,
                 preprocessing = tokenizer.convert_tokens_to_ids,
                 init_token = init_token_idx,
                 eos_token = eos_token_idx,
                 pad_token = pad_token_idx,
                 unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)

fields = {'text': ('text',TEXT), 'label': ('label',LABEL)}

train_data, test_data = data.TabularDataset.splits(
    path='C:/Users/xiu03/lab/korean-pytorch-sentiment-analysis/data',
    train='train_data.csv',
    test='test_data.csv',
    format='csv',
    fields=fields,
)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print(f'훈련 데이터 갯수: {len(train_data)}')
print(f'검증 데이터 갯수: {len(valid_data)}')
print(f'테스트 데이터 갯수: {len(test_data)}')

print(vars(train_data.examples[10]))
tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[10])['text'])
print(tokens)

string = tokenizer.convert_tokens_to_string(tokens)
print(string)

LABEL.build_vocab(train_data)
#라벨 단어장

print(LABEL.vocab.stoi)

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True,
    device = device)

#모델 생성

bert = BertModel.from_pretrained('bert-base-multilingual-cased')
# 사전 훈련된 모델 - bert는 위키피디아 같은 자료로 미리 학습한 모델을 제공한다.

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional
                             else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch_size, sent_len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
            # embedded = [batch_size, sent_len, emb_dim]

        _, hidden = self.rnn(embedded)
        # hideen = [n_layers * n_directions, batch_size, emb_dim]

        if self.rnn.bidirectional:
            # 마지막 레이어의 양방향 히든 벡터만 가져옴
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        # hidden = [batch_size, hid_dim]

        output = self.out(hidden)
        # output = [batch_size, out_dim]

        return output


HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM,
                        N_LAYERS, BIDIRECTIONAL, DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'모델의 파라미터 수는 {count_parameters(model):,}, 이 중 버트 모델의 파라미터 수는 {count_parameters(bert):,}개입니다.')

for name, param in model.named_parameters():

    if name.startswith('bert'):
        param.requires_grad = False

# 버트 모델의 파라미터는 훈련시키면 안됨. 따라서 모델의 파라미터는 동결

print(f'모델의 파라미터 수는 {count_parameters(model):,}개입니다.')

for name, param in model.named_parameters():

    if param.requires_grad == True:
        print(name)

# 모델 구조 확인

print('aaa')
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
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


#훈련시작

N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):
    print('aaa')
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

model.load_state_dict(torch.load('tut6-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

# 추가훈련

N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):
    print('aaa')
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch + 6:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


review_data = pd.read_csv('C:/Users/xiu03/lab/184318_data.csv', encoding='utf-8')
sentence = review_data['document']

output = open('기계학습_감정분석 결과3 -bert.txt', 'w', encoding='utf-8')

print('결과값 도출 시작')

for sen in sentence:
    output.write("평가 문장 : "+sen+'\n')
    output.write("결과 : "+str(predict_sentiment(model, sen))+'\n')
