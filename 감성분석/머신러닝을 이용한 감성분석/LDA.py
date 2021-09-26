import re # 정규표현식 패키지
import tomotopy as tp # 토픽 모델링에 사용할 패키지
from kiwipiepy import Kiwi #형태소 분석에 사용할 패키지
from pyvis.network import Network # 네트워크 시각화에 사용할 패키지

try:
    # 이미 전처리된 코퍼스가 있으면 불러온다.
    corpus = tp.utils.Corpus.load('k.cps')
except IOError:
    # 없으면 전처리를 시작한다.
    kiwi = Kiwi()
    kiwi.prepare()

    # 형태소 분석 후 사용할 태그
    pat_tag = re.compile('NN[GP]|V[VA]|MAG|MM')


    def tokenizer(raw, user_data):
        res, _ = user_data()[0]
        for w, tag, start, l in res:
            if not pat_tag.match(tag) or len(w) <= 1: continue
            yield w + ('다' if tag.startswith('V') else ''), start, l


    corpus = tp.utils.Corpus(
        tokenizer=tokenizer
    )
    # 입력 파일에는 한 줄에 문헌 하나씩이 들어가 있습니다.
    corpus.process((line, kiwi.async_analyze(line)) for line in open('/Users/xiu0327/lab/감성분석/data/38766_output.txt', encoding='utf-8'))
    # 전처리한 코퍼스를 저장한다.
    corpus.save('k.cps')

# 최소 10개 이상 문헌에 등장하고, 전체 출현빈도는 20 이상인 단어만 사용합니다.
# 그리고 상위 10개 고빈도 단어는 분석에서 제외하구요
# 주제 개수는 40개입니다.
mdl = tp.CTModel(tw=tp.TermWeight.ONE, min_df=10, min_cf=20, rm_top=10, k=40, corpus=corpus)
mdl.train(0)

# 문헌 수가 만 개 이상이라면 num_beta_sample을 1~5 정도로 줄여도 됨
# 수 천개라면 최소 10 정도, 수 백개에 불과하다면 20 이상으로 키우는걸 권장
mdl.num_beta_sample = 1
print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))
print('Removed Top words: ', *mdl.removed_top_words)

# 총 1000회 깁스샘플링을 반복합니다.
# 적절한 반복회수는 데이터에 따라 다릅니다.
# ll_per_word 값의 증가가 둔화되거나 멈추는 지점까지만
# 반복하는걸 추천합니다.
for i in range(0, 1000, 20):
    print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))

# 학습된 결과를 시각화 합니다.
g = Network(width=800, height=800, font_color="#333")
correl = mdl.get_correlations().reshape([-1])
correl.sort()

# 상관계수 상위 10%만 간선으로 잇습니다.
top_tenth = mdl.k * (mdl.k - 1) // 10
top_tenth = correl[-mdl.k - top_tenth]

topic_counts = mdl.get_count_by_topics()

for k in range(mdl.k):
    label = "#{}".format(k)
    title = ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=8))
    print('Topic', label, title)
    label += '\n' + ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=3))
    g.add_node(k, label=label, title=title, shape='ellipse', value=float(topic_counts[k]))
    for l, correlation in zip(range(k - 1), mdl.get_correlations(k)):
        if correlation < top_tenth: continue
        g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))

g.barnes_hut(gravity=-1000, spring_length=20)
g.show_buttons()
# 시각화 파일이 topic_network.html 이라는 이름으로 저장됩니다.
# 웹 브라우저로 열어서 확인해보세요.
g.show("topic_network.html")

