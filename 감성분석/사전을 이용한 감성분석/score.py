import json


class KnuSL():

    def data_list(wordname):
        with open('/Users/xiu0327/lab/감성분석/KnuSentiLex/KnuSentiLex/data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
            data = json.load(f)
        result = ['None', 'None']
        for i in range(0, len(data)):
            if data[i]['word'] == wordname or data[i]['word_root'] == wordname:
                result.pop()
                result.pop()
                result.append(data[i]['word_root'])
                result.append(data[i]['polarity'])

        r_word = result[0]
        s_word = result[1]

        return s_word