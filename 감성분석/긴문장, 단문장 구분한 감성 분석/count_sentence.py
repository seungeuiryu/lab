short_sentence = open('ratings_train_short.txt', 'w', encoding='utf-8')
long_sentence = open('ratings_train_long.txt', 'w', encoding='utf-8')

short_sentence.write("document	label"+"\n")
long_sentence.write("document	label"+"\n")

count = 20

with open('ratings_train.txt', 'r', encoding='utf-8') as input_file:
    next(input_file)
    for line in input_file:
        _, sentence, label = line.strip().split('\t')
        if not sentence: continue
        if len(sentence) > count:
            long_sentence.write(sentence+"  "+label+"\n")
        else:
            short_sentence.write(sentence+"  "+label+"\n")
