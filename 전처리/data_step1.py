import json, glob


def process_nsmc(corpus_path, output_fname, with_label=True):
    with open(corpus_path, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        next(f1)  # skip head line
        for line in f1:
            #_, sentence = line.strip().split(',')
            sentence = line.replace('\n', ' ')
            if not sentence: continue
            if with_label:
                #f2.writelines(sentence + "\u241E" + label + "\n")
                f2.writelines(sentence + "\u241E" + "\n")
            else:
                f2.writelines(sentence + "\n")


corpus_path = '/Users/xiu0327/lab/data/blog.txt'
output_fname = './processed_blog.txt'
process_nsmc(corpus_path, output_fname, False)
