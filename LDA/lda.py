"""
Author: Sumin Lim (KAIST)
Description: This file implements LDA model and saves result file. Users should prepare tokenized 
text file as input file of this program.
Usage: python lda.py -tkf tokenized_filename
"""
import argparse
import pandas as pd 
import pickle as pkl
from tqdm import tqdm
from itertools import chain
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

import matplotlib.pyplot as plt

class LDA:
    def __init__(self, tokenized_file):
        self.tokenized = tokenized_file

    def read_data(self):
        print("Read Data ... ")
        tokenized_words = []
        split_lines = []
        with open(self.tokenized, "r", encoding='utf-8') as f:
            for line in f:
                tokenized_words.append(line.strip())
                split_lines.append([x for x in line.strip().split()])

        return tokenized_words, split_lines

    def get_tf(self):
        print("Get term frequency ... ")
        term_count = Counter(chain.from_iterable(self.document_split))
        df_idf = pd.DataFrame(term_count.items(), columns=["Term", "Freq"])
        return df_idf

    def get_tfidf_score(self):
        print("Get Tf-idf Score ... ")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.document)
        self.X = X

        # Tf-idf Full Matrix (sparse)
        df_full = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

        # Tf-idf matrix with row format of document_id, token_id, tf-idf score
        doc_size = X.shape[0]; term_size = X.shape[1]
        terms = vectorizer.get_feature_names()
        doc_id = []; term_id = []; score = []

        for doc in tqdm(range(doc_size)):
            for term in range(term_size):
                if X[doc, term] != 0:
                    doc_id.append(doc)
                    term_id.append(terms[term])
                    score.append(X[doc, term])

        df_tfidf = pd.DataFrame({"Document": doc_id, 
                                 "Term": term_id,
                                 "Score": score})

        return df_full, df_tfidf

    def lda(self, is_graph):
        print("Analyze LDA ... ")
        alpha = input("Please enter the alpha: ")
        iterations = int(input("Please enter the number of iterations: "))
        is_tfidf = input("Using Tf-idf (y/n): ")

        id2word = corpora.Dictionary(self.document_split)
        corpus = [id2word.doc2bow(text) for text in self.document_split]

        if is_tfidf == "y":
            tfidf = gensim.models.TfidfModel(corpus)
            corpus = tfidf[corpus]


        if is_graph == "y":
            if alpha == "auto":
                alpha = "asymmetric"
            perplexities = []; coherences = []

            start = int(input("Please enter the starting number of topic: "))
            end = int(input("Please enter the end number of topic: "))
            step = int(input("Please enter the step to increase: "))

            for num_topic in tqdm(range(start, end+1, step)):
                lda_model = LdaMulticore(corpus=corpus,
                                         num_topics=num_topic,
                                         id2word=id2word,
                                         chunksize=100,
                                         alpha=alpha,
                                         iterations=iterations,
                                         per_word_topics=True)

                perplexities.append(lda_model.log_perplexity(corpus))

                coherence = CoherenceModel(model=lda_model,
                                           texts=self.document_split,
                                           dictionary=id2word,
                                           coherence="c_v")
                coherences.append(coherence.get_coherence())

            x_topic = range(start, end+1, step)
            plt.plot(x_topic, perplexities)
            plt.xlabel("The Number of Topics")
            plt.ylabel("Log Perplexity")
            plt.savefig("log_perplexities_"+"from_"+str(start)+"_to_"+str(end+1)+".png", bbox_inches="tight")
            plt.show()
            plt.clf()

            plt.plot(x_topic, coherences)
            plt.xlabel("The Number of Topics")
            plt.ylabel("Coherence Score")
            plt.savefig("coherence_score_"+"from_"+str(start)+"_to_"+str(end+1)+".png", bbox_inches="tight")
            plt.show()
            plt.close()

        elif is_graph == "n":
            num_topics = input("Please enter the number of topics: ")
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=id2word,
                                                        num_topics=num_topics,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=100,
                                                        alpha=alpha,
                                                        iterations=iterations,
                                                        per_word_topics=True)

            # Perplexity and Coherence Score
            print("Get log perplexity and coherence score ... ")
            print("\nPerplexity: ", lda_model.log_perplexity(corpus))

            coherence_model_lda = CoherenceModel(model=lda_model, 
                                                 texts=self.document_split, 
                                                 dictionary=id2word,
                                                 coherence="c_v")
            coherence_lda = coherence_model_lda.get_coherence()
            print("\nCoherence Score: ", coherence_lda)

            # Save LDA, corpus, dictionary for visualizing
            print("Save LDA model, corpus, dictionary for future visualizing ... ")
            with open("corpus", "wb") as f:
                pkl.dump(corpus, f)

            with open("dictionary", "wb") as f:
                pkl.dump(id2word, f)

            with open("lda_model", "wb") as f:
                pkl.dump(lda_model, f)

            doc_lda = lda_model[corpus]

            # Get topic words 
            print("Get topic words ... ")
            columns = []
            for k, v in id2word.iteritems():
                columns.append(v)

            df_topic = []
            topics = lda_model.get_topics()
            for idx, topic in enumerate(topics):
                temp = list([v, topic[k]] for k, v in zip(id2word.keys(), id2word.values()))
                df_temp = pd.DataFrame(temp, columns=["Topic"+str(idx+1)+"_term",
                                                      "Topic"+str(idx+1)+"_weight"])
                df_temp = df_temp.sort_values(by=["Topic"+str(idx+1)+"_weight"], ascending=False)
                df_temp = df_temp.reset_index(drop=True)
                df_topic.append(df_temp)

            df_topic = pd.concat(df_topic, axis=1)

            # Get Document-Topic Distribution
            print("Get Document-Topic Distribution ... ")
            doc_topic = lda_model.get_document_topics(doc_lda)
            dict_doc_topic = {}
            for idx, doc in enumerate(doc_topic):
                dict_doc_topic[idx] = [x[1] for x in doc]

            df_doc_topic = pd.DataFrame(dict_doc_topic).transpose()
            df_doc_topic.rename(columns={col:"Topic-"+str(col+1) for col in df_doc_topic.columns}, inplace=True)
            df_doc_topic.rename(index={x:"Document"+str(x+1) for x in df_doc_topic.index}, inplace=True)

            print("Get topic weight ... ")
            df_topic_weight = pd.DataFrame(df_doc_topic.sum(axis=0)).reset_index()
            df_topic_weight.rename(columns={"index":"Topic",
                                            0: "Weight (Sum)"}, inplace=True)
            weight_sum = sum(df_topic_weight["Weight (Sum)"])
            df_topic_weight["Weight (%)"] = df_topic_weight["Weight (Sum)"].apply(lambda x: x/weight_sum * 100)
            df_topic_weight["Rank"] = df_topic_weight["Weight (%)"].rank(axis=0, ascending=False)

            return df_topic, df_doc_topic, df_topic_weight

    def similarity(self):
        print("Get cosine similarity ... ")
        cos_sim = linear_kernel(self.X, self.X)
        df_sim = pd.DataFrame(cos_sim)
        df_sim.rename(columns={col:"Document"+str(col) for col in df_sim.columns}, inplace=True)
        df_sim.rename(index={col:"Document"+str(col) for col in df_sim.index}, inplace=True)
        return df_sim

    def main(self):
        self.document, self.document_split = self.read_data()
        self.df_tf = self.get_tf()
        self.tfidf_sparse, self.tfidf_dense = self.get_tfidf_score()
        self.df_sim = self.similarity()

        is_graph = input("Please enter whether you want perplexity and coherence graph or not (y/n): ")
        if is_graph == "n":
            self.df_topic, self.df_doc_topic, self.df_topic_weight = self.lda(is_graph)

            print("Save result file ... ")
            with pd.ExcelWriter("LDA_result.xlsx") as writer:
                self.df_tf.to_excel(writer, sheet_name="TF", index=False, encoding="utf-8")
                self.tfidf_sparse.to_excel(writer, sheet_name="TFIDF_sparse", index=False, encoding="utf-8")
                self.tfidf_dense.to_excel(writer, sheet_name="TFIDF_dense", index=False, encoding="utf-8")
                self.df_topic.to_excel(writer, sheet_name="Topic-Keyword", index=False, encoding="utf-8")
                self.df_doc_topic.to_excel(writer, sheet_name="Topic-Document", encoding="utf-8")
                self.df_topic_weight.to_excel(writer, sheet_name="Topic-Weight", index=False, encoding="utf-8")
                self.df_sim.to_excel(writer, sheet_name="Document-Similarity", encoding="utf-8")
        elif is_graph == "y":
            self.lda(is_graph)

        print("All work is done. Bye!")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tkf", "--tokenizedFileName", required=True, nargs=1, type=str)
    args = parser.parse_args()
    lda = LDA(args.tokenizedFileName[0])

    lda.main()
