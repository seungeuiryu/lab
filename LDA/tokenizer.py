# -*- coding: utf-8 -*-
"""
Author: Sumin Lim (KAIST)
Desciption: This file tokenize document with nouns and save it as text file and modified excel file with new column, nouns 
Usage: python tokenizer.py -tk Okt -if blockchain_patent.xlsx -of test -c 요약
-tk: tokenizer name, it should be one of Hannanum, Komoran, Kkma, Okt
-if: input file name with its extension 
-of: output file name without its extension
-c: column name to analyze in the input file (ex. 요약 column in blockchain_patent.xlsx)
"""

import argparse
import pandas as pd
from itertools import chain
from collections import Counter
from tqdm import tqdm 

import gensim
import gensim.corpora as corpora

from konlpy.tag import Hannanum, Komoran, Kkma, Okt, Mecab
from sklearn.feature_extraction.text import TfidfVectorizer

class Tokenize:
    def __init__(self, tokenizer, input_filename, output_filename, colname):
        self.tokenizer = tokenizer
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.colname = colname

    def main(self):
        if self.tokenizer == "Twitter" or self.tokenizer == "Okt":
            tokenizer = Okt()
        elif self.tokenizer == "Hannanum":
            tokenizer = Hannanum()
        elif self.tokenizer == "Komoran":
            tokenizer = Komoran()
        elif self.tokenizer == "Kkma":
            tokenizer = Kkma()
        elif self.tokenizer == "mecab":
            tokenizer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

        # Read file
        df = pd.read_excel(self.input_filename)

        # tokenizing
        self.nouns = []
        for row in tqdm(df[self.colname]):
            self.nouns.append(tokenizer.nouns(row))

        # Save tokenized text into text file
        filename = self.output_filename + "_" + self.tokenizer
        with open(filename+".txt", "w", encoding='utf-8') as f:
            for sublist in self.nouns:
                line = " ".join(sublist) + "\n"
                f.write(line)

        df["nouns"] = [" ".join(x) for x in self.nouns]
        return df


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tk", "--tokenizer", default="mecab", required=True, nargs=1, type=str)
    parser.add_argument("-if", "--inputFilename", default="lda_sample.xlsx", required=True, nargs=1, type=str)
    parser.add_argument("-of", "--outputFilename", default="test", required=True, nargs=1, type=str)
    parser.add_argument("-c", "--columnName", default="contents", required=True, nargs=1, type=str)
    args = parser.parse_args()

    test = Tokenize(args.tokenizer[0], 
                    args.inputFilename[0],
                    args.outputFilename[0],
                    args.columnName[0])
    test.main()
