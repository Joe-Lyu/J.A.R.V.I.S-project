import re
import sys
import numpy as np
import datetime
from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_PATH = sys.path[0]
PAD_ID=0
# print(sys.path[0])

def load_dataset(filename):
    df = pd.read_csv(filename, encoding="latin1", names=["question", "answer"])
    questions = list(df["question"])
    answers = list(df["answer"])

    return questions, answers
# questions, answers = load_dataset(PROJECT_PATH + r"\s2s.csv")


def get_preprocessed_info(sentences, flag=0):  # sentence List->tokenized sentence List
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ",s)  # substitute none alphabet and num into space (sentences by sentences)
        clean = re.sub(r' b-z ', " ", s)
        w = word_tokenize(clean)
        # stemming
        tokenized_word = [i.lower() for i in w]
        if flag == 1:
            tokenized_word.insert(0, "<go>")
            tokenized_word.append("<eos>")
        words.append(tokenized_word)
    token = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\]^_`{|}~')
    token.fit_on_texts(words)
    return token, words


def get_max_length(wordsList):  # longest sentence word-count
    return [len(max(words, key=len)) for words in wordsList]


def padding_doc(encoded_doc, max_length):
    return pad_sequences(encoded_doc, maxlen=max_length, padding="post")


class Data:
    def __init__(self):
        # load dataset
        questions, answers = load_dataset(PROJECT_PATH + r"\s2s.csv")
        # get info and save to 2 dicts(v2i i2v)and a vocab
        token_q, words_q = get_preprocessed_info(questions)
        token_a, words_a = get_preprocessed_info(answers, flag=1)
        token_all, words_all = get_preprocessed_info(questions + answers, flag=1)
        # encoding+padding NOTE:No need! this is the advantage of seq2seq! BUT for some reason the ragged ndarray won't work well
        max_length_q, max_length_a = get_max_length([words_q, words_a])
        enc_pad_words_q= padding_doc(token_all.texts_to_sequences(words_q), max_length_q)
        enc_pad_words_a = padding_doc(token_all.texts_to_sequences(words_a), max_length_a)
        self.v2i = token_all.word_index  # form vocab_dict
        self.i2v = token_all.index_word  # key and value exchanged num_dict
        self.vocab = self.i2v.values()  # form vocab list FIXME:from answers
        self.x, self.y = enc_pad_words_q, enc_pad_words_a
        self.x, self.y = np.array(self.x), np.array(self.y)  # dataset forming
        self.start_token = self.v2i["<go>"]
        self.end_token = self.v2i["<eos>"]

    def sample(self, n=64):  # get batch
        bi = np.random.randint(0, len(self.x), size=n)
        bx, by = self.x[bi], self.y[bi]
        decoder_len = np.full((len(bx),), by.shape[1] - 1, dtype=np.int32)
        return bx, by, decoder_len

    def idx2str(self, idx):
        x = []
        for i in idx:
            if i==PAD_ID:
                continue
            x.append(self.i2v[i])
            if i == self.end_token:
                break
        return " ".join(x)

    @property
    def num_word(self):
        return len(self.vocab)
