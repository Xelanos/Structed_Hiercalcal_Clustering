import utils
import gensim
import pickle
import logging
import warnings
import winsound
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_words(file):
    with open(file, 'r') as f_in:
        return f_in.read().split()


def get_corpus(dir, load_bigrams=False):
    if not load_bigrams:
        words = utils.run_parrlell_on_dir(dir, get_words)
        bigram_mod = bigrams(words)
        bigram = [bigram_mod[article] for article in words]
        with open('bigrams.pic', 'wb') as f: pickle.dump(bigram, f)
    else:
        with open('bigrams.pic', 'rb') as f: bigram = pickle.load(f)
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram


if __name__ == '__main__':
    train_corpus, train_id2word, bigram_train = get_corpus("CleanedText_Costum", load_bigrams=True)
    print('Corpus and dict loaded')
    topics = 31
    alpha = 0.8
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #     lda_train = gensim.models.ldamulticore.LdaMulticore(
    #         corpus=train_corpus,
    #         num_topics=topics,
    #         id2word=train_id2word,
    #         chunksize=100,
    #         workers=8,
    #         passes=50,
    #         eval_every=1,
    #         eta='auto',
    #         alpha=np.asarray([alpha for _ in range(topics)]),
    #         per_word_topics=True)
    #     lda_train.save(f'saved_lda/lda_train_alpha_{alpha}_auto_eta_{topics}-topics.model')
    #     winsound.PlaySound('c:/windows/media/tada.wav', winsound.SND_FILENAME)
    # print("Done Training lda")
    lda_train = gensim.models.hdpmodel.HdpModel(train_corpus, train_id2word, alpha=1)
    lda_train.save(f'hdpsave/first.model')
    winsound.PlaySound('c:/windows/media/tada.wav', winsound.SND_FILENAME)
    print("Done Training hdp")
