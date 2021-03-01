import os
import math
import utils
import gensim
import pickle
import logging
import warnings
import winsound
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt

logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_words(file):
    with open(file, 'r') as f_in:
        return f_in.read().split()


def get_corpus(dir, load_bigrams=False, load_dict=False):
    if not load_bigrams:
        words = utils.run_parrlell_on_dir(dir, get_words)
        bigram_mod = bigrams(words)
        bigram = [bigram_mod[article] for article in words]
        with open('bigrams.pic', 'wb') as f:
            pickle.dump(bigram, f)
    else:
        with open('bigrams.pic', 'rb') as f:
            bigram = pickle.load(f)
    if not load_dict:
        id2word = gensim.corpora.Dictionary(bigram)
        id2word.filter_extremes(no_below=3, no_above=0.35)
    else:
        id2word = gensim.corpora.Dictionary().load("id2word.dict")

    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram


if __name__ == '__main__':
    train_corpus, train_id2word, bigram_train = get_corpus("CleanedText_Costum", load_bigrams=True, load_dict=True)
    corpus = pd.read_json("corpus.json")
    train_corpus = [train_id2word.doc2bow(text) for text in corpus['words']]
    print('Corpus and dict loaded')

    num_topics = [10, 15, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    alphas = [0.2, 0.5, 0.8]
    for topics in num_topics:
        for alpha in alphas:
            model_name = f"lda_{topics}_topics_{alpha} _alpha"
            model_folder = f"LDA/{model_name}"
            os.mkdir(model_folder)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lda_train = gensim.models.ldamulticore.LdaMulticore(
                    corpus=train_corpus,
                    num_topics=topics,
                    id2word=train_id2word,
                    chunksize=100,
                    workers=8,
                    passes=50,
                    eval_every=1,
                    minimum_probability=0.0,
                    eta='auto',
                    alpha=np.asarray([alpha for _ in range(topics)]),
                    per_word_topics=True)
                lda_train.save(f'{model_folder}/model')
            print(f"Done Training {model_name}")

            fig, axes = plt.subplots(math.ceil(topics / 3), 3, figsize=(40, 40),
                                     sharex=True, sharey=True)

            for topic_id, ax in enumerate(axes.flatten()):
                if topic_id >= topics: break
                fig.add_subplot(ax)
                plt.gca().imshow(WordCloud(mode='RGBA', background_color=None, max_words=20)
                                 .fit_words(dict(lda_train.show_topic(topic_id, 20))))
                plt.gca().axis("off")
                plt.gca().set_title(f"Topic #{topic_id}")
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            plt.savefig(f"{model_folder}/all_word_clouds.jpg")
            plt.close()

            for topic_id in range(topics):
                topic_folder = f"{model_folder}/topic_{topic_id}"
                os.mkdir(topic_folder)
                cloud = WordCloud(mode='RGBA', background_color=None, max_words=20) \
                    .fit_words(dict(lda_train.show_topic(topic_id, 20)))
                cloud.to_file(f"{topic_folder}/cloud.png")
