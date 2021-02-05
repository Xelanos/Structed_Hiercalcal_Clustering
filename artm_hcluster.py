import artm
import utils
import pickle
import gensim
import numpy as np
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


def get_bigrams(words, bi_min=5, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_words(file):
    with open(file, 'r') as f_in:
        return f_in.read().split()

def get_whole_file(file):
    with open(file, 'r') as f_in:
        return f_in.read()

def make_batches(dir, load_bigrams=False):
    texts = utils.run_parrlell_on_dir(dir, get_words)
    if not load_bigrams:
        words = utils.run_parrlell_on_dir(dir, get_words)
        bigram_mod = get_bigrams(words)
        bigram = [bigram_mod[article] for article in words]
        with open('bigrams.pic', 'wb') as f: pickle.dump(bigram, f)
    else:
        with open('bigrams.pic', 'rb') as f: bigram = pickle.load(f)
    print('Finished bigrams')

    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()

    texts = utils.run_parrlell_on_dir(dir, get_whole_file)
    cv = CountVectorizer(vocabulary=id2word.token2id)
    n_wd = np.array(cv.fit_transform(texts).todense()).T
    vocabulary = cv.get_feature_names()
    batch_vectorizer = artm.BatchVectorizer(data_format='bow_n_wd',
                                            n_wd=n_wd,
                                            vocabulary=vocabulary,
                                            target_folder='bc')
    return n_wd, vocabulary



if __name__ == '__main__':
    # n_wd, vocabulary = make_batches("CleanedText_Costum", load_bigrams=True)
    bv = artm.BatchVectorizer(data_path='bc', data_format='batches')
    dictionary = artm.Dictionary()
    dictionary.gather(data_path='bc')
    scores = [artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
                                   artm.SparsityPhiScore(name='SparsityPhiScore'),
                                   artm.SparsityThetaScore(name='SparsityThetaScore'),
                                   artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3),
                                   artm.TopTokensScore(name='TopTokensScore', num_tokens=20)]
    regularizers = [artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.4),
                    artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=2.5e+5)]

    # model_artm = artm.ARTM(num_topics=30, cache_theta=True,
    #                        scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
    #                                artm.SparsityPhiScore(name='SparsityPhiScore'),
    #                                artm.SparsityThetaScore(name='SparsityThetaScore'),
    #                                artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3),
    #                                artm.TopTokensScore(name='TopTokensScore', num_tokens=8)],
    #                        regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.4),
    #                                      artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=2.5e+5)])
    #
    # model_artm.num_document_passes = 4
    # model_artm.initialize(dictionary)
    # model_artm.fit_offline(batch_vectorizer=bv, num_collection_passes=20)

    hATM_model = artm.hARTM(scores=scores, regularizers=regularizers)
    level0 = hATM_model.add_level(num_topics=10)
    level0.initialize(dictionary)
    level0.fit_offline(batch_vectorizer=bv, num_collection_passes=20)


    level1 = hATM_model.add_level(num_topics=20, topic_names=['sub_topic_' + str(i) for i in range(20)])
    level1.initialize(dictionary)
    level1.fit_offline(batch_vectorizer=bv, num_collection_passes=20)


    level2 = hATM_model.add_level(num_topics=40, topic_names=['sub_sub_topic_' + str(i) for i in range(40)])
    level2.initialize(dictionary)
    level2.fit_offline(batch_vectorizer=bv, num_collection_passes=20)


    print('ho')

    words = {}
    for i,topic_name in enumerate(level0.topic_names):
        print(f"{topic_name} : {level0.score_tracker['TopTokensScore'].last_tokens[topic_name]}")
        words[i] = {level0.score_tracker['TopTokensScore'].last_tokens[topic_name][i] : level0.score_tracker['TopTokensScore'].last_weights[topic_name][i]
                  for i in range(20)}


    fig, axes = plt.subplots(5, 2, figsize=(40, 40),
                             sharex=True, sharey=True)
    for topic_id, ax in enumerate(axes.flatten()):
        if topic_id >= 10: break
        fig.add_subplot(ax)
        plt.gca().imshow(WordCloud(mode='RGBA', background_color=None, max_words=20)
                         .fit_words(words[topic_id]))
        plt.gca().axis("off")
        plt.gca().set_title(f"Topic #{topic_id}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    print("\n")

    words = {}
    for i,topic_name in enumerate(level1.topic_names):
        print(f"{topic_name} : {level1.score_tracker['TopTokensScore'].last_tokens[topic_name]}")
        words[i] = {level1.score_tracker['TopTokensScore'].last_tokens[topic_name][i] : level1.score_tracker['TopTokensScore'].last_weights[topic_name][i]
                  for i in range(20)}


    fig, axes = plt.subplots(5, 4, figsize=(40, 40),
                             sharex=True, sharey=True)
    for topic_id, ax in enumerate(axes.flatten()):
        if topic_id >= 20: break
        fig.add_subplot(ax)
        plt.gca().imshow(WordCloud(mode='RGBA', background_color=None, max_words=20)
                         .fit_words(words[topic_id]))
        plt.gca().axis("off")
        plt.gca().set_title(f"Sub Topic #{topic_id}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()



    words = {}
    for i,topic_name in enumerate(level2.topic_names):
        print(f"{topic_name} : {level2.score_tracker['TopTokensScore'].last_tokens[topic_name]}")
        words[i] = {level2.score_tracker['TopTokensScore'].last_tokens[topic_name][i] : level2.score_tracker['TopTokensScore'].last_weights[topic_name][i]
                  for i in range(20)}


    fig, axes = plt.subplots(10, 4, figsize=(40, 40),
                             sharex=True, sharey=True)
    for topic_id, ax in enumerate(axes.flatten()):
        if topic_id >= 40: break
        fig.add_subplot(ax)
        plt.gca().imshow(WordCloud(mode='RGBA', background_color=None, max_words=20)
                         .fit_words(words[topic_id]))
        plt.gca().axis("off")
        plt.gca().set_title(f"Sub Topic #{topic_id}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    print('g')
