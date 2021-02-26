import os
import artm
import utils
import pickle
import gensim
import numpy as np
import pandas as pd
from scipy.stats import entropy
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
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

def cluster(model, topic_num, quantile):
    theta = model.get_theta()
    most_common = get_most_common_papers(model, topic_num, quantile)
    vectors = theta[[entry[0] for entry in most_common]].transpose()
    km = KMeans(n_clusters=1, init='k-means++')
    km.fit(vectors)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, vectors)
    return vectors.iloc[closest]

def get_most_common_papers(model, topic_num, quantile):
    theta = np.array(model.get_theta('topic_{}'.format(topic_num)).iloc[0])
    idx = np.nonzero(theta)[0]
    articles = zip(idx, theta[idx])
    articles = sorted(articles, key = lambda x: x[1], reverse = True)
    return articles[:get_quantile_idx(quantile, articles)]

def get_quantile_idx(quantile, common_papers):
    df = pd.DataFrame([x[1] for x in common_papers])
    q = df.quantile(quantile)
    return (df < q).idxmax()[0]

def make_batches(dir, load_bigrams=False, load_dict=False):
    if not load_bigrams:
        words = utils.run_parrlell_on_dir(dir, get_words)
        bigram_mod = get_bigrams(words)
        bigram = [bigram_mod[article] for article in words]
        with open('bigrams.pic', 'wb') as f: pickle.dump(bigram, f)
    else:
        with open('bigrams.pic', 'rb') as f: bigram = pickle.load(f)
    print('Finished bigrams')

    if not load_dict:
        id2word = gensim.corpora.Dictionary(bigram)
        id2word.filter_extremes(no_below=3, no_above=0.35)
        id2word.compactify()
        print('Finished id2word')
    else:
        id2word = gensim.corpora.Dictionary().load("id2word.dict")

    print('Finished dict')

    df = pd.read_json("corpus.json")
    texts = [' '.join(doc) for doc in df['words'].to_list()]
    cv = CountVectorizer(vocabulary=id2word.token2id)
    n_wd = np.array(cv.fit_transform(texts).todense()).T
    vocabulary = cv.get_feature_names()
    batch_vectorizer = artm.BatchVectorizer(data_format='bow_n_wd',
                                            n_wd=n_wd,
                                            vocabulary=vocabulary,
                                            target_folder='bc4')
    print('Done vectoring')
    return n_wd, vocabulary



if __name__ == '__main__':
    # n_wd, vocabulary = make_batches("CleanedText_Costum", load_bigrams=True, load_dict=True)
    corpus = pd.read_json("corpus.json")
    bv = artm.BatchVectorizer(data_path='bc4', data_format='batches')
    dictionary = artm.Dictionary()
    dictionary.gather(data_path='bc4')

    level0_num_topics = [33,34,35,36,37,38,39]
    level1_num_topics = [40, 50, 60]

    for num_topics in level0_num_topics:
        for next_level_num_topics in level1_num_topics:

            model_name = f"ARTM_{num_topics}_{next_level_num_topics}"
            model_folder = f"Artm/{model_name}"
            os.mkdir(model_folder)

            scores = [artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary),
                                           artm.SparsityPhiScore(name='SparsityPhiScore'),
                                           artm.SparsityThetaScore(name='SparsityThetaScore'),
                                           artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3),
                                           artm.TopTokensScore(name='TopTokensScore', num_tokens=20)]
            regularizers = [artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.4),
                            artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=2.5e+5)]


            hATM_model = artm.hARTM(scores=scores, regularizers=regularizers, cache_theta=True)
            level0 = hATM_model.add_level(num_topics=num_topics)
            level0.initialize(dictionary)
            level0.fit_offline(batch_vectorizer=bv, num_collection_passes=20)


            level1 = hATM_model.add_level(num_topics=next_level_num_topics, topic_names=['sub_topic_' + str(i) for i in range(next_level_num_topics)],
                                          parent_level_weight=0.75)
            level1.set_parent_model(level0)
            level1.initialize(dictionary)
            level1.fit_offline(batch_vectorizer=bv, num_collection_passes=20)

            level0.save(f"{model_folder}/model", model_name)


            print('done')

            topic_to_subs = pd.DataFrame(level1.get_parent_psi())
            topic_to_subs.columns = ['topic_{}'.format(i) for i in range(num_topics)]
            topic_to_subs.to_csv(f'{model_folder}/topic_to_sub_topic.csv')

            ent = entropy(topic_to_subs)
            with open(f"{model_folder}/entropy.txt", 'w') as f:
                f.write(f"Entropy sum is :{ent.sum()}\n")
                f.write(f"Entropy average is :{np.average(ent)}\n")


            words = {}
            for i,topic_name in enumerate(level0.topic_names):
                print(f"{topic_name} : {level0.score_tracker['TopTokensScore'].last_tokens[topic_name]}")
                words[i] = {level0.score_tracker['TopTokensScore'].last_tokens[topic_name][i] : level0.score_tracker['TopTokensScore'].last_weights[topic_name][i]
                          for i in range(20)}


            fig, axes = plt.subplots(5, 6, figsize=(40, 40),
                                     sharex=True, sharey=True)
            for topic_id, ax in enumerate(axes.flatten()):
                if topic_id >= num_topics: break
                fig.add_subplot(ax)
                plt.gca().imshow(WordCloud(mode='RGBA', background_color=None, max_words=20)
                                 .fit_words(words[topic_id]))
                plt.gca().axis("off")
                plt.gca().set_title(f"Topic #{topic_id}")
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            plt.savefig(f"{model_folder}/all_word_clouds.jpg")

            for topic_id in range(num_topics):
                topic_folder = f"{model_folder}/topic_{topic_id}"
                os.mkdir(topic_folder)
                cloud = WordCloud(mode='RGBA', background_color=None, max_words=20).fit_words(words[topic_id])
                cloud.to_file(f"{topic_folder}/cloud.png")


    print('g')
