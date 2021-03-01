import os
import utils
import pickle
import gensim
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from gensim.test.utils import datapath

def get_words(file):
    with open(file, 'r') as f_in:
        return f_in.read().split()

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


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



def cluster(theta, topic_num, quantile):
    most_common = get_most_common_papers(theta, topic_num, quantile)
    vectors = theta[[entry[0] for entry in most_common]].transpose()
    km = KMeans(n_clusters=1, init='k-means++')
    km.fit(vectors)
    # closest_idx = np.argsort(-1 *km.transform(theta.T).squeeze())
    return km


def get_most_common_papers(theta, topic_num, quantile):
    theta = np.array(theta.iloc[topic_num])
    idx = np.nonzero(theta)[0]
    articles = zip(idx, theta[idx])
    articles = sorted(articles, key=lambda x: x[1], reverse=True)
    return articles[:get_quantile_idx(quantile, articles)]


def get_all_most_common_papers(theta, topic_num):
    theta = np.array(theta.iloc[topic_num])
    idx = np.nonzero(theta)[0]
    articles = zip(idx, theta[idx])
    articles = sorted(articles, key=lambda x: x[1], reverse=True)
    return articles

def get_quantile_idx(quantile, common_papers):
    df = pd.DataFrame([x[1] for x in common_papers])
    q = df.quantile(quantile)
    return (df < q).idxmax()[0]


def get_theta(model, corpus):
    df_data = []
    for i in range(len(corpus)):
        df_data.append([probability[1] for probability in model[corpus[i]][0]])
    return pd.DataFrame(df_data, columns=[f'topic_{i}' for i in range(model.num_topics)]).T



if __name__ == '__main__':
    root = "LDA"
    cluster_quantile = 0.85
    train_corpus, train_id2word, bigram_train = get_corpus("CleanedText_Costum", load_bigrams=True, load_dict=True)
    corpus = pd.read_json("corpus.json")
    train_corpus = [train_id2word.doc2bow(text) for text in corpus['words']]
    for filename in os.listdir(root):
        print(filename)
        model_folder = f'{root}/{filename}'
        model = gensim.models.ldamulticore.LdaMulticore.load(f'{model_folder}/model')
        num_of_topics = model.num_topics
        theta = get_theta(model, train_corpus)
        theta.to_csv(f"{model_folder}/theta.csv")
        for topic_id in range(num_of_topics):
            topic_folder = f'{model_folder}/topic_{topic_id}'
            cl = cluster(theta, topic_id, cluster_quantile)
            top5_idx = np.argsort(cl.transform(theta.T).squeeze())[:5]
            top5 = theta.iloc[:, top5_idx]
            top5_articles = corpus.iloc[top5.columns]
            top5_articles.to_csv(f"{topic_folder}/top5.csv")

            probability = theta.iloc[topic_id].sort_values(ascending=False, ignore_index=True)
            plt.plot(probability)
            plt.title("Topic probability ordinal graph")
            plt.ylabel("Topic probability")
            plt.savefig(f"{topic_folder}/oridinal.png")
            plt.close()

            last_q_index = 0
            all_papers = get_all_most_common_papers(theta, topic_id)
            color = []
            all_quantile_vectors = []
            quantiles = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0.1]
            for q in quantiles:
                q_idx = get_quantile_idx(q/ 10, all_papers)
                quantile_vectors = np.asarray(all_papers[last_q_index: q_idx])
                aritcles_ids = quantile_vectors[:,0]
                distances = cl.transform(theta[aritcles_ids].T).squeeze()
                quantile_vectors[:, 0] = distances
                all_quantile_vectors.append(quantile_vectors)
                color.extend([f'{q *10}%'] * len(aritcles_ids))
                last_q_index = q_idx
            quantile_data = np.concatenate(all_quantile_vectors)
            sns.scatterplot(quantile_data[:, 0], quantile_data[:, 1], hue=color,
                            palette=sns.color_palette("tab10", n_colors=10), alpha=0.5)
            plt.xlabel("Distance from centroid")
            plt.ylabel("Topic probability")
            plt.savefig(f"{topic_folder}/quantiles.png")
            plt.close()

    print('g')
