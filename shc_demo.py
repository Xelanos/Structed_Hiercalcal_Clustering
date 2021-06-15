import os
import tqdm
import math
import pickle
import gensim
import shutil
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from gensim.models.doc2vec import Doc2Vec
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def run_parrlell_on_dir(dir, func):
    files_path = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for file in files:
            files_path.append(f'{dir}\\{file}')
    with mp.Pool(mp.cpu_count() - 1) as pool:
        return list(tqdm.tqdm(pool.imap_unordered(func, files_path), total=len(files_path), desc=func.__name__))

def make_corpus(directory):
    data = {'file_name': [], 'words': []}
    for root, dirs, files in os.walk(directory, topdown=False):
        for file_name in files:
            with open(f'{directory}/{file_name}', 'r') as f:
                data['file_name'].append(file_name)
                data['words'].append(f.read().split())

    df = pd.DataFrame.from_dict(data)
    df.to_json("corpus.json")


def get_words(file):
    with open(file, 'r') as f_in:
        return f_in.read().split()


def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_all_words(dir):
    words = []
    for root, dirs, files in os.walk(dir, topdown=False):
        for file in files:
            with open(f'{dir}\\{file}', 'r') as f:
                words.extend(f.read().split())
    return words


def get_corpus(dir, load_bigrams=False, load_dict=False):

    if not load_bigrams:
        # words = run_parrlell_on_dir(dir, get_words)
        words = get_all_words(dir)
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

def run_lda(corpus_dir, num_topics, alpha):
    train_corpus, train_id2word, bigram_train = get_corpus(corpus_dir)
    corpus = pd.read_json("corpus.json")
    train_corpus = [train_id2word.doc2bow(text) for text in corpus['words']]
    model_name = f"lda_{num_topics}_topics_{alpha}_alpha"
    model_folder = f"Results/LDA_topics/{model_name}"
    os.mkdir(model_folder)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_train = gensim.models.ldamodel.LdaModel(
            corpus=train_corpus,
            num_topics=num_topics,
            id2word=train_id2word,
            chunksize=100,
            passes=50,
            eval_every=1,
            minimum_probability=0.0,
            eta='auto',
            alpha=np.asarray([alpha for _ in range(num_topics)]),
            per_word_topics=True)
        lda_train.save(f'{model_folder}/model')

    fig, axes = plt.subplots(math.ceil(num_topics / 3), 3, figsize=(40, 40),
                             sharex=True, sharey=True)

    for topic_id, ax in enumerate(axes.flatten()):
        if topic_id >= num_topics: break
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

    for topic_id in range(num_topics):
        topic_folder = f"{model_folder}/topic_{topic_id}"
        os.mkdir(topic_folder)
        cloud = WordCloud(mode='RGBA', background_color=None, max_words=20) \
            .fit_words(dict(lda_train.show_topic(topic_id, 20)))
        cloud.to_file(f"{topic_folder}/cloud.png")
    return lda_train, model_name, train_id2word


def get_theta(model, corpus):
    df_data = []
    for i in range(len(corpus)):
        df_data.append([probability[1] for probability in model[corpus[i]][0]])
    return pd.DataFrame(df_data, columns=[f'topic_{i}' for i in range(model.num_topics)]).T



def cluster(theta, topic_num, quantile):
    most_common = get_most_common_papers(theta, topic_num, quantile)
    vectors = theta.iloc[:, [entry[0] for entry in most_common]].transpose()
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




def analyze_lda(model, model_name, id2word):
    cluster_quantile = 0.80
    model_folder = f"Results/LDA_topics/{model_name}"
    corpus = pd.read_json("corpus.json")
    train_corpus = [id2word.doc2bow(text) for text in corpus['words']]
    theta = get_theta(model, train_corpus)
    theta.to_csv(f"{model_folder}/theta.csv")
    theta = pd.read_csv(f"{model_folder}/theta.csv", index_col=0)
    num_of_topics = model.num_topics
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
            q_idx = get_quantile_idx(q / 10, all_papers)
            quantile_vectors = np.asarray(all_papers[last_q_index: q_idx])
            aritcles_ids = quantile_vectors[:, 0]
            distances = cl.transform(theta.iloc[:, aritcles_ids].T).squeeze()
            quantile_vectors[:, 0] = distances
            all_quantile_vectors.append(quantile_vectors)
            color.extend([f'{q * 10}%'] * len(aritcles_ids))
            last_q_index = q_idx
        quantile_data = np.concatenate(all_quantile_vectors)
        sns.scatterplot(quantile_data[:, 0], quantile_data[:, 1], hue=color,
                        palette=sns.color_palette("tab10", n_colors=10), alpha=0.5)
        plt.xlabel("Distance from centroid")
        plt.ylabel("Topic probability")
        plt.savefig(f"{topic_folder}/quantiles.png")
        plt.close()


def doc2vec_vectorize():
    corpus = pd.read_json("corpus.json")
    model = Doc2Vec.load("d2v.model")
    corpus_docs = corpus['words']
    vectors = np.zeros((len(corpus_docs), model.vector_size))
    for i,words in enumerate(corpus_docs):
        vectors[i] = model.infer_vector(words)
    return vectors

def make_cluster_centers(corpus_vectors, agglo_model, n_clusters):
    clusters_vectors = {cluster_id: [] for cluster_id in range(n_clusters)}
    for i, cluster in enumerate(agglo_model.labels_):
        clusters_vectors[cluster].append(corpus_vectors[i])
    return np.asarray(
        [KMeans(n_clusters=1).fit(np.asarray(clusters_vectors[i])).cluster_centers_.squeeze()
            for i in range(n_clusters)])



def make_word_cloud(corpus, corpus_vectors, cluster_data):
    distances = np.zeros(len(cluster_data['docs']))
    for i, doc_id in enumerate(cluster_data['docs']):
        distances[i] = np.linalg.norm(cluster_data['center'] - corpus_vectors[doc_id])
    distances = distances / np.linalg.norm(distances) #normaliazing
    words_in_cluster = {}
    for i, doc_id in enumerate(cluster_data['docs']):
        words = corpus['words'][doc_id]
        for word in words:
            if word not in words_in_cluster:
                words_in_cluster[word] = 0
            words_in_cluster[word] += 1 - distances[i]
    return WordCloud(mode='RGBA', background_color=None, max_words=20).fit_words(words_in_cluster), distances



def run_agglo_clusters(vectors):
    corpus = pd.read_json("corpus.json")
    bottom_level_topics = 32
    for i in range(4, 0, -1):
        level = 5 - i
        n_clusters = int(bottom_level_topics / (2 ** i))
        folder = f'Results/Hiercal/level{level}'
        os.mkdir(folder)
        cl = AgglomerativeClustering(compute_distances=True, n_clusters=n_clusters)
        model = cl.fit(vectors)

        clusters_data = {cluster_id: {'docs': []} for cluster_id in range(n_clusters)}
        for i, cluster_id in enumerate(model.labels_):
            clusters_data[cluster_id]['docs'].append(i)
        centers = make_cluster_centers(vectors, model, n_clusters)
        for cluster_id in range(n_clusters):
            clusters_data[cluster_id].update({'center': centers[cluster_id]})
            wc, distances = make_word_cloud(corpus, vectors, clusters_data[cluster_id])
            clusters_data[cluster_id].update({'distances': distances})
            plt.gca().imshow(wc)
            plt.gca().axis("off")
            plt.gca().set_title(f"Topic #{i}_{cluster_id}")
            plt.savefig(f"{folder}/{level}_{cluster_id}.png")
            plt.close()



def main(args):

    # print("Pre-Processing Corpus...", end='')
    # sleep(1)
    # print("DONE")

    make_corpus(args.corpus)
    os.mkdir('Results')
    os.mkdir('Results/LDA_topics')

    print("Starting first method")
    print("Training LDA...", end='')
    lda, model_name, id2word = run_lda(args.corpus, args.num_topics, args.alpha)
    print("DONE")

    print("Clustering LDA results...", end='')
    analyze_lda(lda, model_name, id2word)
    print("DONE")
    print("Finished first method")

    print("Starting second method")
    os.mkdir('Results/Hiercal')
    print("Training Doc2Vec model...", end='')
    vectors = doc2vec_vectorize()
    print("DONE")

    print("Running Agglomerative Hieratical Clustering",end='')
    run_agglo_clusters(vectors)
    print("DONE")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHC client demonstration')
    parser.add_argument('corpus', metavar='corpus_folder',
                        help='Number of topics in corpus')
    parser.add_argument('num_topics', metavar='Num_of_topics', type=int, default=20,
                        help='Number of topics in corpus')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='LDA model alpha')
    parser.add_argument('--words_filter', metavar='path_to_words_file',
                        help='Number of topics in corpus')
    parser.add_argument('--use_tf_id',
                        help='if wanting to use doc2vec', default=True, action='store_false')

    args = parser.parse_args()
    main(args)
