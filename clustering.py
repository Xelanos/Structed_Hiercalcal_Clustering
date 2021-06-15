from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from wordcloud import WordCloud ,STOPWORDS
from joblib import dump, load
import pandas as pd
import numpy as np


most_common_words = ['judge', 'yes', 'know', 'thank', 'documnet', 'time', 'document', 'people',
                     'ask', 'think', 'say', 'tell', 'come', 'right', 'like']

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def tf_idf_vectorize(corpus):
    tfidf_tranform = TfidfVectorizer(use_idf=True, min_df=10, max_df=0.5)
    vectors = tfidf_tranform.fit_transform(' '.join(words) for words in corpus['words'].to_list())
    return vectors.toarray()


def doc2vec_vectorize(corpus):
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
            if word in most_common_words:
                continue
            if word not in words_in_cluster:
                words_in_cluster[word] = 0
            words_in_cluster[word] += 1 - distances[i]
    return WordCloud(mode='RGBA', background_color=None, max_words=20,
                     stopwords=most_common_words + list(STOPWORDS)).fit_words(words_in_cluster), distances


if __name__ == '__main__':
    n_clusters = 8
    corpus = pd.read_json("corpus.json")
    vectors = doc2vec_vectorize(corpus)
    print("Done vectorization")
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
        plt.gca().set_title(f"Topic #{cluster_id}")
        plt.show()
        print(f"Topic #{cluster_id} closests docs:")
        for doc_id in clusters_data[0]['distances'].argsort()[:5]:
            print(corpus['file_name'][doc_id])
        print("\n")





    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
