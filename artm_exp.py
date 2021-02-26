import os
import artm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    root = "Artm"
    cluster_quantile = 0.85
    corpus = pd.read_json("corpus.json")
    bv = artm.BatchVectorizer(data_path='bc4', data_format='batches')
    for filename in os.listdir(root):
        print(filename)
        model_folder = f'{root}/{filename}'
        num_of_topics = int(filename.split('_')[1])
        model = artm.ARTM(num_topics=num_of_topics)
        model.load(f'{model_folder}/model')
        theta = model.transform(bv)
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
