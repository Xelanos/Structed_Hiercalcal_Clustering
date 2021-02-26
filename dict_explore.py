import utils
import gensim
import pickle
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt

def get_bigrams(words, bi_min=5, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_words(file):
    with open(file, 'r') as f_in:
        return f_in.read().split()


def make_dict(dir, load_bigrams=False):
    if not load_bigrams:
        words = utils.run_parrlell_on_dir(dir, get_words)
        bigram_mod = get_bigrams(words)
        bigram = [bigram_mod[article] for article in words]
        with open('bigrams.pic', 'wb') as f:
            pickle.dump(bigram, f)
    else:
        with open('bigrams.pic', 'rb') as f:
            bigram = pickle.load(f)
    print('Finished bigrams')

    id2word = gensim.corpora.Dictionary(bigram)
    return id2word


if __name__ == '__main__':
    original = make_dict("CleanedText_Costum", True)
    lower_filters = range(1, 10)
    upper_filters = range(2, 40)
    upper_filters = [num / 100 for num in upper_filters]

    heat_map = []
    for lower_filter in lower_filters:
        row = []
        for upper_filter in upper_filters:
            copy = deepcopy(original)
            copy.filter_extremes(lower_filter, upper_filter)
            row.append(len(original) - len(copy))
        heat_map.append(row)

    df = pd.DataFrame._from_arrays(heat_map, lower_filters, upper_filters)
    ax = sns.heatmap(df)
    ax.set(ylabel='upper filter (appear in too much %)', xlabel="lower filter (don't appear in at least)",
           title='Number of words filtered')
    plt.show()

    per_lower_filter = []
    for lower_filter in range(1, 20):
        copy = deepcopy(original)
        copy.filter_extremes(lower_filter, 0.3)
        per_lower_filter.append(len(original) - len(copy))
    plt.plot(per_lower_filter)
    plt.title("words removed as a function of lower filter")
    plt.show()
