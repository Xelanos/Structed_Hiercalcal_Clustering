import os
import tqdm
import spacy
import utils
import string
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")

COSTUM_STOP_WORDS = ['pron', 'page', ]


def print_all_words():
    all_words = []
    for root, dirs, files in os.walk("CleanedText_Costum", topdown=False):
        for i, name in enumerate(files):
            if i % 100 == 99: print(f'{i * 100 / 11008:.2f}%')
            with open(os.path.join(root, name), 'r') as f:
                words = f.read().split()
                all_words.extend(words)

    with open('all_words.txt', 'w') as f:
        for word in all_words:
            f.write(f'{word}\n')


def count_words():
    cnt = Counter()
    with open("all_words.txt", 'r') as f:
        for word in f.readlines():
            cnt[word.strip()] += 1
    print(f'total dictionary size: {len(cnt)}')

    with open("common_words.csv", 'w') as f:
        for word, count in cnt.most_common(40):
            f.write(f'{word}, {count}\n')


def get_ent_counters(file):
    counters = {'GPE': Counter(), 'ORG': Counter(), 'PERSON': Counter(), 'NORP': Counter(), 'LOC': Counter(),
                'FAC': Counter(), 'EVENT': Counter(), 'LAW': Counter()}
    with open(f'CleanedText_nopron/{file}', 'r') as f:
        ents = nlp(f.read()).ents
        for e in ents:
            label = e.label_
            if label in counters.keys():
                counters[e.label_][e.text] += 1

    return counters


def ent_anaylysis():
    counters = {'GPE': Counter(), 'ORG': Counter(), 'PERSON': Counter(), 'NORP': Counter(), 'LOC': Counter(),
                'FAC': Counter(), 'EVENT': Counter(), 'LAW': Counter()}
    for root, dirs, files in os.walk("CleanedText_nopron", topdown=False):
        with mp.Pool(mp.cpu_count() - 4) as pool:
            res = list(tqdm.tqdm(pool.imap_unordered(get_ent_counters, files), total=len(files)))
    for counter_dict in res:
        for label in counters.keys():
            counters[label] += counter_dict[label]

    for label in counters.keys():
        with open(f'named2/most_common_{label}.csv', 'w') as f:
            for word, count in counters[label].most_common(200):
                f.write(f'{word}, {count}\n')

    print('g')


def get_size(file):
    return os.path.getsize(file) / 1000


def print_sizes():
    sizes = utils.run_parrlell_on_dir("CleanedText_Costum", get_size)
    sizes = pd.DataFrame(sizes)
    sns.displot(sizes, legend=False)
    plt.title("All Sizes")
    plt.xlabel("File sizes in KB")
    plt.show()

    q_low = sizes.quantile(0.09)
    q_hi = sizes.quantile(0.99)
    df_filtered = sizes[(sizes < q_hi) & (sizes > q_low)]
    sns.displot(df_filtered,  legend=False, kde=True)
    plt.title("Sizes between 9% and 99% quantile")
    plt.xlabel("File sizes in KB")
    plt.show()


if __name__ == '__main__':
    print_all_words()
    count_words()
    # print_sizes()
    # ent_anaylysis()
