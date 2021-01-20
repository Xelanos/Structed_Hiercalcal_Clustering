import math
import gensim
from wordcloud import WordCloud
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lda = gensim.models.LdaMulticore.load('saved_lda/lda_train_high_alpha_12-topics.model')

    fig, axes = plt.subplots(3, math.ceil(lda.num_topics / 3), figsize=(10, 10),
                             sharex=True, sharey=True)

    for topic_id, ax in enumerate(axes.flatten()):
        if topic_id >= lda.num_topics: break
        fig.add_subplot(ax)
        plt.gca().imshow(WordCloud(mode='RGBA', background_color=None, max_words=100)
                         .fit_words(dict(lda.show_topic(topic_id, 200))))
        plt.gca().axis("off")
        plt.gca().set_title(f"Topic #{topic_id}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    print('g')
