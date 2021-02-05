import math
import gensim
from wordcloud import WordCloud
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # lda = gensim.models.LdaMulticore.load('saved_lda/lda_train_auto_eta_35-topics.model')
    # num_topics = lda.num_topics
    lda = gensim.models.hdpmodel.HdpModel.load('hdpsave/first.model')

    num_topics = 21

    fig, axes = plt.subplots(math.ceil(num_topics / 3), 3, figsize=(40, 40),
                             sharex=True, sharey=True)

    for topic_id, ax in enumerate(axes.flatten()):
        if topic_id >= num_topics: break
        fig.add_subplot(ax)
        plt.gca().imshow(WordCloud(mode='RGBA', background_color=None, max_words=20)
                         .fit_words(dict(lda.show_topic(topic_id, 20))))
        plt.gca().axis("off")
        plt.gca().set_title(f"Topic #{topic_id}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    print('g')
