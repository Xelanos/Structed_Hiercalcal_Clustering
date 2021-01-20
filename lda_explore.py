import math
import gensim
from wordcloud import WordCloud
from matplotlib import pyplot as plt


if __name__ == '__main__':
    lda = gensim.models.LdaMulticore.load('saved_lda/lda_train_7-topics.model')

    fig, axes = plt.subplots(lda.num_topics / 3, math.ceil(lda.num_topics / 3), figsize=(10, 10), sharex=True, sharey=True)

    for topic_id in range(lda.num_topics):
        plt.figure()
        plt.imshow(WordCloud(mode='RGBA', background_color=None, max_words=100)
                   .fit_words(dict(lda.show_topic(topic_id, 200))))
        plt.axis("off")
        plt.title("Topic #" + str(topic_id))
        plt.show()
    print('g')
