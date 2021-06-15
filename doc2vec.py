from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd

max_epochs = 100
vec_size = 100
alpha = 0.025

corpus = pd.read_json("corpus.json")
tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(corpus['words'])]

model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=20, workers=8)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
