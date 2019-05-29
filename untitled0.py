# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:19:25 2018

@author: user
"""


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1, size=50, sg=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
## access vector for one word
print(model['sentence'])
## save model
model.save('model.bin')
## load model
new_model = Word2Vec.load('model.bin')
print(new_model)
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
print(model.wv.most_similar('this'))
print("similarity between this and sentence")
print(model.wv.similarity('this','sentence'))
c_list=['this','is']
print("predicted word for this,is")
print(model.predict_output_word(c_list,2))
