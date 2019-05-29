from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
# define training data
#import numpy as np
#f = open('wordemb.txt','w')

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)


result = model.wv.most_similar('this')
print(result)
result = model.wv.similarity('this','for')
print("\nsimilarity between this and for is:\t")
print(result)
print("\nPredicted similarity between this and for is:\t")
c_list=['this','for']
print(model.predict_output_word(c_list,2))
