import numpy as np
import word2vec

"""
word2vec
"""
word2vec.word2phrase('./all.txt', './all-phrases', verbose=True)

# train model
word2vec.word2vec(
        train = './all-phrases',
        output = './all.bin',
        # size=200,
        window=25,
        negative=5,
        # # iter_=ITERATIONS,
        min_count=100,
        # alpha=0.025,
        # cbow=1,
        threads=4
        )






import numpy as np
import word2vec
import nltk

import matplotlib.pyplot as plt
from adjustText import adjust_text

model = word2vec.load('./all.bin')

PLOT_NUM = 500

# get vocabs, vecs# {{{
vocabs = []
vecs = []
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:PLOT_NUM]
vocabs = vocabs[:PLOT_NUM]
# }}}

# Dimensionality Reduction# {{{
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)
# }}}

plt.figure(figsize=(14, 8))

# Plotting# {{{
# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]

texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
# }}}

plt.savefig('q2_11.png', dpi=600)
plt.show()
