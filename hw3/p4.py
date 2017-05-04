#! python3
"""
@author: b04902053
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from matplotlib import pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from vis.visualization import visualize_saliency

model = load_model('./674561.hdf5')
print('Model loaded.')

image_paths = [
    "./pngs/test/1054.png",
    "./pngs/test/1085.png",
    "./pngs/test/2832.png",
    "./pngs/test/2980.png"
]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(path, grayscale=True)
    seed_img = seed_img.reshape(48, 48, 1)
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = normalize(x)
    # x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, 0, [pred_class], seed_img, alpha=0.5)
    heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()
