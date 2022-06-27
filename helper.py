import numpy as np
import math
import pickle as pkl
from PIL import Image


def unpickle(filename):
    pikd = open(filename, 'rb')
    data = pkl.load(pikd)
    pikd.close()
    return data


def avg_embeddings_patches(xs, n_patches, dims_target, model):
    def get_splits(x):
        n_sqrt = math.sqrt(n_patches)
        if not n_sqrt.is_integer():
            raise Exception('n_patches must be a perfect square!')
        n_sqrt = int(n_sqrt)
        width = x.shape[1] / n_sqrt
        height = x.shape[0] / n_sqrt
        img = Image.fromarray(np.uint8(x * 255))
        img = img.crop((0, 0, n_sqrt * width, n_sqrt * height))  # cut off excess
        res = np.zeros((n_patches, dims_target[0], dims_target[1], x.shape[2]))
        for row in range(n_sqrt):
            for col in range(n_sqrt):
                left = (col * width) % img.width
                upper = (row * height) % img.height
                right = left + width
                lower = upper + height
                img_cropped = img.crop((left, upper, right, lower))
                img_cropped = img_cropped.resize(dims_target, resample=Image.LANCZOS)
                res[row*n_sqrt + col] = np.array(img_cropped) / 255.0
        return res

    es = np.array([model.encoder(get_splits(x))[0].numpy() for x in xs])
    es_mean = np.mean(es, axis=1)
    return es_mean
