# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np
import bio_embeddings.embed as embed


def get_embedder(embedder_idx):
    if embedder_idx == "OneHot": embedder = OneHotEmbedder()
    elif embedder_idx == "ESM":  embedder = embed.ESMEmbedder()
    return embedder


def get_embedding_dim(embedder_idx):
    if   embedder_idx == "OneHot": dim = 20
    elif embedder_idx == "ESM":    dim = 1280

    return dim


class OneHotEmbedder():
    """ One-Hot embedder """
    def __init__(self):
        super(OneHotEmbedder, self).__init__()
        self.alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    def embed(self, sequence):
        """ embed a sequence """
        embedding = np.zeros((len(sequence), len(self.alphabet)))
        for i in range(len(sequence)):
            if sequence[i].upper() in self.alphabet:
                embedding[i, self.alphabet.index(sequence[i].upper())] = 1

        return embedding
