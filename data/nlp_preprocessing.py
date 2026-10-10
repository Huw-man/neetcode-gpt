import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List
from itertools import chain

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        words = set()
        for sentence in chain(positive, negative):
            for w in sentence.split(' '):
                words.add(w)

        vocab = {}
        for i, w in enumerate(sorted(list(words))):
            vocab[w] = (i+1)

        tensors = []
        for sentence in chain(positive, negative):
            encoded = []
            for w in sentence.split(' '):
                encoded.append(vocab[w])
            tensors.append(torch.tensor(encoded))
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)


        
        
        


