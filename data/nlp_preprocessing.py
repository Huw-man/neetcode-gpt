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
        tokenized = []
        for sentence in chain(positive, negative):
            tokenized_s = []
            for w in sentence.split(' '):
                words.add(w)
                tokenized_s.append(w)
            tokenized.append(tokenized_s)

        vocab = {}
        for i, w in enumerate(sorted(list(words))):
            vocab[w] = (i+1)

        tensors = []
        for sentence in tokenized:
            encoded = []
            for w in sentence:
                encoded.append(vocab[w])
            tensors.append(torch.tensor(encoded))
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)


        
        
        


