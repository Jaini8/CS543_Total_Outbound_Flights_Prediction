import pickle

import numpy as np


from itertools import *

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid


hidden1 = repeat( (nn.Linear(8, 8), nn.ReLU()), 50)
hidden1 = chain.from_iterable(hidden1)

hidden2 = repeat( (nn.Linear(5, 5), nn.ReLU()), 50)
hidden2 = chain.from_iterable(hidden2)



class M(torch.nn.Module):
    def __init__(self, num_origin, num_dest):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(M, self).__init__()
        self.emb_size = 128
        self.origin_embs = nn.Embedding(num_origin, self.emb_size)
#         self.dest_embs = nn.Embedding(num_dest, self.emb_size)
        self.month_embs = nn.Embedding(13, self.emb_size)
        self.day_embs = nn.Embedding(32, self.emb_size)
        
        length = 3 * self.emb_size
        self.predictor = nn.Sequential(
                            nn.Linear(length, 8),
                            nn.ReLU(),
                            *hidden1,                
                            nn.Linear(8, 5),
                            nn.BatchNorm1d(5),
                            nn.ReLU(),
                            *hidden2,
                            nn.BatchNorm1d(5),
                            nn.Linear(5, 2),
                            nn.ReLU(),
                            nn.Linear(2, 1),
                        )

    def forward(self, X):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        month, day, origin = X[:, 0], X[:, 1], X[:, 2]
        
        origin_embs = self.origin_embs(origin.int())
#         dest_embs = self.dest_embs(dest.int())
        month_embs = self.month_embs(month.int())
        day_embs = self.day_embs(day.int())
        
        X = torch.hstack([month_embs, day_embs, origin_embs])
        y_pred = self.predictor(X)
        return y_pred