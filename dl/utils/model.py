import torch.nn as nn


class Model(nn.Module):

    @property
    def trainable_parameters(self):
        return self.parameters()
