import torch.nn as nn


class Model(nn.Module):

    def trainable_parameters(self):
        return self.parameters()
