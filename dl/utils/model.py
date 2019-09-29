import torch
import torch.nn as nn


class Model(nn.Module):

    def trainable_parameters(self):
        return self.parameters()

    def load_from_path(self, checkpoint_path: str, key: str = 'model_state_dict'):
        checkpoint = torch.load(checkpoint_path)
        self.load(checkpoint, key)

    def load(self, checkpoint, key: str = 'model_state_dict'):
        self.load_state_dict(checkpoint[key])
