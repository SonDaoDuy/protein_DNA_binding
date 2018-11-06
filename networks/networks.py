import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'DNA_target':
            from .NDA_target import Network
            network = Network(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print ("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)

