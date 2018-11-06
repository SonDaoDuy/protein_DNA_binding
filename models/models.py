import os
import torch

class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None

        if model_name == 'DNA_model':
            from .DNA_model import DNAmodel
            model = DNAmodel(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model.name)
        return model


class BaseModel(object):

    def __init__(self, opt):
        self._name = 'BaseModel'

        self._opt = opt
        self._gpu_ids = opt.gpu_ids
        self._is_train = opt.is_train

        self._Tensor = torch.cuda.FloatTensor if self._gpu_ids else torch.Tensor
        self._save_dir = os.path.join(opt.checkpoints_dir, opt.name)


    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    # used in test time, no backprop
    def test(self):
        assert False, "test not implemented"

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        optimizer.load_state_dict(torch.load(load_path))
        print ('loaded optimizer: %s' % load_path)

    def _save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print ('saved net: %s' % save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        network.load_state_dict(torch.load(load_path))
        print ('loaded net: %s' % load_path)

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)