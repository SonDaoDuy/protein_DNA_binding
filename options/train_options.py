from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--total_epoch', type=int, default=180, help='# of epochs to linearly decay learning rate to zero')
        self._parser.add_argument('--threshold', type=float, default=0.9, help='threshold to decide a positive sample')
        self._parser.add_argument('--seed', type=int, default=0, help='seed for k-fold validate')
        self._parser.add_argument('--slide', type=int, default=11, help='slice window')
        self._parser.add_argument('--scan', type=int, default=6, help='scan window')
        self._parser.add_argument('--lr_net', type=float, default=0.01, help='initial learning rate for G adam')
        self._parser.add_argument('--adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--adam_b2', type=float, default=0.999, help='beta2 for G adam')

        self.is_train = True