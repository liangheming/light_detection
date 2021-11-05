import math
import numpy as np


class EpochWarmUpCosineDecayOneCycle(object):
    def __init__(self,
                 init_lr=0.01,
                 warmup_epoch=1,
                 epochs=300,
                 total_epochs=None,
                 iter_per_epoch=300,
                 warmup_factors=0.01,
                 warmup_momentum=None,
                 momentum=None,
                 warmup_bias_lr=0.1,
                 bias_idx=2,
                 cosine_decay_rate=0.1):
        self.init_lr = init_lr
        self.warmup_epoch = warmup_epoch
        self.epochs = epochs
        self.iter_per_epoch = iter_per_epoch
        self.warmup_factors = warmup_factors
        self.warmup_momentum = warmup_momentum
        self.momentum = momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.bias_idx = bias_idx
        self.cosine_decay_rate = cosine_decay_rate
        assert warmup_epoch > 0
        assert total_epochs is None or total_epochs >= epochs
        self.total_epochs = total_epochs
        self.warmup_iter = warmup_epoch * self.iter_per_epoch

    def cosine_lr_factor(self, epoch, total_epochs, start_ratio=1):
        return ((1 - math.cos(epoch * math.pi / total_epochs)) / 2) * (
                self.cosine_decay_rate - start_ratio) + start_ratio

    @staticmethod
    def cosine_lr_factor_ext(epoch, total_epochs, start_ratio=1.0, cosine_decay_rate=0.1):
        return ((1 - math.cos(epoch * math.pi / total_epochs)) / 2) * (
                cosine_decay_rate - start_ratio) + start_ratio

    def get_warm_up_lr(self, it, epoch):
        cur_it = it + epoch * self.iter_per_epoch
        lr = np.interp(cur_it, [0, self.warmup_iter], [self.init_lr * self.warmup_factors, self.init_lr])
        return lr

    def get_warm_up_bias_lr(self, it, epoch):
        cur_it = it + epoch * self.iter_per_epoch
        lr = np.interp(cur_it, [0, self.warmup_iter], [0.1, self.init_lr])
        return lr

    def get_warm_up_momentum(self, it, epoch):
        assert self.warmup_momentum is not None and self.momentum is not None
        cur_it = it + epoch * self.iter_per_epoch
        momentum = np.interp(cur_it, [0, self.warmup_iter], [self.warmup_momentum, self.momentum])
        return momentum

    def set_warm_up_lr(self, it, epoch, optimizer):
        lr = self.get_warm_up_lr(it, epoch)
        bias_lr = self.get_warm_up_bias_lr(it, epoch)
        for i, x in enumerate(optimizer.param_groups):
            x['lr'] = lr if i != self.bias_idx else bias_lr
            if "momentum" in x and self.momentum is not None and self.warmup_momentum is not None:
                x['momentum'] = self.get_warm_up_momentum(it, epoch)

    def set_cosine_lr(self, epoch, optimizer):
        lr = self.cosine_lr_factor(epoch - self.warmup_epoch, self.epochs - self.warmup_epoch) * self.init_lr
        for i, x in enumerate(optimizer.param_groups):
            x['lr'] = lr

    def set_cosine_lr_ext(self, epoch, optimizer):
        lr = self.cosine_lr_factor_ext(epoch - self.epochs, self.total_epochs - self.epochs,
                                       self.cosine_decay_rate,
                                       self.cosine_decay_rate ** 2) * self.init_lr
        for i, x in enumerate(optimizer.param_groups):
            x['lr'] = lr

    def set_constant_lr(self, optimizer):
        for i, x in enumerate(optimizer.param_groups):
            x['lr'] = self.init_lr * 0.1

    def __call__(self, optimizer, it, epoch):
        if self.total_epochs is not None and epoch >= self.total_epochs:
            raise NotImplementedError()
        if epoch < self.warmup_epoch:
            self.set_warm_up_lr(it, epoch, optimizer)
        elif self.warmup_epoch <= epoch < self.epochs:
            self.set_cosine_lr(epoch, optimizer)
        else:
            if self.total_epochs is not None:
                self.set_cosine_lr_ext(epoch, optimizer)
            else:
                self.set_constant_lr(optimizer)

    def get_lr(self, it, epoch):
        if self.total_epochs is not None and epoch >= self.total_epochs:
            raise NotImplementedError()
        if epoch < self.warmup_epoch:
            lr = self.get_warm_up_lr(it, epoch)
        elif self.warmup_epoch <= epoch < self.epochs:
            lr = self.cosine_lr_factor_ext(epoch - self.warmup_epoch,
                                           self.epochs - self.warmup_epoch,
                                           1,
                                           self.cosine_decay_rate) * self.init_lr
        else:
            if self.total_epochs is not None:
                lr = self.cosine_lr_factor_ext(epoch - self.epochs, self.total_epochs - self.epochs,
                                               self.cosine_decay_rate,
                                               self.cosine_decay_rate ** 2) * self.init_lr
            else:
                lr = self.init_lr * self.cosine_decay_rate
        return lr


class IterWarmUpMultiLRDecay(object):
    def __init__(self,
                 init_lr=0.01,
                 warmup_iter=1000,
                 epochs=300,
                 iter_per_epoch=300,
                 warmup_factors=0.01,
                 milestones=[130, 160, 175],
                 gama=0.1
                 ):
        super(IterWarmUpMultiLRDecay, self).__init__()
        self.init_lr = init_lr
        self.warmup_iter = warmup_iter
        self.warmup_factors = warmup_factors
        self.epochs = epochs
        self.milestones = milestones
        self.iter_per_epoch = iter_per_epoch
        self.gama = gama
        self.warm_up_epoch_up_bound = math.ceil(warmup_iter / iter_per_epoch)
        assert milestones[0] >= self.warm_up_epoch_up_bound

    def get_warm_up_lr(self, it, epoch):
        cur_it = it + epoch * self.iter_per_epoch
        lr = np.interp(cur_it, [0, self.warmup_iter], [self.init_lr * self.warmup_factors, self.init_lr])
        return lr

    def get_lr(self, it, epoch):
        cur_iter = it + epoch * self.iter_per_epoch
        if epoch < self.warm_up_epoch_up_bound:
            lr = self.init_lr
            if cur_iter < self.warmup_iter:
                lr = self.get_warm_up_lr(it, epoch)
        else:
            milestones = np.array(self.milestones)
            lr = self.init_lr * self.gama ** ((epoch >= milestones).sum())
        return lr

    def __call__(self, optimizer, it, epoch):
        lr = self.get_lr(it, epoch)
        for x in optimizer.param_groups:
            x['lr'] = lr


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    lr_ad = IterWarmUpMultiLRDecay(
        milestones=[5, 10, 25],
        init_lr=0.1,
        warmup_iter=5,
        iter_per_epoch=6,
        warmup_factors=0.00001)
    ys = list()
    for epoch in range(30):
        for it in range(6):
            lr = lr_ad.get_lr(it, epoch)
            ys.append(lr)
    xs = np.arange(len(ys))
    ys = np.array(ys)
    print(ys)
    plt.plot(xs, ys)
    plt.show()
