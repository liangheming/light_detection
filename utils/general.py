import os
import torch
import math
import random
import cv2 as cv
import numpy as np
from copy import deepcopy
from torch import nn
import torch.distributed as dist


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    In addition, sets the env variable `PL_GLOBAL_SEED` which will be passed to
    spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        print(f"No correct seed found, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    cv.setRNGSeed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class Config(object):
    """
    """

    def __init__(self):
        super(Config, self).__init__()

    def __str__(self):
        init_str = "{"
        for k, v in self.__dict__.items():
            init_str += ("\'" + k + "\': " + str(v) + ', ')
        ret_str = init_str[:-2] + "}"
        return ret_str

    @staticmethod
    def dict2obj(dict_obj):
        """
        :param dict_obj:
        :return:
        """
        config = Config()
        for k, v in dict_obj.items():
            assert isinstance(k, str), "only support <str, obj> like dict convert"
            config.__dict__[k] = v
            if isinstance(v, dict):
                config.__dict__[k] = Config.dict2obj(v)
        return config

    def obj2dict(self):
        """
        :return:
        """
        attr_dict = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                attr_dict[k] = v.obj2dict()
            else:
                attr_dict[k] = v
        return attr_dict


def reduce_sum(tensor):
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def rand_seed(seed=888):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def is_parallel(model):
    # is model is parallel with DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

    def reset_updates(self, num=None):
        if num is None:
            num = 1
        self.updates = num


class EpochWarmUpCosineDecayLRAdjust(object):
    def __init__(self, init_lr=0.01,
                 epochs=300,
                 warm_up_epoch=1,
                 iter_per_epoch=1000,
                 gamma=1.0,
                 alpha=0.1,
                 bias_idx=None):
        assert warm_up_epoch < epochs and epochs - warm_up_epoch >= 1
        self.init_lr = init_lr
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.gamma = gamma
        self.alpha = alpha
        self.bias_idx = bias_idx
        self.flag = np.array([warm_up_epoch, epochs]).astype(np.int)
        self.flag = np.unique(self.flag)
        self.warm_up_iter = self.warm_up_epoch * iter_per_epoch

    def cosine(self, current, total):
        return ((1 + math.cos(current * math.pi / total)) / 2) ** self.gamma * (1 - self.alpha) + self.alpha

    def get_lr(self, ite, epoch):
        current_iter = self.iter_per_epoch * epoch + ite
        if epoch < self.warm_up_epoch:
            up_lr = np.interp(current_iter, [0, self.warm_up_iter], [0, self.init_lr])
            down_lr = np.interp(current_iter, [0, self.warm_up_iter], [0.1, self.init_lr])
            return up_lr, down_lr
        num_pow = (self.flag <= epoch).sum() - 1
        cosine_ite = (epoch - self.flag[num_pow] + 1)
        cosine_all_ite = (self.flag[num_pow + 1] - self.flag[num_pow])
        cosine_weights = self.cosine(cosine_ite, cosine_all_ite)
        lr = cosine_weights * self.init_lr
        return lr, lr

    def __call__(self, optimizer, ite, epoch):
        ulr, dlr = self.get_lr(ite, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = dlr if self.bias_idx is not None and i == self.bias_idx else ulr
        return ulr, dlr


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


class AverageLogger(object):
    def __init__(self):
        self.data = 0.
        self.count = 0.

    def update(self, data, count=None):
        self.data += data
        if count is not None:
            self.count += count
        else:
            self.count += 1

    def avg(self):
        return self.data / self.count

    def sum(self):
        return self.data

    def reset(self):
        self.data = 0.
        self.count = 0.


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv
