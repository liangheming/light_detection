import torch
from torch import nn
from pytorch_lightning import LightningModule
from utils.general import EpochWarmUpCosineDecayOneCycle
from model.loss.yolox_loss import YOLOXLoss


class TrainTask(LightningModule):
    def __init__(self, cfg, model):
        super(TrainTask, self).__init__()
        self.cfg = cfg
        self.model = model
        self.loss_func = YOLOXLoss(
            model=model
        )
        # num_training_samples = len(self.trainer.datamodule.train_dataloader())
        self.lr_adjuster = None

    def on_train_start(self):
        self.lr_adjuster = EpochWarmUpCosineDecayOneCycle(
            init_lr=self.cfg['optim']['lr'],
            warmup_epoch=self.cfg['optim']['warmup_epoch'],
            epochs=self.cfg['optim']['epochs'],
            total_epochs=self.cfg['optim']['epochs'],
            iter_per_epoch=self.trainer.num_training_batches,
            momentum=self.cfg['optim']['momentum'],
            bias_idx=2
        )

    def training_step(self, batch, batch_idx):
        inp, label, meta_info = batch
        inp = inp.to(self.device)
        label = label.to(self.device)
        target = {"label": label, "batch_len": meta_info['batch_len']}
        predict = self.model(inp)
        loss, iou, obj, cls, gt_num = self.loss_func(predict, target)
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        # self.log("iou", iou, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        # self.log("obj", obj, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        # self.log("cls", cls, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_idx=None,
            optimizer_closure=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
    ):
        self.lr_adjuster(optimizer, batch_idx, epoch)
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", torch.tensor(lr), on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        g0, g1, g2 = [], [], []
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        optimizer = torch.optim.SGD(g0,
                                    lr=self.cfg['optim']['lr'],
                                    momentum=self.cfg['optim']['momentum'],
                                    nesterov=True)
        optimizer.add_param_group({'params': g1, 'weight_decay': self.cfg['optim']['weight_decay']})
        optimizer.add_param_group({'params': g2})
        return optimizer

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items.pop("loss", None)
        return items
