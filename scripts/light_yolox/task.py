import torch
import os
import time
import logging
from typing import Any, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn
from pytorch_lightning import LightningModule
from utils.general import EpochWarmUpCosineDecayOneCycle, ModelEMA
from utils.coco_metrics import coco_map_tensor


class TrainTask(LightningModule):
    def __init__(self, cfg: Dict, model: nn.Module):
        super(TrainTask, self).__init__()
        self.cfg = cfg
        self.model = model
        self.lr_adjuster = None
        self.ema = None
        self.map_flag = 0.0
        self.emap_flag = 0.0

    def on_train_start(self):
        self.ema = ModelEMA(self.model)
        self.lr_adjuster = EpochWarmUpCosineDecayOneCycle(
            init_lr=self.cfg['optim']['lr'],
            warmup_epoch=self.cfg['optim']['warmup_epoch'],
            epochs=self.cfg['optim']['epochs'],
            total_epochs=self.cfg['optim']['epochs'],
            iter_per_epoch=self.trainer.num_training_batches,
            momentum=self.cfg['optim']['momentum'],
            bias_idx=2
        )
        if self.trainer.is_global_zero:
            logging.basicConfig(
                level=logging.INFO,
                filename=os.path.join(self.cfg['save_dir'], "log.txt"),
                filemode="w",
            )

    def on_train_epoch_start(self):
        self.tic = time.time()
        self.info("=" * 80)
        self.info("Training Start {:0>3d}|{:0>3d}".format(self.current_epoch, self.cfg['optim']['epochs']))

    def training_step(self, batch, batch_idx):
        inp, label, meta_info = batch
        inp = inp.to(self.device)
        label = label.to(self.device)
        target = {"label": label, "batch_len": meta_info['batch_len']}
        loss, iou, obj, cls, gt_num = self.model(inp, target)

        self.log("loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("iou", iou, on_step=True, on_epoch=False, sync_dist=True)
        self.log("obj", obj, on_step=True, on_epoch=False, sync_dist=True)
        self.log("cls", cls, on_step=True, on_epoch=False, sync_dist=True)
        if batch_idx % self.cfg['optim']['log_intervel'] == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            msg = "Train [{:0>3d}|{:0>3d}]({:0>4d}|{:0>4d}) "
            msg += "loss: {:6.4f} iou: {:6.4f} obj: {:6.4f} cls: {:6.4f} lr: {:8.6f} match_num {:0>4d}"
            msg = msg.format(
                self.current_epoch,
                self.cfg['optim']['epochs'],
                int(batch_idx),
                int(self.trainer.num_training_batches),
                loss.item() / self.trainer.num_gpus,
                iou.item() / self.trainer.num_gpus,
                obj.item() / self.trainer.num_gpus,
                cls.item() / self.trainer.num_gpus,
                lr,
                int(gt_num / self.trainer.num_gpus))
            self.info(msg)
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

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.ema.update(self.model)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        if self.trainer.is_global_zero:
            torch.save(self.model.state_dict(), os.path.join(self.cfg['save_dir'], "model_last.pth"))
            torch.save(self.ema.ema.state_dict(), os.path.join(self.cfg['save_dir'], "ema_last.pth"))
        self.ema.update_attr(self.model)
        duration = time.time() - self.tic
        self.info("Training End {:0>3d}|{:0>3d} duration: {:0>2d}:{:0>2d}"
                  .format(self.current_epoch, self.cfg['optim']['epochs'], int(duration / 60), int(duration % 60)))
        self.info("=" * 80)

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
        items.pop("loss", None)
        return items

    def validation_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            return
        if (self.current_epoch + 1) % self.cfg['optim']['val_intervel'] != 0:
            return
        inp, label, meta_info = batch
        inp = inp.to(self.device)
        gts = label.to(self.device).split(meta_info["batch_len"])
        predicts = self.model.predict(inp)
        ema_predicts = self.ema.ema.predict(inp)
        return gts, predicts, ema_predicts

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if len(outputs) == self.trainer.num_sanity_val_steps:
            return
        if self.current_epoch == 0:
            return
        if (self.current_epoch + 1) % self.cfg['optim']['val_intervel'] != 0:
            return
        predict_list = list()
        gt_list = list()
        ema_list = list()
        for gts, predicts, ema_predicts in outputs:
            predict_list.extend(predicts)
            gt_list.extend(gts)
            ema_list.extend(ema_predicts)
        mp, mr, map50, map_val = coco_map_tensor(predict_list, gt_list)
        emp, emr, emap50, emap = coco_map_tensor(ema_list, gt_list)
        self.log("mp", mp * 100, on_epoch=True, on_step=False, sync_dist=True)
        self.log("mr", mr * 100, on_epoch=True, on_step=False, sync_dist=True)
        self.log("map50", map50 * 100, on_epoch=True, on_step=False, sync_dist=True)
        self.log("map", map_val * 100, on_epoch=True, on_step=False, sync_dist=True)
        self.log("emp", emp * 100, on_epoch=True, on_step=False, sync_dist=True)
        self.log("emr", emr * 100, on_epoch=True, on_step=False, sync_dist=True)
        self.log("emap50", emap50 * 100, on_epoch=True, on_step=False, sync_dist=True)
        self.log("emap", emap * 100, on_epoch=True, on_step=False, sync_dist=True)
        mp_mean = self.all_gather(mp).mean().item() * 100
        mr_mean = self.all_gather(mr).mean().item() * 100
        map50_mean = self.all_gather(map50).mean().item() * 100
        map_mean = self.all_gather(map_val).mean().item() * 100
        emp_mean = self.all_gather(emp).mean().item() * 100
        emr_mean = self.all_gather(emr).mean().item() * 100
        emap50_mean = self.all_gather(emap50).mean().item() * 100
        emap_mean = self.all_gather(emap).mean().item() * 100
        if self.trainer.is_global_zero:
            if self.map_flag <= map_mean:
                self.map_flag = map_mean
                torch.save(self.model.state_dict(), os.path.join(self.cfg['save_dir'], "model_best.pth"))
            if self.emap_flag <= emap_mean:
                self.emap_flag = emap_mean
                torch.save(self.ema.ema.state_dict(), os.path.join(self.cfg['save_dir'], "ema_best.pth"))

        self.info("=" * 60)
        self.info("ORI performance mp:{:6.4f} mr:{:6.4f} mAP50:{:6.4f} mAP:{:6.4f}\n".format(mp_mean,
                                                                                             mr_mean,
                                                                                             map50_mean,
                                                                                             map_mean))
        self.info("EMA performance mp:{:6.4f} mr:{:6.4f} mAP50:{:6.4f} mAP:{:6.4f}\n".format(emp_mean,
                                                                                             emr_mean,
                                                                                             emap50_mean,
                                                                                             emap_mean))
        self.info("best ORI mAP: {:6.4f} | best EMA mAP: {:6.4f}".format(self.map_flag, self.emap_flag))
        self.info("=" * 60)

    def info(self, msg):
        if self.trainer.is_global_zero:
            if self.cfg['optim']['log_print']:
                print(msg)
            else:
                logging.info(msg)
