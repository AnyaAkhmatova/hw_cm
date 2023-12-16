import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

import numpy as np
import pandas as pd

import wandb

from tqdm import tqdm

from .base_trainer import BaseTrainer
from hw_cm.metric import compute_metrics
from hw_cm.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler, config, device)

        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = self.len_epoch // 4

        self.train_metrics = MetricTracker(
            "loss", "grad norm", "accuracy", writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "accuracy", writer=self.writer
        )
        self.evaluation_additional_metrics = {
            "eer": 0, 
            "frr": 0, 
            "far": 0, 
            "thr": 0
        }
        self.eval_logits = []
        self.eval_labels = []

        self.df = pd.DataFrame(columns=["audio", "bonafide proba", "spoof proba", "true label"])

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["audio", "label"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.df = pd.DataFrame(columns=["audio", "bonafide proba", "spoof proba", "true label"])
        self.writer.set_step((epoch - 1) * self.len_epoch)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):

            batch = self.process_batch(
                batch,
                is_train=True,
                metrics=self.train_metrics
            )

            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )

                self._log_scalars(self.train_metrics)
                self._log_predictions(**batch)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx >= self.len_epoch:
                break

        self.writer.add_table('train samples', table=self.df)

        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(batch["audio"])
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["loss"] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        preds = torch.argmax(batch["logits"], dim=-1)
        accuracy = (preds == batch["label"]).float().mean()
        batch["accuracy"] = accuracy

        metrics.update("loss", batch["loss"].item())
        metrics.update("accuracy", batch["accuracy"].item())
        if not is_train:
            self.eval_logits.append(batch["logits"])
            self.eval_labels.append(batch["label"])
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        self.df = pd.DataFrame(columns=["audio", "bonafide proba", "spoof proba", "true label"])
        self.eval_logits = []
        self.eval_labels = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics
                )
            self.eval_labels = torch.cat(self.eval_labels, 0)
            self.eval_logits = torch.cat(self.eval_logits, 0)
            eer, frr, far, thr = compute_metrics(self.eval_labels, self.eval_logits)
            self.evaluation_additional_metrics["eer"] = eer
            self.evaluation_additional_metrics["frr"] = frr
            self.evaluation_additional_metrics["far"] = far
            self.evaluation_additional_metrics["thr"] = thr
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics, self.evaluation_additional_metrics)
            self._log_predictions(**batch)

        if part == 'dev':
            for name, p in self.model.named_parameters():
                if 'bias' in name:
                    self.writer.add_histogram(name, p, bins="auto")

        self.writer.add_table(part + ' samples', table=self.df)
        
        to_log = self.evaluation_metrics.result()
        to_log["eer"] = eer
        to_log["frr"] = frr
        to_log["far"] = far
        to_log["thr"] = thr
        return to_log

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def _log_predictions(self, audio, logits, label, **kwargs):
        ind = np.random.choice(audio.shape[0])
        cur_audio = audio[ind]
        probs = nn.functional.softmax(logits[ind, :], dim=-1)
        bonafide_proba = probs[1].item()
        spoof_proba = probs[0].item()
        cur_label = label[ind].item()
        self.df.loc[len(self.df)] = [wandb.Audio(cur_audio.reshape(-1, 1).detach().cpu().numpy(), sample_rate=16000), bonafide_proba, spoof_proba, cur_label]

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker, additional_metrics=None):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
        if additional_metrics is not None:
            for metric_name in additional_metrics.keys():
                self.writer.add_scalar(f"{metric_name}", additional_metrics[metric_name])

