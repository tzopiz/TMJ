from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Any, Callable, cast

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.nn import functional as F  # noqa
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,  # learning rate scheduler.
    accelerator: Accelerator,
    epoch_num: int,
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,  # tensorboard logger
    save_on_val: bool = True,  # saves checkpoint on the validation stage
    show_every_x_batch: int = 30,
) -> None:
    global_train_step, global_val_step = 0, 0
    for epoch in tqdm(range(epoch_num)):
        print("-" * 30)
        print(f"Epoch {epoch}/{epoch_num}")

        global_train_step = train_step(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            loss_function=loss_function,
            metric_function=metric_function,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_train_step=global_train_step,
            save_on_val=save_on_val,
            show_every_x_batch=show_every_x_batch,
        )

        if val_dataloader is None:
            continue

        global_val_step = validation_step(
            epoch=epoch,
            model=model,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            metric_function=metric_function,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_val_step=global_val_step,
            save_on_val=save_on_val,
        )


def train_step(
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        loss_function: Callable[[Any, Any], torch.Tensor],
        metric_function: Callable[[Any, Any], torch.Tensor],
        lr_scheduler: LRScheduler,
        accelerator: Accelerator,
        checkpointer: CheckpointSaver,
        tb_logger: SummaryWriter | None,
        global_train_step: int,
        save_on_val: bool = True,
        show_every_x_batch: int = 30,
) -> int:
    model.train()
    batch_idx = 0
    total_train_loss, total_train_metric = 0.0, 0.0

    for inputs, targets in tqdm(train_dataloader, desc="Training", dynamic_ncols=True):
        batch_idx += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        metric = metric_function(outputs, targets)
        total_train_loss += loss.item()
        total_train_metric += metric.item()

        accelerator.backward(loss)
        optimizer.step()

        # Log batch loss and metric
        if not batch_idx % show_every_x_batch:
            tqdm.set_postfix(loss=loss.item(), metric=metric.item())

        if tb_logger is not None:
            tb_logger.add_scalar("loss_train_batch", loss.item(), global_train_step)
            tb_logger.add_scalar("metric_train_batch", metric.item(), global_train_step)
            global_train_step += 1

    lr_scheduler.step()

    # Compute average loss and metric for the epoch
    total_train_loss /= len(train_dataloader)
    total_train_metric /= len(train_dataloader)
    print(f"Epoch train loss: {total_train_loss:.5f}")
    print(f"Epoch train metric: {total_train_metric:.5f}")

    if tb_logger is not None:
        tb_logger.add_scalar("loss_train_epoch", total_train_loss, epoch)
        tb_logger.add_scalar("metric_train_epoch", total_train_metric, epoch)

    # Save checkpoint based on training metric (or validation metric if desired)
    if not save_on_val:
        checkpointer.save(metric_val=total_train_metric, epoch=epoch)

    return global_train_step

def validation_step(
    epoch: int,
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,
    global_val_step: int,
    save_on_val: bool = True,
) -> int:
    model.eval()

    total_val_loss, total_val_metric = 0.0, 0.0

    # Для улучшения читаемости и обновления прогресса в tqdm
    for inputs, targets in tqdm(val_dataloader, desc="Validation", dynamic_ncols=True):
        try:
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                metric = metric_function(outputs, targets)
                total_val_loss += loss.item()
                total_val_metric += metric.item()

            # Логирование результатов для TensorBoard
            if tb_logger is not None:
                tb_logger.add_scalar("loss_val_batch", loss.item(), global_val_step)
                tb_logger.add_scalar("metric_val_batch", metric.item(), global_val_step)
                global_val_step += 1

        except Exception as e:
            print(f"Error in validation batch: {e}")
            continue  # продолжаем с следующей итерации

    if len(val_dataloader) > 0:  # защита от деления на ноль
        total_val_loss /= len(val_dataloader)
        total_val_metric /= len(val_dataloader)
    else:
        print("Warning: Validation dataloader is empty!")
        total_val_loss, total_val_metric = 0.0, 0.0

    # Вывод средних потерь и метрик по всей эпохе
    print(f"Epoch validation loss: {total_val_loss:.5f}")
    print(f"Epoch validation metric: {total_val_metric:.5f}")

    # Логирование результатов по эпохе
    if tb_logger is not None:
        tb_logger.add_scalar("loss_val_epoch", total_val_loss, epoch)
        tb_logger.add_scalar("metric_val_epoch", total_val_metric, epoch)

    # Сохранение чекпойнта
    if save_on_val:
        checkpointer.save(metric_val=total_val_metric, epoch=epoch)

    return global_val_step


class MulticlassCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=input, target=torch.argmax(target, dim=1))

@dataclass
class Checkpoint:
    metric_val: float
    epoch: int
    save_path: Path


class CheckpointSaver:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        metric_name: str,
        save_dir: str,
        rm_save_dir: bool = False,
        max_history: int = 1,
        should_minimize: bool = True,
    ) -> None:
        """
        Args:
            accelerator: huggingface's accelerator
            model: model
            metric_name: name of the metric to log
            save_dir: checkpoint save dir
            max_history: number of checkpoints to store
            should_minimize: if True, metric should be minimized; false, otherwise
        """
        self._accelerator = accelerator
        self._model = model
        self.metric_name = metric_name
        self.save_dir = Path(save_dir)
        self.max_history = max_history
        self.should_minimize = should_minimize

        self._storage: list[Checkpoint] = []

        if os.path.exists(save_dir) and rm_save_dir:
            rmtree(save_dir)

        os.makedirs(save_dir, exist_ok=True)

    def save(self, metric_val: float, epoch: int) -> None:
        """Saves the checkpoint.

        Args:
            metric_val: value of the metric.
            epoch: epoch step.
        """
        save_name_prefix = f"model_e{epoch:03d}_checkpoint"
        save_path = self._save_checkpoint(
            model=self._model, epoch=epoch, save_name_prefix=save_name_prefix
        )
        self._storage.append(
            Checkpoint(metric_val=metric_val, epoch=epoch, save_path=save_path)
        )
        self._storage = sorted(
            self._storage, key=lambda x: x.metric_val, reverse=not self.should_minimize
        )
        if len(self._storage) > self.max_history:
            worst_item = self._storage.pop()
            os.remove(worst_item.save_path)

        copyfile(
            src=self._storage[0].save_path,
            dst=self.save_dir / "model_checkpoint_best.pt",
        )
        print(
            f"Best epoch {self.metric_name} value is {self._storage[0].metric_val:.4f} "
            f"on {self._storage[0].epoch} epoch"
        )

    def _save_checkpoint(
        self, model: nn.Module, epoch: int, save_name_prefix: str
    ) -> Path:
        save_path = pjoin(self.save_dir, f"{save_name_prefix}.pt")
        self._accelerator.wait_for_everyone()
        unwrapped_model = self._accelerator.unwrap_model(model)
        self._accelerator.save(
            obj={"epoch": epoch, "model_state_dict": unwrapped_model.state_dict()},
            f=save_path,
        )
        return Path(save_path)


def load_checkpoint(model: nn.Module, load_path: str) -> nn.Module:
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model