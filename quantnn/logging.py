"""
===============
quantnn.logging
===============

This module provides a generic training logger to handle the logging of
training information.
"""
from datetime import datetime

import numpy as np

import rich
from rich.align import Align
from rich.text import Text
from rich.console import Console, RenderGroup
from rich.columns import Columns
from rich.panel import Panel
from rich.padding import Padding
from rich.table import Table, Column
import rich.rule
from rich.progress import (SpinnerColumn, BarColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

import xarray as xr

_TRAINING = "training"
_VALIDATION = "validation"

class Progress(rich.progress.Progress):
    """
    Custom progress bar that automatically enters the progress
    bar context on construction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().start()

    def get_renderables(self):
        """
        Overrides get_renderables method to add a title to the progress
        bar.
        """
        table = self.make_tasks_table(self.tasks)
        table.title = "\n Training progress:"
        table.width = 90
        yield table


def _make_table(epoch,
                total_loss_training,
                total_loss_validation=None,
                losses_training=None,
                losses_validation=None,
                metrics=None,
                learning_rate=None,
                header=True):
    """
    Draws a table to track losses and metrics over the training process.

    Args:
        epoch: The current epoch
        time: The time that the epoch took
        total_loss_training: The training loss summed for all outputs.
        total_loss_validation: The validation loss summed for all outputs.
        losses_training: dict containing the training losses for each output.
        losses_validation: dict containing the validation losses for each
            output.
        learning_rate: The learning rate during the epoch.
        header: Whether or not to print the header.
    """
    col_width = 8
    multi_target = losses_training is not None and len(losses_training) > 1

    if header:
        title = "\n\n Training history\n"
    else:
        title = None

    # Calculate width of table and columns
    epoch_width = 15
    if not multi_target:
        train_width = 20
    else:
        train_width = min(40, (col_width + 2) * (len(losses_training) + 1))

    val_width = 0
    if total_loss_validation is not None:
        val_width = 20
        if multi_target:
            val_width = min(40, (col_width + 2) * (len(losses_training) + 1))

    all_metrics_width = 0
    if metrics is not None:
        if not multi_target:
            metrics_width = (col_width + 2)
        else:
            metrics_width = len(losses_training) * (col_width + 2) + 1
        all_metrics_width = len(metrics) * metrics_width

    table_width = epoch_width + train_width + val_width + all_metrics_width

    table = rich.table.Table(expand=False,
                             box=rich.box.SIMPLE,
                             title=title,
                             show_header=header,
                             show_footer=False,
                             show_edge=False,
                             width=table_width,
                             leading=0)

    table.add_column(Text("Epoch", style="Grey"),
                     justify="center",
                     width=epoch_width)
    table.add_column(Text("Training loss", style="red bold"),
                     justify="center",
                     width=train_width)
    if total_loss_validation is not None:
        table.add_column(Text("Validation loss", style="blue bold"),
                         justify="center",
                         width=val_width)
    if metrics is not None:
        for name, m in metrics.items():
            table.add_column(Text(name, style="purple bold"),
                             justify="center",
                             width=metrics_width)

    def make_header_columns():
        # Epoch and LR
        columns = [Text("#", justify="right", style="bold")]
        if learning_rate is not None:
            columns += [Text("LR", justify="right")]
        yield Columns(columns, align="center", width=5)

        # Training losses
        text = Align(Text("Total", justify="right", style="bold red"), width=col_width, align="center")
        if multi_target:
            columns = [text] + [Align(Text(n, justify="right", style="red"), width=col_width)
                                for n in losses_training.keys()]
            yield Columns(columns, align="center", width=col_width)
        else:
            yield text

        # Validation losses
        if total_loss_validation is not None:
            text = Align(Text("Total", justify="center", style="bold blue"), width=col_width, align="center")
            if multi_target:
                columns = [text] + [Align(Text(n, justify="center", style="blue"), width=col_width)
                                    for n in losses_validation.keys()]
                yield Columns(columns, align="center", width=col_width)
            else:
                yield text

        # Metrics
        if metrics is not None:
            for name, values in metrics.items():
                if isinstance(values, dict):
                    columns = [Align(Text(n, justify="center", style="purple"), width=col_width)
                                    for n in values.keys()]
                    yield Columns(columns, align="center", width=col_width)
                else:
                    yield Align(Text(""), width=col_width)

    def make_columns():
        yield Columns([
            Align(Text(f"{epoch:3}", justify="right", style="bold"), width=4),
            Align(Text(f"{learning_rate:1.3f}", justify="right"), width=5),
        ], align="center", width=5)

        text = Align(Text(f"{total_loss_training:3.3f}", style="bold red"),
                     align="center", width=col_width)
        if multi_target:
            columns = [text] + [Align(Text(f"{l:3.3f}", justify="right", style="red"), width=col_width, align="right")
                        for _, l in losses_training.items()]
            yield Columns(columns, align="center", width=col_width)
        else:
            yield text

        if total_loss_validation is not None:
            text = Align(
                Text(
                    f"{total_loss_validation:.3f}",
                    justify="center",
                    style="bold blue"),
                width=col_width,
                align="center"
            )
            if multi_target:
                columns = [text]
                for _, l in losses_validation.items():
                    columns.append(
                        Align(Text(f"{l:.3f}",
                                   justify="center",
                                   style="blue"),
                              width=col_width,
                              align="center")
                        )
                yield Columns(columns, align="center", width=col_width)
            else:
                yield text

        # Metrics
        if metrics is not None:
            for name, values in metrics.items():
                if isinstance(values, dict):
                    columns = [Align(Text(f"{v:.3f}", justify="center", style="purple"), width=col_width)
                               for _, v in values.items()]
                    yield Columns(columns, align="center", width=col_width)
                else:
                    yield Align(Text(f"{values:.3f}"), width=col_width, style="purple")

    if header:
        table.add_row(*make_header_columns())
        table.add_row()
    else:
        table.add_row(*make_columns())

    return table

class TrainingLogger:
    """
    Logger class that prints training statistics to standard out.
    """
    def __init__(self, n_epochs, log_rate=9):
        """
        Create a new logger instance.

        Args:
            n_epochs: The number of epochs for which the training will last.
            log_rate: The number of training steps to perform between
                 subsequent messages.
        """
        self.i_epoch = 0
        self.i_train_batch = 0
        self.i_val_batch = 0
        self.n_epochs = n_epochs

        self.train_loss = 0.0
        self.train_samples = 0
        self.train_losses = {}
        self.val_loss = 0.0
        self.val_samples = 0
        self.val_losses = {}

        self.attributes = None

        self.log_rate = log_rate
        self.epoch_start_time = datetime.now()

        self.console = Console()

        # Logging state
        self.state = _TRAINING
        self.initialized = False
        self.progress = None

        self.history = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None

    def start_progress_bar(self, of=None):
        """
        Starts the progress bar for the current epoch.

        Args:
             of: The total number of batches in the epoch or None
                  if that is unknown.
        """
        if self.progress is None:
            self.progress = Progress(
                TextColumn(f"Epoch {self.i_epoch + 1}"),
                SpinnerColumn(),
                TextColumn("Batch {task.fields[batch]:3} / {task.fields[of]:3}"),
                BarColumn(),
                TimeElapsedColumn(table_column=Column(header="Elapsed")),
                TextColumn("/"),
                TimeRemainingColumn(table_column=Column(header="Remaining")),
                TextColumn("|"),
                TextColumn("[red]Loss: {task.fields[batch_loss]:1.3f}[red],"),
                TextColumn("[red][b]Avg.: {task.fields[running_mean]:1.3f}[/b][/red]"),
                auto_refresh=False,
                transient=True
            )

            if of is None:
                of = 0
            fields={
                "running_mean": np.nan,
                "batch_loss": np.nan,
                "of": of,
                "batch": 0,
            }
            self.task = self.progress.add_task(
                f"Epoch {self.i_epoch + 1}",
                total=of,
                **fields)
        self.progress.start()
        self.progress.update(self.task, completed=0, total=of)
        self.progress.refresh()


    def set_attributes(self, attributes):
        """
        Stores attributes that describe the training in the logger.
        These will be stored in the logger history.

        Args:
            Dictionary of attributes to store in the history of the
            logger.
        """
        self.attributes = attributes


    def epoch_begin(self, model):
        """
        Called at the beginning of each epoch.

        Args:
            The model that is trained in its current state.
        """
        pass

    def training_step(self,
                      total_loss,
                      n_samples,
                      of=None,
                      losses=None):
        """
        Log processing of a training batch. This method should be called
        after each batch is processed so that the logger can keep track
        of training progress.

        Args:
            total_loss: The total loss of the current batch.
            n_samples: The number of samples in the batch.
            of: If available the number of batches in the epoch.
            losses: Dictionary containing the losses for each
                 model output.
        """
        if (self.i_train_batch == 0):
            self.train_loss = 0.0
            self.train_samples = 0
            if losses is not None and len(losses) > 1:
                for k in losses:
                    self.train_losses[k] = 0.0
            self.start_progress_bar(of=of)

        self.train_loss += n_samples * total_loss
        self.train_samples += n_samples
        self.i_train_batch += 1

        if losses is not None and len(losses) > 1:
            for k in losses:
                self.train_losses[k] += n_samples * losses[k]

        self.progress.update(
            task_id=self.task,
            completed=self.i_train_batch,
            running_mean=self.train_loss / self.train_samples,
            batch=self.i_train_batch,
            of=of,
            batch_loss=total_loss
        )
        if (self.i_train_batch % self.log_rate) == 0:
            self.progress.refresh()

    def validation_step(self,
                        total_loss,
                        n_samples,
                        of=None,
                        losses=None):
        """
        Log processing of a validation batch.

        Args:
            i: The index of the current batch.
            total_loss: The sum of all losses of the current batch.
            n_samples: The number of samples in the batch.
            of: If available the number of batches in the epoch.
            losses: Dictionary containing the losses for each
                 model output.
        """
        if (self.i_val_batch == 0):
            self.val_loss = 0.0
            self.val_samples = 0
            if losses is not None and len(losses) > 1:
                for k in losses:
                    self.val_losses[k] = 0.0

        self.val_loss += n_samples * total_loss
        self.val_samples += n_samples
        self.i_val_batch += 1

        if losses is not None and len(losses) > 1:
            for k in losses:
                self.val_losses[k] += n_samples * losses[k]

    def epoch(self, learning_rate=None, metrics=None):
        """
        Log processing of epoch.

        Args:
            learning_rate: If available the learning rate of the optimizer.
        """
        # Calculate statistics from epoch
        train_loss = self.train_loss / self.train_samples
        train_losses = {k: v / self.train_samples for k, v in self.train_losses.items()}

        if self.val_samples > 0:
            val_loss = self.val_loss / self.val_samples
            val_losses = {k: v / self.train_samples for k, v in self.val_losses.items()}
        else:
            val_loss = None
            val_losses = None

        self.i_epoch += 1
        self.i_train_batch = 0
        self.i_val_batch = 0

        metric_values = {}
        if metrics is not None:
            for m in metrics:
                try:
                    metric_values[m.name] = m.get_values()
                except AttributeError as e:
                    pass

        # Combine stats in dataset
        data = {}
        data["training_loss"] = (("epochs",), [train_loss])
        if len(train_losses) > 1:
            for t in train_losses:
                k = "training_loss_" + t
                data[k] = (("epochs",), [train_losses[t]])


        if val_loss is not None:
            data["validation_loss"] = (("epochs",), [val_loss])
            if len(val_losses) > 1:
                for t in val_losses:
                    k = "validation_loss_" + t
                    data[k] = (("epochs",), [val_losses[t]])

        for name, values in metric_values.items():
            if isinstance(values, dict):
                for target, value in values.items():
                    k = name + "_" + target
                    data[k] = (("epochs",), [value])
            else:
                data[name] = (("epochs"), [values])
        data["epochs"] = [self.i_epoch]
        dataset = xr.Dataset(data)
        if self.history is None:
            self.history = xr.Dataset(dataset)
        else:
            self.history = xr.concat([self.history, dataset], dim="epochs")


        # Write output
        self.progress.stop()
        self.progress = None
        if (self.i_epoch <= 1):
            table_row = _make_table(self.i_epoch,
                                    total_loss_training=train_loss,
                                    total_loss_validation=val_loss,
                                    losses_training=train_losses,
                                    losses_validation=val_losses,
                                    learning_rate=learning_rate,
                                    metrics=metric_values,
                                    header=True)
            self.console.print(table_row)
        table_row = _make_table(self.i_epoch,
                                total_loss_training=train_loss,
                                total_loss_validation=val_loss,
                                losses_training=train_losses,
                                losses_validation=val_losses,
                                learning_rate=learning_rate,
                                metrics=metric_values,
                                header=False)
        self.console.print(table_row)

    def training_end(self):
        """
        Called to signal the end of the training to the logger.
        """
        if self.attributes is not None:
            self.history.attrs.update(self.attributes)



