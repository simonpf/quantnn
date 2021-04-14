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

_TRAINING = "training"
_VALIDATION = "validation"

class Progress(rich.progress.Progress):
    """
    Custom progress bar that automatically enters the progress
    bar context on construction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__enter__()

    def close(self):
        self.__exit__(None, None, None)

    def get_renderables(self):
        table = self.make_tasks_table(self.tasks)
        table.title = "\n Training progress:"
        table.width = 90
        yield table

def _make_table(epoch,
                total_loss_training,
                total_loss_validation=None,
                losses_training=None,
                losses_validation=None,
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
    multi_target = losses_training is not None

    if header:
        title = "\n\n Training history\n"
    else:
        title = None

    table = rich.table.Table(expand=False,
                             box=rich.box.SIMPLE,
                             title=title,
                             show_header=header,
                             show_footer=False,
                             show_edge=False,
                             leading=0)
    table.width = 90
    table.add_column(Text("Epoch", style="Grey"),
                     justify="center",
                     max_width=15)

    if losses_training is None:
        width = 20
    else:
        width = min(40, 10 * len(losses_training) + 10)

    table.add_column(Text("Training loss", style="red bold"),
                     justify="center",
                     width=width)

    if total_loss_validation is not None:
        if losses_validation is None:
            width = 20
        else:
            width = min(40, 10 * len(losses_training) + 10)
        table.add_column(Text("Validation loss", style="blue bold"),
                         justify="center",
                         width=width)

    def make_header_columns():
        columns = [Align(Text("#", justify="center", style="bold"), align="center", width=5)]
        if learning_rate is not None:
            columns += [Align(Text("LR", justify="center"), width=10)]
        yield Columns(columns, align="center", expand=True)

        text = Align(Text("Total", justify="right", style="bold red"), width=10)
        if multi_target:
            columns = [text] + [Align(Text(n, justify="right", style="red"), width=10, align="right")
                                for n in losses_training.keys()]
            yield Columns(columns, equal=True, expand=True, align="center")
        else:
            yield text

        if losses_validation is not None:
            text = Align(Text("Total", justify="center", style="bold blue"), width=10)
            if multi_target:
                columns = [text] + [Align(Text(n, justify="center", style="blue"), width=10)
                                    for n in losses_validation.keys()]
                yield Columns(columns, expand=True, align="center")
            else:
                yield text

    def make_columns():
        yield Columns([
            Align(Text(f"{epoch:3}", justify="center", style="bold"), width=5),
            Align(Text(f"{learning_rate:1.3f}", justify="center"), width=10),
        ], align="center", expand=True)

        text = Align(Text(f"{total_loss_training:3.3f}", style="bold red"),
                     align="center", width=10)
        if multi_target:
            columns = [text] + [Align(Text(f"{l:3.3f}", justify="right", style="red"), width=10, align="right")
                        for _, l in losses_training.items()]
            yield Columns(columns, equal=True, expand=True, align="center")
        else:
            yield text

        if losses_validation is not None:
            text = Align(
                Text(
                    f"{total_loss_validation:.3f}",
                    justify="center",
                    style="bold blue"),
                width=7
            )
            if multi_target:
                columns = [text]
                for _, l in losses_validation.items():
                    columns.append(
                        Align(Text(f"{l:.3f}",
                                   justify="center",
                                   style="rblue"),
                              width=10)
                        )
                yield Columns(columns, expand=True, align="center")
            else:
                yield text

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
    def __init__(self, n_epochs, log_rate=100):
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

        self.log_rate = log_rate
        self.epoch_start_time = datetime.now()

        self.console = Console()

        # Logging state
        self.state = _TRAINING
        self.initialized = False
        self.progress = None

    def __del__(self):
        if self.progress is not None:
            self.progress.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.progress is not None:
            self.progress.__exit__(*args, **kwargs)

    def initialize(self, total_loss, losses=None):

        multi_loss = losses is not None

        titles = [
            RenderGroup(
                rich.panel.Panel("Epoch", box=rich.box.ROUNDED),
                rich.panel.Panel("Epoch", box=rich.box.SIMPLE)
            ),
            rich.panel.Panel("Training\n", box=rich.box.SIMPLE_HEAVY),
            rich.panel.Panel("Validation", box=rich.box.SIMPLE_HEAVY)
        ]
        columns = rich.columns.Columns(titles,
                                       expand=True)

        losses = {"Name 1": 1, "Name 2": 2}


    def make_progress_bar(self, of=None):
        if self.progress is not None:
            self.progress.__exit__(None, None, None)
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
            transient=True,
        )

        self.progress.__enter__()

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
        if (self.progress is None) or (self.state == _VALIDATION):
            self.make_progress_bar(of=of)
            self.state = _TRAINING

        if (self.i_train_batch == 0):
            self.train_loss = 0.0
            self.train_samples = 0
            if losses is not None and len(losses) > 1:
                for k in losses:
                    self.train_losses[k] = 0.0

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
        train_loss = self.train_loss / self.train_samples
        if self.val_samples > 0:
            val_loss = self.val_loss / self.val_samples
            val_losses = self.val_losses
        else:
            val_loss = None
            val_losses = None

        self.i_epoch += 1
        self.i_train_batch = 0
        self.i_val_batch = 0

        self.progress.__exit__(None, None, None)
        self.progress = None
        if (self.i_epoch <= 1):
            table_row = _make_table(self.i_epoch,
                                    train_loss,
                                    val_loss,
                                    losses_training=self.train_losses,
                                    losses_validation=val_losses,
                                    learning_rate=learning_rate,
                                    header=True)
            self.console.print(table_row)
        table_row = _make_table(self.i_epoch,
                                train_loss,
                                val_loss,
                                losses_training=self.train_losses,
                                losses_validation=val_losses,
                                learning_rate=learning_rate,
                                header=False)
        self.console.print(table_row)
