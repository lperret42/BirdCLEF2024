import pandas as pd
import torch
import pytorch_lightning as pl
from typing import List, Tuple, Any

from .metrics import score


class BirdModel(pl.LightningModule):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 lr_scheduler: Any, loss_fn: Any) -> None:
        """Initialize the BirdModel with model components and loss function.

        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            lr_scheduler: Learning rate scheduler.
            loss_fn: Loss function used during training.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.val_logits: List[torch.Tensor] = []
        self.val_labels: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Process one training step with provided batch of data.

        Args:
            batch: The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        _id, x, y, w = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y, weight=w)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int) -> None:
        """Process one validation step with provided batch of data.

        Args:
            batch: The batch of data.
            batch_idx (int): The index of the batch.
        """
        _id, x, y, w = batch
        logits = self(x)
        self.val_logits.append(logits)
        self.val_labels.append(y)

    def on_validation_epoch_end(self) -> None:
        """Handle end of validation epoch: calculate and log validation ROC AUC."""
        logits = torch.cat(self.val_logits, dim=0)
        labels = torch.cat(self.val_labels, dim=0)

        # to float32 for sklearn compatibility
        probs = torch.sigmoid(logits).cpu().detach().float()
        labels_ohe = labels.cpu().detach().float()

        # just to build temporary df in format required by kaggle official metric
        num_classes = probs.shape[1]
        submission_df = pd.DataFrame(probs.numpy(), columns=[f'class_{i}' for i in range(num_classes)])
        solution_df = pd.DataFrame(labels_ohe.numpy(), columns=[f'class_{i}' for i in range(num_classes)])

        id_col = 'id'
        submission_df[id_col] = range(len(submission_df))
        solution_df[id_col] = range(len(submission_df))

        roc_auc_score = score(solution_df, submission_df, row_id_column_name=id_col)
        self.log('val_roc_auc', roc_auc_score, prog_bar=True, logger=True)

        self.val_logits = []
        self.val_labels = []

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        """Configure the optimizers and learning rate schedulers for training.

        Returns:
            Tuple: A tuple containing the list of optimizers and list of LR schedulers.
        """
        return [self.optimizer], [self.lr_scheduler]
