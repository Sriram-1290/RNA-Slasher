"""
Training utilities for enhanced RNA-FM model.
Includes stable training components like label smoothing, cosine annealing, 
gradient clipping, early stopping, and mixed precision training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
try:
    from .config import EnhancedRNAConfig
except ImportError:
    from config import EnhancedRNAConfig


class LabelSmoothingBCELoss(nn.Module):
    """Label smoothing for binary classification"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Use BCE loss with logits for numerical stability
        return F.binary_cross_entropy(pred, smooth_target)


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    train_loss: float
    val_loss: float
    train_roc_auc: float
    val_roc_auc: float
    train_f1: float
    val_f1: float
    train_precision: float
    val_precision: float
    train_recall: float
    val_recall: float
    learning_rate: float
    gradient_norm: Optional[float] = None


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 monitor: str = 'val_roc_auc', mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.mode_worse = np.greater if mode == 'min' else np.less
        
    def __call__(self, metrics: TrainingMetrics) -> bool:
        score = getattr(metrics, self.monitor)
        
        if self.best_score is None:
            self.best_score = score
        elif self.mode_worse(score, self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


class ModelCheckpoint:
    """Model checkpointing utility"""
    
    def __init__(self, filepath: str, monitor: str = 'val_roc_auc', 
                 mode: str = 'max', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        self.mode_better = np.less if mode == 'min' else np.greater
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def __call__(self, model: nn.Module, metrics: TrainingMetrics) -> bool:
        score = getattr(metrics, self.monitor)
        
        if not self.save_best_only:
            # Save every epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics.__dict__,
                'epoch': metrics.epoch
            }, self.filepath.format(epoch=metrics.epoch))
            return True
        
        # Save only best model
        if self.best_score is None or self.mode_better(score, self.best_score):
            self.best_score = score
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics.__dict__,
                'epoch': metrics.epoch,
                'best_score': self.best_score
            }, self.filepath)
            return True
            
        return False


class MetricsTracker:
    """Track and log training metrics"""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        self.metrics_history: List[TrainingMetrics] = []
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_metrics(self, metrics: TrainingMetrics):
        """Log metrics for current epoch"""
        self.metrics_history.append(metrics)
        
        # Log to console and file
        self.logger.info(
            f"Epoch {metrics.epoch:3d} | "
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"Val Loss: {metrics.val_loss:.4f} | "
            f"Train ROC-AUC: {metrics.train_roc_auc:.4f} | "
            f"Val ROC-AUC: {metrics.val_roc_auc:.4f} | "
            f"Train F1: {metrics.train_f1:.4f} | "
            f"Val F1: {metrics.val_f1:.4f} | "
            f"LR: {metrics.learning_rate:.6f}"
        )
        
        if metrics.gradient_norm is not None:
            self.logger.info(f"Gradient Norm: {metrics.gradient_norm:.4f}")
    
    def save_metrics(self, filepath: Optional[str] = None):
        """Save metrics history to JSON"""
        if filepath is None:
            filepath = os.path.join(self.log_dir, 'metrics_history.json')
        
        metrics_dict = [m.__dict__ for m in self.metrics_history]
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        if not self.metrics_history:
            return
            
        epochs = [m.epoch for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history]
        train_aucs = [m.train_roc_auc for m in self.metrics_history]
        val_aucs = [m.val_roc_auc for m in self.metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # ROC-AUC curves
        axes[0, 1].plot(epochs, train_aucs, label='Train ROC-AUC', color='blue')
        axes[0, 1].plot(epochs, val_aucs, label='Val ROC-AUC', color='red')
        axes[0, 1].set_title('Training and Validation ROC-AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 scores
        train_f1s = [m.train_f1 for m in self.metrics_history]
        val_f1s = [m.val_f1 for m in self.metrics_history]
        axes[1, 0].plot(epochs, train_f1s, label='Train F1', color='blue')
        axes[1, 0].plot(epochs, val_f1s, label='Val F1', color='red')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        lrs = [m.learning_rate for m in self.metrics_history]
        axes[1, 1].plot(epochs, lrs, label='Learning Rate', color='green')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


class StableTrainer:
    """Enhanced training loop with stability improvements"""
    
    def __init__(self, model: nn.Module, config: EnhancedRNAConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = LabelSmoothingBCELoss(config.training.label_smoothing)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize learning rate scheduler
        if config.training.use_cosine_annealing:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.training.cosine_t_initial,
                T_mult=config.training.cosine_t_mult,
                eta_min=config.training.cosine_eta_min
            )
        else:
            self.scheduler = None
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.training.use_mixed_precision else None
        
        # Initialize utilities
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            monitor=config.training.early_stopping_monitor,
            mode=config.training.early_stopping_mode
        )
        
        self.checkpoint = ModelCheckpoint(
            filepath=os.path.join(config.logging.checkpoint_dir, 'best_model.pth'),
            monitor=config.logging.checkpoint_monitor,
            mode=config.logging.checkpoint_mode,
            save_best_only=config.logging.save_best_only
        )
        
        self.metrics_tracker = MetricsTracker(config.logging.log_dir)
        
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        # Convert to binary predictions
        binary_preds = (predictions >= 0.5).astype(int)
        binary_targets = (targets >= 0.5).astype(int)
        
        try:
            roc_auc = roc_auc_score(binary_targets, predictions)
        except ValueError:
            roc_auc = 0.0
            
        try:
            f1 = f1_score(binary_targets, binary_preds)
            precision = precision_score(binary_targets, binary_preds)
            recall = recall_score(binary_targets, binary_preds)
        except ValueError:
            f1 = precision = recall = 0.0
        
        return {
            'roc_auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with autocast():
                    predictions = self.model(data)
                    loss = self.criterion(predictions, targets)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                predictions = self.model(data)
                loss = self.criterion(predictions, targets)
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.calculate_metrics(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        predictions = self.model(data)
                        loss = self.criterion(predictions, targets)
                else:
                    predictions = self.model(data)
                    loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self.calculate_metrics(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, metrics
    
    def get_gradient_norm(self) -> float:
        """Calculate gradient norm for monitoring"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def train(self, train_loader, val_loader, epochs: Optional[int] = None) -> List[TrainingMetrics]:
        """Main training loop"""
        if epochs is None:
            epochs = self.config.training.epochs
        
        self.metrics_tracker.logger.info(f"Starting training for {epochs} epochs")
        self.metrics_tracker.logger.info(f"Device: {self.device}")
        self.metrics_tracker.logger.info(f"Mixed precision: {self.scaler is not None}")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Calculate gradient norm
            gradient_norm = self.get_gradient_norm() if self.config.logging.track_gradient_norms else None
            
            # Create metrics object
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_roc_auc=train_metrics['roc_auc'],
                val_roc_auc=val_metrics['roc_auc'],
                train_f1=train_metrics['f1'],
                val_f1=val_metrics['f1'],
                train_precision=train_metrics['precision'],
                val_precision=val_metrics['precision'],
                train_recall=train_metrics['recall'],
                val_recall=val_metrics['recall'],
                learning_rate=current_lr,
                gradient_norm=gradient_norm
            )
            
            # Log metrics
            self.metrics_tracker.log_metrics(metrics)
            
            # Save checkpoint
            if self.checkpoint(self.model, metrics):
                self.metrics_tracker.logger.info(f"Saved best model at epoch {epoch + 1}")
            
            # Check early stopping
            if self.early_stopping(metrics):
                self.metrics_tracker.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Save final metrics and plots
        self.metrics_tracker.save_metrics()
        if self.config.logging.track_learning_curves:
            plot_path = os.path.join(self.config.logging.log_dir, 'training_curves.png')
            self.metrics_tracker.plot_training_curves(plot_path)
        
        return self.metrics_tracker.metrics_history