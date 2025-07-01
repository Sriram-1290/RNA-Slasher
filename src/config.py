"""
Configuration management for RNA-FM enhanced model.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters"""
    
    # Model architecture parameters
    d_model: int = 128  # Model dimension
    n_head: int = 8     # Number of attention heads
    n_layers: int = 3   # Number of cross-attention layers
    dropout: float = 0.1
    
    # Sequence parameters
    seq_len: int = 21   # siRNA length
    mrna_len: int = 80  # mRNA length
    embed_dim: int = 4  # One-hot encoding dimension (AUCG) - compatible with existing data
    
    # CNN parameters
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # LSTM parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 96]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 5, 7]


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Stable training parameters
    gradient_clip_val: float = 1.0
    label_smoothing: float = 0.1
    
    # Learning rate scheduling
    use_cosine_annealing: bool = True
    cosine_t_initial: int = 20
    cosine_t_mult: int = 2
    cosine_eta_min: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = 'val_roc_auc'
    early_stopping_mode: str = 'max'
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Cross-validation
    n_folds: int = 5
    random_state: int = 42


@dataclass
class RegularizationConfig:
    """Configuration class for regularization parameters"""
    
    # Dropout rates
    attention_dropout: float = 0.1
    cnn_dropout: float = 0.2
    lstm_dropout: float = 0.2
    mlp_dropout: float = 0.3
    
    # Batch normalization
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.1
    
    # Layer normalization
    use_layer_norm: bool = True
    
    # Residual connections
    use_residual_connections: bool = True


@dataclass
class LoggingConfig:
    """Configuration class for logging parameters"""
    
    # Logging levels and directories
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    experiment_name: str = 'enhanced_rna_fm'
    
    # Metrics tracking
    track_attention_weights: bool = True
    track_gradient_norms: bool = True
    track_learning_curves: bool = True
    
    # Checkpoint saving
    save_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    checkpoint_monitor: str = 'val_roc_auc'
    checkpoint_mode: str = 'max'
    
    # Visualization
    save_attention_plots: bool = True
    plot_dir: str = 'plots'


@dataclass
class EnhancedRNAConfig:
    """Complete configuration for enhanced RNA-FM model"""
    
    model: ModelConfig
    training: TrainingConfig
    regularization: RegularizationConfig
    logging: LoggingConfig
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 regularization_config: Optional[RegularizationConfig] = None,
                 logging_config: Optional[LoggingConfig] = None):
        
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.regularization = regularization_config or RegularizationConfig()
        self.logging = logging_config or LoggingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'regularization': self.regularization.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'EnhancedRNAConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model_config=ModelConfig(**config_dict['model']),
            training_config=TrainingConfig(**config_dict['training']),
            regularization_config=RegularizationConfig(**config_dict['regularization']),
            logging_config=LoggingConfig(**config_dict['logging'])
        )
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    config_obj = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)


def get_default_config() -> EnhancedRNAConfig:
    """Get default configuration for enhanced RNA-FM model"""
    return EnhancedRNAConfig()


def get_high_performance_config() -> EnhancedRNAConfig:
    """Get high-performance configuration for enhanced RNA-FM model"""
    config = EnhancedRNAConfig()
    
    # Enhanced model architecture
    config.model.d_model = 256
    config.model.n_head = 16
    config.model.n_layers = 4
    config.model.dropout = 0.05
    
    # More aggressive training
    config.training.epochs = 150
    config.training.learning_rate = 5e-4
    config.training.gradient_clip_val = 0.5
    config.training.label_smoothing = 0.05
    
    # Enhanced regularization
    config.regularization.attention_dropout = 0.05
    config.regularization.cnn_dropout = 0.1
    config.regularization.lstm_dropout = 0.1
    config.regularization.mlp_dropout = 0.2
    
    return config


def get_fast_training_config() -> EnhancedRNAConfig:
    """Get configuration optimized for fast training/development"""
    config = EnhancedRNAConfig()
    
    # Smaller model
    config.model.d_model = 64
    config.model.n_head = 4
    config.model.n_layers = 2
    
    # Faster training
    config.training.epochs = 50
    config.training.batch_size = 32
    config.training.learning_rate = 2e-3
    config.training.early_stopping_patience = 10
    
    return config