"""
Target Model (T) for Utility Task Classification

This module implements the target model that performs non-identity tasks
(e.g., gender classification, expression classification) on obfuscated images.

The Target Model's goal is to maintain high accuracy on utility tasks
while the Obfuscator tries to preserve this utility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from enum import Enum

class UtilityTask(Enum):
    """Enumeration of supported utility tasks"""
    GENDER_CLASSIFICATION = "gender"
    EXPRESSION_CLASSIFICATION = "expression"
    AGE_CLASSIFICATION = "age"
    EMOTION_CLASSIFICATION = "emotion"
    FACE_DETECTION = "face_detection"

class TargetModel(nn.Module):
    """
    Target model for utility task classification.
    
    This model performs non-identity tasks on obfuscated images,
    serving as the utility constraint for the obfuscator.
    """
    
    def __init__(self, 
                 task: UtilityTask = UtilityTask.GENDER_CLASSIFICATION,
                 num_classes: int = 2,
                 input_size: Tuple[int, int] = (160, 160),
                 backbone: str = 'resnet18',
                 pretrained: bool = True):
        super().__init__()
        
        self.task = task
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Load backbone
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 512
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 2048
        elif backbone == 'efficientnet':
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Task-specific classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for utility task classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Task logits of shape (B, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify utility task
        task_logits = self.classifier(features)
        
        return task_logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict utility task with softmax probabilities.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Task probabilities of shape (B, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def compute_utility_loss(self, 
                           obfuscated_images: torch.Tensor, 
                           utility_labels: torch.Tensor,
                           target_accuracy: float = 0.9) -> Tuple[torch.Tensor, dict]:
        """
        Compute utility loss for adversarial training.
        
        Args:
            obfuscated_images: Obfuscated images (B, C, H, W)
            utility_labels: True utility task labels (B,)
            target_accuracy: Target accuracy for the utility task
            
        Returns:
            utility_loss: Loss for the target model
            metrics: Dictionary containing accuracy and other metrics
        """
        # Forward pass
        utility_logits = self.forward(obfuscated_images)
        
        # Compute cross-entropy loss
        utility_loss = F.cross_entropy(utility_logits, utility_labels)
        
        # Compute accuracy
        with torch.no_grad():
            predicted = torch.argmax(utility_logits, dim=1)
            accuracy = (predicted == utility_labels).float().mean()
            
            # Top-2 accuracy for binary tasks
            if self.num_classes == 2:
                top2_accuracy = accuracy  # Same as top-1 for binary
            else:
                top2_pred = torch.topk(utility_logits, 2, dim=1)[1]
                top2_accuracy = torch.any(top2_pred == utility_labels.unsqueeze(1), dim=1).float().mean()
        
        metrics = {
            'utility_loss': utility_loss.item(),
            'utility_accuracy': accuracy.item(),
            'top2_accuracy': top2_accuracy.item(),
            'utility_preservation': accuracy.item() / target_accuracy if target_accuracy > 0 else 0,
            'target_accuracy': target_accuracy
        }
        
        return utility_loss, metrics

class MultiTaskTargetModel(nn.Module):
    """
    Multi-task target model that can perform multiple utility tasks simultaneously.
    
    This model can classify gender, expression, age, etc. on the same obfuscated image.
    """
    
    def __init__(self, 
                 tasks: List[UtilityTask],
                 num_classes_per_task: Dict[UtilityTask, int],
                 input_size: Tuple[int, int] = (160, 160),
                 backbone: str = 'resnet18',
                 pretrained: bool = True):
        super().__init__()
        
        self.tasks = tasks
        self.num_classes_per_task = num_classes_per_task
        self.input_size = input_size
        
        # Load backbone
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 512
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task in tasks:
            self.task_heads[task.value] = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes_per_task[task])
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for head in self.task_heads.values():
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[UtilityTask, torch.Tensor]:
        """
        Forward pass for multi-task classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary of task logits
        """
        # Extract shared features
        shared_features = self.shared_features(self.backbone(x))
        
        # Compute task-specific logits
        task_logits = {}
        for task in self.tasks:
            task_logits[task] = self.task_heads[task.value](shared_features)
        
        return task_logits
    
    def compute_multi_task_loss(self, 
                               obfuscated_images: torch.Tensor, 
                               task_labels: Dict[UtilityTask, torch.Tensor],
                               task_weights: Optional[Dict[UtilityTask, float]] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss.
        
        Args:
            obfuscated_images: Obfuscated images (B, C, H, W)
            task_labels: Dictionary of task labels
            task_weights: Optional weights for each task
            
        Returns:
            multi_task_loss: Combined loss for all tasks
            metrics: Dictionary containing per-task metrics
        """
        if task_weights is None:
            task_weights = {task: 1.0 for task in self.tasks}
        
        # Forward pass
        task_logits = self.forward(obfuscated_images)
        
        # Compute per-task losses
        total_loss = 0.0
        task_metrics = {}
        
        for task in self.tasks:
            if task in task_labels:
                logits = task_logits[task]
                labels = task_labels[task]
                
                # Compute cross-entropy loss
                task_loss = F.cross_entropy(logits, labels)
                weighted_loss = task_loss * task_weights[task]
                total_loss += weighted_loss
                
                # Compute accuracy
                with torch.no_grad():
                    predicted = torch.argmax(logits, dim=1)
                    accuracy = (predicted == labels).float().mean()
                
                task_metrics[f'{task.value}_loss'] = task_loss.item()
                task_metrics[f'{task.value}_accuracy'] = accuracy.item()
        
        # Overall metrics
        overall_metrics = {
            'multi_task_loss': total_loss.item(),
            'num_tasks': len(self.tasks),
            **task_metrics
        }
        
        return total_loss, overall_metrics

class AdaptiveTargetModel(TargetModel):
    """
    Adaptive target model that can adjust its sensitivity to obfuscation.
    
    This model can be made more or less sensitive to obfuscation quality
    to test the obfuscator's adaptive capabilities.
    """
    
    def __init__(self, 
                 task: UtilityTask = UtilityTask.GENDER_CLASSIFICATION,
                 num_classes: int = 2,
                 input_size: Tuple[int, int] = (160, 160),
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 sensitivity: float = 1.0):
        super().__init__(task, num_classes, input_size, backbone, pretrained)
        
        self.sensitivity = sensitivity
        self.base_sensitivity = 1.0
        
        # Adaptive components
        self.adaptive_dropout = nn.Dropout(0.5)
        self.adaptive_scale = nn.Parameter(torch.ones(1))
    
    def set_sensitivity(self, sensitivity: float):
        """Set the sensitivity of the target model to obfuscation"""
        self.sensitivity = sensitivity
        self.adaptive_scale.data.fill_(sensitivity)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive scaling"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply adaptive scaling
        features = features * self.adaptive_scale
        
        # Apply adaptive dropout
        features = self.adaptive_dropout(features)
        
        # Classify utility task
        task_logits = self.classifier(features)
        
        return task_logits
    
    def compute_adaptive_utility_loss(self, 
                                    obfuscated_images: torch.Tensor, 
                                    utility_labels: torch.Tensor,
                                    obfuscation_quality: float = 0.5) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive utility loss based on obfuscation quality.
        
        Args:
            obfuscated_images: Obfuscated images (B, C, H, W)
            utility_labels: True utility task labels (B,)
            obfuscation_quality: Quality of obfuscation (0.0 = perfect, 1.0 = no obfuscation)
            
        Returns:
            adaptive_utility_loss: Loss for the target model
            metrics: Dictionary containing accuracy and other metrics
        """
        # Adjust sensitivity based on obfuscation quality
        # Higher obfuscation quality = lower sensitivity
        adaptive_sensitivity = self.base_sensitivity * (1.0 - obfuscation_quality * 0.3)
        self.set_sensitivity(adaptive_sensitivity)
        
        # Compute standard utility loss
        utility_loss, metrics = self.compute_utility_loss(obfuscated_images, utility_labels)
        
        # Add adaptation penalty
        adaptation_penalty = torch.abs(self.adaptive_scale - 1.0) * 0.05
        adaptive_utility_loss = utility_loss + adaptation_penalty
        
        metrics.update({
            'adaptive_utility_loss': adaptive_utility_loss.item(),
            'adaptive_sensitivity': adaptive_sensitivity,
            'obfuscation_quality': obfuscation_quality
        })
        
        return adaptive_utility_loss, metrics

# Utility functions for target model training and evaluation

def train_target_model(target_model: TargetModel,
                     dataloader: torch.utils.data.DataLoader,
                     optimizer: torch.optim.Optimizer,
                     device: torch.device,
                     num_epochs: int = 10,
                     verbose: bool = True) -> dict:
    """
    Train the target model on original (non-obfuscated) data.
    
    Args:
        target_model: The target model to train
        dataloader: DataLoader with original images and utility task labels
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of training epochs
        verbose: Whether to print training progress
        
    Returns:
        training_history: Dictionary containing training metrics
    """
    target_model.train()
    training_history = {
        'epoch_losses': [],
        'epoch_accuracies': []
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (images, utility_labels) in enumerate(dataloader):
            images = images.to(device)
            utility_labels = utility_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            utility_logits = target_model(images)
            loss = F.cross_entropy(utility_logits, utility_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                predicted = torch.argmax(utility_logits, dim=1)
                accuracy = (predicted == utility_labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1
            
            if verbose and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, '
                      f'Acc={accuracy.item():.4f}')
        
        # Record epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        training_history['epoch_losses'].append(avg_loss)
        training_history['epoch_accuracies'].append(avg_accuracy)
        
        if verbose:
            print(f'Epoch {epoch} Summary: Loss={avg_loss:.4f}, '
                  f'Acc={avg_accuracy:.4f}')
    
    return training_history

def evaluate_target_model(target_model: TargetModel,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device) -> dict:
    """
    Evaluate the target model on obfuscated data.
    
    Args:
        target_model: The trained target model
        dataloader: DataLoader with obfuscated images and utility task labels
        device: Device to evaluate on
        
    Returns:
        evaluation_metrics: Dictionary containing evaluation results
    """
    target_model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, utility_labels in dataloader:
            images = images.to(device)
            utility_labels = utility_labels.to(device)
            
            # Forward pass
            utility_logits = target_model(images)
            loss = F.cross_entropy(utility_logits, utility_labels)
            
            # Compute metrics
            predicted = torch.argmax(utility_logits, dim=1)
            accuracy = (predicted == utility_labels).float().mean()
            
            total_loss += loss.item() * images.size(0)
            total_accuracy += accuracy.item() * images.size(0)
            num_samples += images.size(0)
    
    evaluation_metrics = {
        'avg_loss': total_loss / num_samples,
        'avg_accuracy': total_accuracy / num_samples,
        'utility_preservation': total_accuracy / num_samples,
        'num_samples': num_samples
    }
    
    return evaluation_metrics
