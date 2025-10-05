"""
Adversary Model (A) for Identity Classification Attack

This module implements the adversarial attacker that attempts to break privacy
by classifying the original identity from obfuscated images.

The Adversary's goal is to maximize identity classification accuracy on obfuscated images,
while the Obfuscator tries to minimize this accuracy (privacy destruction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class IdentityAdversary(nn.Module):
    """
    Adversary model for identity classification attack.
    
    This model attempts to classify the original identity from obfuscated images.
    The Obfuscator's success is measured by how poorly this adversary performs.
    """
    
    def __init__(self, 
                 num_identities: int = 8000,
                 input_size: Tuple[int, int] = (160, 160),
                 backbone: str = 'resnet18'):
        super().__init__()
        
        self.num_identities = num_identities
        self.input_size = input_size
        
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=True)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 512
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 2048
        elif backbone == 'efficientnet':
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Identity classification head
        self.identity_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_identities)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.identity_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for identity classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Identity logits of shape (B, num_identities)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify identity
        identity_logits = self.identity_classifier(features)
        
        return identity_logits
    
    def predict_identity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict identity with softmax probabilities.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Identity probabilities of shape (B, num_identities)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def compute_attack_loss(self, 
                           obfuscated_images: torch.Tensor, 
                           identity_labels: torch.Tensor,
                           target_accuracy: float = 0.0) -> Tuple[torch.Tensor, dict]:
        """
        Compute attack loss for adversarial training.
        
        Args:
            obfuscated_images: Obfuscated images (B, C, H, W)
            identity_labels: True identity labels (B,)
            target_accuracy: Target accuracy for the adversary (default: 0.0 for privacy)
            
        Returns:
            attack_loss: Loss for the adversary
            metrics: Dictionary containing accuracy and other metrics
        """
        # Forward pass
        identity_logits = self.forward(obfuscated_images)
        
        # Compute cross-entropy loss
        attack_loss = F.cross_entropy(identity_logits, identity_labels)
        
        # Compute accuracy
        with torch.no_grad():
            predicted = torch.argmax(identity_logits, dim=1)
            accuracy = (predicted == identity_labels).float().mean()
            
            # Top-5 accuracy
            top5_pred = torch.topk(identity_logits, 5, dim=1)[1]
            top5_accuracy = torch.any(top5_pred == identity_labels.unsqueeze(1), dim=1).float().mean()
        
        metrics = {
            'attack_loss': attack_loss.item(),
            'identity_accuracy': accuracy.item(),
            'top5_accuracy': top5_accuracy.item(),
            'privacy_score': 1.0 - accuracy.item(),  # Higher privacy score = lower adversary accuracy
            'target_accuracy': target_accuracy
        }
        
        return attack_loss, metrics

class AdaptiveAdversary(IdentityAdversary):
    """
    Adaptive adversary that can adjust its strength based on obfuscation quality.
    
    This adversary can be made stronger or weaker to test the obfuscator's
    adaptive capabilities.
    """
    
    def __init__(self, 
                 num_identities: int = 8000,
                 input_size: Tuple[int, int] = (160, 160),
                 backbone: str = 'resnet18',
                 adaptation_strength: float = 1.0):
        super().__init__(num_identities, input_size, backbone)
        
        self.adaptation_strength = adaptation_strength
        self.base_strength = 1.0
        
        # Adaptive components
        self.adaptive_dropout = nn.Dropout(0.5)
        self.adaptive_scale = nn.Parameter(torch.ones(1))
        
    def set_adaptation_strength(self, strength: float):
        """Set the adaptation strength of the adversary"""
        self.adaptation_strength = strength
        self.adaptive_scale.data.fill_(strength)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive scaling"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply adaptive scaling
        features = features * self.adaptive_scale
        
        # Apply adaptive dropout
        features = self.adaptive_dropout(features)
        
        # Classify identity
        identity_logits = self.identity_classifier(features)
        
        return identity_logits
    
    def compute_adaptive_loss(self, 
                             obfuscated_images: torch.Tensor, 
                             identity_labels: torch.Tensor,
                             obfuscation_quality: float = 0.5) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive loss based on obfuscation quality.
        
        Args:
            obfuscated_images: Obfuscated images (B, C, H, W)
            identity_labels: True identity labels (B,)
            obfuscation_quality: Quality of obfuscation (0.0 = perfect, 1.0 = no obfuscation)
            
        Returns:
            adaptive_loss: Loss for the adversary
            metrics: Dictionary containing accuracy and other metrics
        """
        # Adjust adversary strength based on obfuscation quality
        # Higher obfuscation quality = weaker adversary
        adaptive_strength = self.base_strength * (1.0 - obfuscation_quality * 0.5)
        self.set_adaptation_strength(adaptive_strength)
        
        # Compute standard attack loss
        attack_loss, metrics = self.compute_attack_loss(obfuscated_images, identity_labels)
        
        # Add adaptation penalty
        adaptation_penalty = torch.abs(self.adaptive_scale - 1.0) * 0.1
        adaptive_loss = attack_loss + adaptation_penalty
        
        metrics.update({
            'adaptive_loss': adaptive_loss.item(),
            'adaptation_strength': adaptive_strength,
            'obfuscation_quality': obfuscation_quality
        })
        
        return adaptive_loss, metrics

class MultiScaleAdversary(IdentityAdversary):
    """
    Multi-scale adversary that attacks at different resolutions.
    
    This adversary can detect identity information at multiple scales,
    making it more robust to obfuscation.
    """
    
    def __init__(self, 
                 num_identities: int = 8000,
                 input_size: Tuple[int, int] = (160, 160),
                 backbone: str = 'resnet18',
                 scales: list = [1.0, 0.75, 0.5]):
        super().__init__(num_identities, input_size, backbone)
        
        self.scales = scales
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        
        # Multi-scale fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.num_identities * len(scales), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_identities)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing"""
        batch_size = x.size(0)
        scale_logits = []
        
        for i, scale in enumerate(self.scales):
            # Resize input
            if scale != 1.0:
                scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Extract features and classify
            features = self.backbone(scaled_x)
            logits = self.identity_classifier(features)
            
            # Apply scale weight
            weighted_logits = logits * self.scale_weights[i]
            scale_logits.append(weighted_logits)
        
        # Fuse multi-scale predictions
        fused_features = torch.cat(scale_logits, dim=1)
        final_logits = self.fusion_layer(fused_features)
        
        return final_logits

# Utility functions for adversary training and evaluation

def train_adversary(adversary: IdentityAdversary,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   num_epochs: int = 10,
                   verbose: bool = True) -> dict:
    """
    Train the adversary model on original (non-obfuscated) data.
    
    Args:
        adversary: The adversary model to train
        dataloader: DataLoader with original images and identity labels
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of training epochs
        verbose: Whether to print training progress
        
    Returns:
        training_history: Dictionary containing training metrics
    """
    adversary.train()
    training_history = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'epoch_top5_accuracies': []
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_top5_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (images, identity_labels) in enumerate(dataloader):
            images = images.to(device)
            identity_labels = identity_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            identity_logits = adversary(images)
            loss = F.cross_entropy(identity_logits, identity_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                predicted = torch.argmax(identity_logits, dim=1)
                accuracy = (predicted == identity_labels).float().mean()
                
                top5_pred = torch.topk(identity_logits, 5, dim=1)[1]
                top5_accuracy = torch.any(top5_pred == identity_labels.unsqueeze(1), dim=1).float().mean()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            epoch_top5_accuracy += top5_accuracy.item()
            num_batches += 1
            
            if verbose and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, '
                      f'Acc={accuracy.item():.4f}, Top5={top5_accuracy.item():.4f}')
        
        # Record epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_top5_accuracy = epoch_top5_accuracy / num_batches
        
        training_history['epoch_losses'].append(avg_loss)
        training_history['epoch_accuracies'].append(avg_accuracy)
        training_history['epoch_top5_accuracies'].append(avg_top5_accuracy)
        
        if verbose:
            print(f'Epoch {epoch} Summary: Loss={avg_loss:.4f}, '
                  f'Acc={avg_accuracy:.4f}, Top5={avg_top5_accuracy:.4f}')
    
    return training_history

def evaluate_adversary(adversary: IdentityAdversary,
                      dataloader: torch.utils.data.DataLoader,
                      device: torch.device) -> dict:
    """
    Evaluate the adversary model on obfuscated data.
    
    Args:
        adversary: The trained adversary model
        dataloader: DataLoader with obfuscated images and identity labels
        device: Device to evaluate on
        
    Returns:
        evaluation_metrics: Dictionary containing evaluation results
    """
    adversary.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_top5_accuracy = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, identity_labels in dataloader:
            images = images.to(device)
            identity_labels = identity_labels.to(device)
            
            # Forward pass
            identity_logits = adversary(images)
            loss = F.cross_entropy(identity_logits, identity_labels)
            
            # Compute metrics
            predicted = torch.argmax(identity_logits, dim=1)
            accuracy = (predicted == identity_labels).float().mean()
            
            top5_pred = torch.topk(identity_logits, 5, dim=1)[1]
            top5_accuracy = torch.any(top5_pred == identity_labels.unsqueeze(1), dim=1).float().mean()
            
            total_loss += loss.item() * images.size(0)
            total_accuracy += accuracy.item() * images.size(0)
            total_top5_accuracy += top5_accuracy.item() * images.size(0)
            num_samples += images.size(0)
    
    evaluation_metrics = {
        'avg_loss': total_loss / num_samples,
        'avg_accuracy': total_accuracy / num_samples,
        'avg_top5_accuracy': total_top5_accuracy / num_samples,
        'privacy_score': 1.0 - (total_accuracy / num_samples),
        'num_samples': num_samples
    }
    
    return evaluation_metrics
