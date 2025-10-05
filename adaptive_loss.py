"""
Adaptive Loss Function for Utility-Privacy Trade-off

This module implements the core adaptive loss function that balances:
1. Utility Preservation (Target Model performance)
2. Privacy Destruction (Adversary performance)
3. Adaptive Budget Control (Spatial and magnitude sparsity)

The loss function enables the Obfuscator to learn optimal obfuscation
that minimizes visual perturbation while maximizing privacy protection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

class AdaptiveLoss(nn.Module):
    """
    Adaptive loss function for utility-privacy trade-off optimization.
    
    The loss function combines:
    - Utility preservation (Target Model performance)
    - Privacy destruction (Adversary performance)  
    - Adaptive budget control (Spatial and magnitude sparsity)
    """
    
    def __init__(self,
                 alpha: float = 1.0,  # Utility preservation weight
                 beta: float = 1.0,   # Privacy destruction weight
                 gamma: float = 0.1,  # Budget control weight
                 delta: float = 0.01, # Spatial sparsity weight
                 epsilon: float = 0.01, # Magnitude sparsity weight
                 target_utility_accuracy: float = 0.9,
                 target_privacy_accuracy: float = 0.0,
                 adaptive_weights: bool = True):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        
        self.target_utility_accuracy = target_utility_accuracy
        self.target_privacy_accuracy = target_privacy_accuracy
        self.adaptive_weights = adaptive_weights
        
        # Adaptive weight parameters
        if adaptive_weights:
            self.alpha_param = nn.Parameter(torch.tensor(alpha))
            self.beta_param = nn.Parameter(torch.tensor(beta))
            self.gamma_param = nn.Parameter(torch.tensor(gamma))
        else:
            self.alpha_param = torch.tensor(alpha)
            self.beta_param = torch.tensor(beta)
            self.gamma_param = torch.tensor(gamma)
        
        # Loss history for adaptive adjustment
        self.loss_history = {
            'utility_loss': [],
            'privacy_loss': [],
            'budget_loss': [],
            'total_loss': []
        }
        
        # Performance tracking
        self.performance_history = {
            'utility_accuracy': [],
            'privacy_accuracy': [],
            'budget_usage': []
        }
    
    def forward(self,
                obfuscated_images: torch.Tensor,
                masks: torch.Tensor,
                utility_logits: torch.Tensor,
                utility_labels: torch.Tensor,
                privacy_logits: torch.Tensor,
                privacy_labels: torch.Tensor,
                step: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the adaptive loss function.
        
        Args:
            obfuscated_images: Obfuscated images (B, C, H, W)
            masks: Obfuscation masks (B, 1, H, W)
            utility_logits: Target model logits (B, num_utility_classes)
            utility_labels: Utility task labels (B,)
            privacy_logits: Adversary logits (B, num_identity_classes)
            privacy_labels: Identity labels (B,)
            step: Training step for adaptive adjustment
            
        Returns:
            total_loss: Combined adaptive loss
            loss_components: Dictionary of individual loss components
        """
        # 1. Utility Preservation Loss
        utility_loss = self._compute_utility_loss(utility_logits, utility_labels)
        
        # 2. Privacy Destruction Loss
        privacy_loss = self._compute_privacy_loss(privacy_logits, privacy_labels)
        
        # 3. Adaptive Budget Control Loss
        budget_loss = self._compute_budget_loss(masks, obfuscated_images)
        
        # 4. Spatial Sparsity Loss
        spatial_loss = self._compute_spatial_sparsity_loss(masks)
        
        # 5. Magnitude Sparsity Loss
        magnitude_loss = self._compute_magnitude_sparsity_loss(masks)
        
        # Combine losses with adaptive weights
        total_loss = (self.alpha_param * utility_loss +
                     self.beta_param * privacy_loss +
                     self.gamma_param * budget_loss +
                     self.delta * spatial_loss +
                     self.epsilon * magnitude_loss)
        
        # Record loss history
        self._update_loss_history(utility_loss, privacy_loss, budget_loss, total_loss)
        
        # Compute performance metrics
        performance_metrics = self._compute_performance_metrics(
            utility_logits, utility_labels, privacy_logits, privacy_labels, masks
        )
        
        # Adaptive weight adjustment
        if self.adaptive_weights and step > 0:
            self._adjust_adaptive_weights(performance_metrics, step)
        
        # Prepare loss components dictionary
        loss_components = {
            'total_loss': total_loss.item(),
            'utility_loss': utility_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'budget_loss': budget_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'magnitude_loss': magnitude_loss.item(),
            'alpha_weight': self.alpha_param.item(),
            'beta_weight': self.beta_param.item(),
            'gamma_weight': self.gamma_param.item(),
            **performance_metrics
        }
        
        return total_loss, loss_components
    
    def _compute_utility_loss(self, utility_logits: torch.Tensor, utility_labels: torch.Tensor) -> torch.Tensor:
        """Compute utility preservation loss"""
        # Standard cross-entropy loss
        utility_loss = F.cross_entropy(utility_logits, utility_labels)
        
        # Add target accuracy penalty
        with torch.no_grad():
            predicted = torch.argmax(utility_logits, dim=1)
            accuracy = (predicted == utility_labels).float().mean()
            
            # Penalty if accuracy is below target
            if accuracy < self.target_utility_accuracy:
                accuracy_penalty = (self.target_utility_accuracy - accuracy) * 10.0
                utility_loss = utility_loss + accuracy_penalty
        
        return utility_loss
    
    def _compute_privacy_loss(self, privacy_logits: torch.Tensor, privacy_labels: torch.Tensor) -> torch.Tensor:
        """Compute privacy destruction loss"""
        # Standard cross-entropy loss
        privacy_loss = F.cross_entropy(privacy_logits, privacy_labels)
        
        # Add target accuracy penalty (we want low accuracy for privacy)
        with torch.no_grad():
            predicted = torch.argmax(privacy_logits, dim=1)
            accuracy = (predicted == privacy_labels).float().mean()
            
            # Penalty if accuracy is above target (we want low accuracy)
            if accuracy > self.target_privacy_accuracy:
                accuracy_penalty = (accuracy - self.target_privacy_accuracy) * 10.0
                privacy_loss = privacy_loss + accuracy_penalty
        
        return privacy_loss
    
    def _compute_budget_loss(self, masks: torch.Tensor, obfuscated_images: torch.Tensor) -> torch.Tensor:
        """Compute adaptive budget control loss"""
        # L1 regularization on mask to encourage sparsity
        l1_loss = torch.mean(torch.abs(masks))
        
        # L2 regularization on mask to encourage smoothness
        l2_loss = torch.mean(masks ** 2)
        
        # Total variation loss for spatial smoothness
        tv_loss = self._compute_total_variation_loss(masks)
        
        # Adaptive budget based on current performance
        budget_usage = torch.mean(masks)
        target_budget = 0.1  # Target 10% of image area
        
        if budget_usage > target_budget:
            budget_penalty = (budget_usage - target_budget) * 5.0
        else:
            budget_penalty = 0.0
        
        budget_loss = l1_loss + 0.1 * l2_loss + 0.01 * tv_loss + budget_penalty
        
        return budget_loss
    
    def _compute_spatial_sparsity_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute spatial sparsity loss to encourage localized obfuscation"""
        # Encourage masks to be concentrated in specific regions
        # Compute spatial variance to encourage concentration
        spatial_variance = torch.var(masks.view(masks.size(0), -1), dim=1)
        spatial_loss = torch.mean(spatial_variance)
        
        # Encourage masks to be centered (for face obfuscation)
        center_loss = self._compute_center_focus_loss(masks)
        
        return spatial_loss + 0.1 * center_loss
    
    def _compute_magnitude_sparsity_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute magnitude sparsity loss to encourage minimal obfuscation"""
        # Encourage masks to have minimal magnitude
        magnitude_loss = torch.mean(masks)
        
        # Encourage binary-like masks (either 0 or 1)
        binary_loss = torch.mean(masks * (1 - masks))
        
        return magnitude_loss + 0.1 * binary_loss
    
    def _compute_total_variation_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss for spatial smoothness"""
        # Compute gradients
        grad_h = torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])
        grad_w = torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])
        
        tv_loss = torch.mean(grad_h) + torch.mean(grad_w)
        return tv_loss
    
    def _compute_center_focus_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute center focus loss to encourage face-centered obfuscation"""
        B, C, H, W = masks.shape
        
        # Create center weight matrix
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, device=masks.device).float()
        x_coords = torch.arange(W, device=masks.device).float()
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Distance from center
        distance = torch.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
        max_distance = torch.sqrt(torch.tensor(center_y ** 2 + center_x ** 2, dtype=torch.float32))
        center_weights = 1.0 - (distance / max_distance)
        
        # Apply center weights to masks
        weighted_masks = masks * center_weights.unsqueeze(0).unsqueeze(0)
        center_loss = -torch.mean(weighted_masks)  # Negative to encourage center focus
        
        return center_loss
    
    def _compute_performance_metrics(self,
                                   utility_logits: torch.Tensor,
                                   utility_labels: torch.Tensor,
                                   privacy_logits: torch.Tensor,
                                   privacy_labels: torch.Tensor,
                                   masks: torch.Tensor) -> Dict[str, float]:
        """Compute performance metrics"""
        with torch.no_grad():
            # Utility metrics
            utility_predicted = torch.argmax(utility_logits, dim=1)
            utility_accuracy = (utility_predicted == utility_labels).float().mean()
            
            # Privacy metrics
            privacy_predicted = torch.argmax(privacy_logits, dim=1)
            privacy_accuracy = (privacy_predicted == privacy_labels).float().mean()
            
            # Budget metrics
            budget_usage = torch.mean(masks)
            mask_sparsity = torch.mean((masks > 0.1).float())
            
            # Privacy score (higher is better)
            privacy_score = 1.0 - privacy_accuracy
            
            # Utility preservation score
            utility_preservation = utility_accuracy / self.target_utility_accuracy
            
            # Combined trade-off score
            tradeoff_score = privacy_score * utility_preservation
        
        return {
            'utility_accuracy': utility_accuracy.item(),
            'privacy_accuracy': privacy_accuracy.item(),
            'privacy_score': privacy_score.item(),
            'utility_preservation': utility_preservation.item(),
            'budget_usage': budget_usage.item(),
            'mask_sparsity': mask_sparsity.item(),
            'tradeoff_score': tradeoff_score.item()
        }
    
    def _update_loss_history(self, utility_loss: torch.Tensor, privacy_loss: torch.Tensor,
                           budget_loss: torch.Tensor, total_loss: torch.Tensor):
        """Update loss history for adaptive adjustment"""
        self.loss_history['utility_loss'].append(utility_loss.item())
        self.loss_history['privacy_loss'].append(privacy_loss.item())
        self.loss_history['budget_loss'].append(budget_loss.item())
        self.loss_history['total_loss'].append(total_loss.item())
        
        # Keep only recent history
        max_history = 1000
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history:]
    
    def _adjust_adaptive_weights(self, performance_metrics: Dict[str, float], step: int):
        """Adjust adaptive weights based on performance"""
        if step < 100:  # Wait for initial convergence
            return
        
        # Get recent performance
        utility_accuracy = performance_metrics['utility_accuracy']
        privacy_accuracy = performance_metrics['privacy_accuracy']
        budget_usage = performance_metrics['budget_usage']
        
        # Adjust alpha (utility weight) based on utility performance
        if utility_accuracy < self.target_utility_accuracy * 0.9:
            # Utility is too low, increase utility weight
            self.alpha_param.data *= 1.01
        elif utility_accuracy > self.target_utility_accuracy * 1.1:
            # Utility is too high, decrease utility weight
            self.alpha_param.data *= 0.99
        
        # Adjust beta (privacy weight) based on privacy performance
        if privacy_accuracy > self.target_privacy_accuracy + 0.1:
            # Privacy is too low, increase privacy weight
            self.beta_param.data *= 1.01
        elif privacy_accuracy < self.target_privacy_accuracy - 0.1:
            # Privacy is too high, decrease privacy weight
            self.beta_param.data *= 0.99
        
        # Adjust gamma (budget weight) based on budget usage
        if budget_usage > 0.2:  # Using too much budget
            self.gamma_param.data *= 1.01
        elif budget_usage < 0.05:  # Using too little budget
            self.gamma_param.data *= 0.99
        
        # Clamp weights to reasonable ranges
        self.alpha_param.data = torch.clamp(self.alpha_param.data, 0.1, 10.0)
        self.beta_param.data = torch.clamp(self.beta_param.data, 0.1, 10.0)
        self.gamma_param.data = torch.clamp(self.gamma_param.data, 0.01, 1.0)
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive weights"""
        return {
            'alpha': self.alpha_param.item(),
            'beta': self.beta_param.item(),
            'gamma': self.gamma_param.item()
        }
    
    def set_target_performance(self, target_utility_accuracy: float, target_privacy_accuracy: float):
        """Set target performance levels"""
        self.target_utility_accuracy = target_utility_accuracy
        self.target_privacy_accuracy = target_privacy_accuracy
    
    def get_loss_history(self) -> Dict[str, List[float]]:
        """Get loss history for analysis"""
        return self.loss_history.copy()
    
    def reset_history(self):
        """Reset loss history"""
        for key in self.loss_history:
            self.loss_history[key] = []

class PrivacyBudgetController:
    """
    Privacy budget controller for adaptive obfuscation.
    
    This controller manages the privacy budget and adjusts obfuscation
    strength based on current performance.
    """
    
    def __init__(self,
                 initial_budget: float = 0.1,
                 min_budget: float = 0.01,
                 max_budget: float = 0.5,
                 adaptation_rate: float = 0.01):
        self.initial_budget = initial_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.adaptation_rate = adaptation_rate
        
        self.current_budget = initial_budget
        self.budget_history = []
        self.performance_history = []
    
    def update_budget(self, 
                     utility_accuracy: float,
                     privacy_accuracy: float,
                     target_utility: float = 0.9,
                     target_privacy: float = 0.0) -> float:
        """
        Update privacy budget based on current performance.
        
        Args:
            utility_accuracy: Current utility task accuracy
            privacy_accuracy: Current privacy task accuracy
            target_utility: Target utility accuracy
            target_privacy: Target privacy accuracy
            
        Returns:
            Updated privacy budget
        """
        # Compute performance gaps
        utility_gap = target_utility - utility_accuracy
        privacy_gap = privacy_accuracy - target_privacy
        
        # Adjust budget based on gaps
        if utility_gap > 0.05:  # Utility too low
            # Increase budget to preserve more utility
            budget_adjustment = self.adaptation_rate * utility_gap
        elif privacy_gap > 0.05:  # Privacy too low
            # Increase budget to improve privacy
            budget_adjustment = self.adaptation_rate * privacy_gap
        else:
            # Performance is good, try to reduce budget
            budget_adjustment = -self.adaptation_rate * 0.1
        
        # Update budget
        self.current_budget += budget_adjustment
        self.current_budget = np.clip(self.current_budget, self.min_budget, self.max_budget)
        
        # Record history
        self.budget_history.append(self.current_budget)
        self.performance_history.append({
            'utility_accuracy': utility_accuracy,
            'privacy_accuracy': privacy_accuracy,
            'utility_gap': utility_gap,
            'privacy_gap': privacy_gap
        })
        
        return self.current_budget
    
    def get_budget(self) -> float:
        """Get current privacy budget"""
        return self.current_budget
    
    def get_budget_history(self) -> List[float]:
        """Get budget history"""
        return self.budget_history.copy()
    
    def reset_budget(self):
        """Reset budget to initial value"""
        self.current_budget = self.initial_budget
        self.budget_history = []
        self.performance_history = []
