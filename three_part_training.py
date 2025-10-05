"""
Three-Part Training Loop for Adaptive Obfuscation

This module implements the coordinated training of:
1. Obfuscator (O) - Generates obfuscated images and masks
2. Target Model (T) - Performs utility tasks on obfuscated images
3. Adversary (A) - Attempts to break privacy on obfuscated images

The training loop implements the adversarial game where:
- O tries to minimize utility loss + budget loss while maximizing privacy loss
- T tries to maximize utility performance on obfuscated images
- A tries to maximize privacy performance on obfuscated images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from models.obfuscator_mask import MaskOnlyObfuscator
from models.target_model import TargetModel, UtilityTask
from models.adversary import IdentityAdversary
from adaptive_loss import AdaptiveLoss, PrivacyBudgetController
from facenet_pytorch import MTCNN

class ThreePartTrainer:
    """
    Three-part trainer for adaptive obfuscation system.
    
    Coordinates training of Obfuscator (O), Target Model (T), and Adversary (A)
    to achieve optimal utility-privacy trade-off.
    """
    
    def __init__(self,
                 device: torch.device,
                 num_identities: int = 8000,
                 num_utility_classes: int = 2,
                 utility_task: UtilityTask = UtilityTask.GENDER_CLASSIFICATION,
                 target_utility_accuracy: float = 0.9,
                 target_privacy_accuracy: float = 0.0):
        
        self.device = device
        self.num_identities = num_identities
        self.num_utility_classes = num_utility_classes
        self.utility_task = utility_task
        self.target_utility_accuracy = target_utility_accuracy
        self.target_privacy_accuracy = target_privacy_accuracy
        
        # Initialize models
        self.obfuscator = MaskOnlyObfuscator().to(device)
        self.target_model = TargetModel(
            task=utility_task,
            num_classes=num_utility_classes
        ).to(device)
        self.adversary = IdentityAdversary(
            num_identities=num_identities
        ).to(device)
        
        # Initialize face detector for localized obfuscation
        self.mtcnn = MTCNN(keep_all=False, device=device)
        
        # Initialize adaptive loss function
        self.adaptive_loss = AdaptiveLoss(
            target_utility_accuracy=target_utility_accuracy,
            target_privacy_accuracy=target_privacy_accuracy
        ).to(device)
        
        # Initialize privacy budget controller
        self.budget_controller = PrivacyBudgetController()
        
        # Training history
        self.training_history = {
            'obfuscator_loss': [],
            'target_model_loss': [],
            'adversary_loss': [],
            'utility_accuracy': [],
            'privacy_accuracy': [],
            'budget_usage': [],
            'tradeoff_score': []
        }
        
        # Performance tracking
        self.best_tradeoff_score = 0.0
        self.best_obfuscator_state = None
    
    def setup_optimizers(self,
                        lr_obfuscator: float = 1e-4,
                        lr_target: float = 1e-4,
                        lr_adversary: float = 1e-4,
                        lr_adaptive_loss: float = 1e-5):
        """Setup optimizers for all models"""
        
        # Obfuscator optimizer
        self.opt_obfuscator = optim.Adam(
            self.obfuscator.parameters(),
            lr=lr_obfuscator,
            betas=(0.5, 0.999)
        )
        
        # Target model optimizer
        self.opt_target = optim.Adam(
            self.target_model.parameters(),
            lr=lr_target,
            betas=(0.5, 0.999)
        )
        
        # Adversary optimizer
        self.opt_adversary = optim.Adam(
            self.adversary.parameters(),
            lr=lr_adversary,
            betas=(0.5, 0.999)
        )
        
        # Adaptive loss optimizer (for adaptive weights)
        if self.adaptive_loss.adaptive_weights:
            self.opt_adaptive_loss = optim.Adam(
                self.adaptive_loss.parameters(),
                lr=lr_adaptive_loss
            )
        else:
            self.opt_adaptive_loss = None
    
    def train_phase_1(self, dataloader: DataLoader, num_epochs: int = 10):
        """
        Phase 1: Pre-train Target Model and Adversary on original data
        
        This phase establishes baseline performance for both models
        before introducing obfuscation.
        """
        print("Phase 1: Pre-training Target Model and Adversary...")
        
        # Pre-train target model
        self.target_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for images, utility_labels in dataloader:
                images = images.to(self.device)
                utility_labels = utility_labels.to(self.device)
                
                self.opt_target.zero_grad()
                
                # Forward pass
                utility_logits = self.target_model(images)
                loss = nn.CrossEntropyLoss()(utility_logits, utility_labels)
                
                # Backward pass
                loss.backward()
                self.opt_target.step()
                
                # Compute metrics
                with torch.no_grad():
                    predicted = torch.argmax(utility_logits, dim=1)
                    accuracy = (predicted == utility_labels).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            print(f"Target Model Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")
        
        # Pre-train adversary
        self.adversary.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for images, identity_labels in dataloader:
                images = images.to(self.device)
                identity_labels = identity_labels.to(self.device)
                
                self.opt_adversary.zero_grad()
                
                # Forward pass
                identity_logits = self.adversary(images)
                loss = nn.CrossEntropyLoss()(identity_logits, identity_labels)
                
                # Backward pass
                loss.backward()
                self.opt_adversary.step()
                
                # Compute metrics
                with torch.no_grad():
                    predicted = torch.argmax(identity_logits, dim=1)
                    accuracy = (predicted == identity_labels).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            print(f"Adversary Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")
        
        print("Phase 1 completed!")
    
    def train_phase_2(self, dataloader: DataLoader, num_epochs: int = 50):
        """
        Phase 2: Joint training of all three models
        
        This phase implements the adversarial game where:
        - Obfuscator tries to fool the adversary while preserving utility
        - Target model tries to maintain utility performance
        - Adversary tries to break privacy
        """
        print("Phase 2: Joint training of all models...")
        
        for epoch in range(num_epochs):
            epoch_metrics = self._train_epoch(dataloader, epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"O_Loss={epoch_metrics['obfuscator_loss']:.4f}, "
                  f"T_Loss={epoch_metrics['target_model_loss']:.4f}, "
                  f"A_Loss={epoch_metrics['adversary_loss']:.4f}, "
                  f"U_Acc={epoch_metrics['utility_accuracy']:.4f}, "
                  f"P_Acc={epoch_metrics['privacy_accuracy']:.4f}, "
                  f"Budget={epoch_metrics['budget_usage']:.4f}, "
                  f"Tradeoff={epoch_metrics['tradeoff_score']:.4f}")
            
            # Save best model
            if epoch_metrics['tradeoff_score'] > self.best_tradeoff_score:
                self.best_tradeoff_score = epoch_metrics['tradeoff_score']
                self.best_obfuscator_state = self.obfuscator.state_dict().copy()
            
            # Update privacy budget
            current_budget = self.budget_controller.update_budget(
                epoch_metrics['utility_accuracy'],
                epoch_metrics['privacy_accuracy'],
                self.target_utility_accuracy,
                self.target_privacy_accuracy
            )
            
            # Record training history
            for key, value in epoch_metrics.items():
                self.training_history[key].append(value)
        
        print("Phase 2 completed!")
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch of the three-part system"""
        
        # Set models to training mode
        self.obfuscator.train()
        self.target_model.train()
        self.adversary.train()
        
        epoch_metrics = {
            'obfuscator_loss': 0.0,
            'target_model_loss': 0.0,
            'adversary_loss': 0.0,
            'utility_accuracy': 0.0,
            'privacy_accuracy': 0.0,
            'budget_usage': 0.0,
            'tradeoff_score': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            utility_labels = labels.to(self.device)  # Assuming utility labels for now
            identity_labels = labels.to(self.device)  # Assuming identity labels for now
            
            # Generate obfuscated images and masks
            with torch.no_grad():
                masks = self.obfuscator(images)
            
            # Apply localized obfuscation
            obfuscated_images = self._apply_localized_obfuscation(images, masks)
            
            # 1. Train Target Model (T)
            self.opt_target.zero_grad()
            utility_logits = self.target_model(obfuscated_images)
            target_loss, target_metrics = self.target_model.compute_utility_loss(
                obfuscated_images, utility_labels, self.target_utility_accuracy
            )
            target_loss.backward()
            self.opt_target.step()
            
            # 2. Train Adversary (A)
            self.opt_adversary.zero_grad()
            privacy_logits = self.adversary(obfuscated_images)
            adversary_loss, adversary_metrics = self.adversary.compute_attack_loss(
                obfuscated_images, identity_labels, self.target_privacy_accuracy
            )
            adversary_loss.backward()
            self.opt_adversary.step()
            
            # 3. Train Obfuscator (O)
            self.opt_obfuscator.zero_grad()
            
            # Recompute logits for obfuscator training
            with torch.no_grad():
                utility_logits = self.target_model(obfuscated_images)
                privacy_logits = self.adversary(obfuscated_images)
            
            # Compute adaptive loss
            obfuscator_loss, loss_components = self.adaptive_loss(
                obfuscated_images, masks, utility_logits, utility_labels,
                privacy_logits, identity_labels, epoch
            )
            
            obfuscator_loss.backward()
            self.opt_obfuscator.step()
            
            # 4. Update adaptive loss weights (if enabled)
            if self.opt_adaptive_loss is not None:
                self.opt_adaptive_loss.zero_grad()
                # Adaptive loss weights are updated in the forward pass
                self.opt_adaptive_loss.step()
            
            # Accumulate metrics
            epoch_metrics['obfuscator_loss'] += obfuscator_loss.item()
            epoch_metrics['target_model_loss'] += target_loss.item()
            epoch_metrics['adversary_loss'] += adversary_loss.item()
            epoch_metrics['utility_accuracy'] += target_metrics['utility_accuracy']
            epoch_metrics['privacy_accuracy'] += adversary_metrics['identity_accuracy']
            epoch_metrics['budget_usage'] += loss_components['budget_usage']
            epoch_metrics['tradeoff_score'] += loss_components['tradeoff_score']
            
            num_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _apply_localized_obfuscation(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Apply localized obfuscation using face detection"""
        B, C, H, W = images.shape
        obfuscated_images = images.clone()
        
        for bi in range(B):
            img_i = images[bi:bi+1]
            mask_i = masks[bi:bi+1]
            
            # Convert to numpy for MTCNN
            img_np = (img_i[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8')
            boxes, _ = self.mtcnn.detect([img_np])
            
            if boxes is not None and len(boxes) > 0 and len(boxes[0]) > 0:
                # Select largest face
                box = boxes[0][0].astype(int)
                x1, y1, x2, y2 = box
                
                # Clamp bbox to image dimensions
                x1 = max(0, min(W-1, x1))
                x2 = max(x1+1, min(W, x2))
                y1 = max(0, min(H-1, y1))
                y2 = max(y1+1, min(H, y2))
                
                # Clamp oversized boxes
                bw = x2 - x1
                bh = y2 - y1
                max_w = int(0.6 * W)
                max_h = int(0.6 * H)
                
                if bw > max_w:
                    cx = (x1 + x2) // 2
                    hw = max_w // 2
                    x1 = max(0, cx - hw)
                    x2 = min(W, cx + hw)
                
                if bh > max_h:
                    cy = (y1 + y2) // 2
                    hh = max_h // 2
                    y1 = max(0, cy - hh)
                    y2 = min(H, cy + hh)
                
                # Extract face crop and apply obfuscation
                face_crop = img_i[:, :, y1:y2, x1:x2]
                mask_crop = mask_i[:, :, y1:y2, x1:x2]
                
                # Apply obfuscation (simple blur for now)
                obfuscated_crop = self._apply_obfuscation_operator(face_crop)
                
                # Blend with mask
                blended_crop = (1 - mask_crop) * face_crop + mask_crop * obfuscated_crop
                
                # Paste back
                obfuscated_images[bi, :, y1:y2, x1:x2] = blended_crop[0]
        
        return obfuscated_images
    
    def _apply_obfuscation_operator(self, face_crop: torch.Tensor) -> torch.Tensor:
        """Apply obfuscation operator to face crop"""
        # Simple Gaussian blur for now
        import kornia
        
        H, W = face_crop.shape[-2], face_crop.shape[-1]
        sigma = 8.0
        k_desired = max(3, int(2 * (3 * sigma) + 1))
        k_max = max(3, min(int(H), int(W)))
        k = min(k_desired, k_max)
        k = k if (k % 2 == 1) else (k - 1)
        k = max(3, k)
        
        sigma_eff = float(sigma)
        if k < k_desired:
            sigma_eff = max(0.5, (k - 1) / (2.0 * 3.0))
        
        gb = kornia.filters.GaussianBlur2d((k, k), (sigma_eff, sigma_eff))
        return gb(face_crop)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the trained system"""
        self.obfuscator.eval()
        self.target_model.eval()
        self.adversary.eval()
        
        total_metrics = {
            'utility_accuracy': 0.0,
            'privacy_accuracy': 0.0,
            'budget_usage': 0.0,
            'tradeoff_score': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                utility_labels = labels.to(self.device)
                identity_labels = labels.to(self.device)
                
                # Generate obfuscated images
                masks = self.obfuscator(images)
                obfuscated_images = self._apply_localized_obfuscation(images, masks)
                
                # Evaluate target model
                utility_logits = self.target_model(obfuscated_images)
                _, target_metrics = self.target_model.compute_utility_loss(
                    obfuscated_images, utility_labels, self.target_utility_accuracy
                )
                
                # Evaluate adversary
                privacy_logits = self.adversary(obfuscated_images)
                _, adversary_metrics = self.adversary.compute_attack_loss(
                    obfuscated_images, identity_labels, self.target_privacy_accuracy
                )
                
                # Compute budget usage
                budget_usage = torch.mean(masks).item()
                
                # Compute tradeoff score
                privacy_score = 1.0 - adversary_metrics['identity_accuracy']
                utility_preservation = target_metrics['utility_accuracy'] / self.target_utility_accuracy
                tradeoff_score = privacy_score * utility_preservation
                
                # Accumulate metrics
                total_metrics['utility_accuracy'] += target_metrics['utility_accuracy']
                total_metrics['privacy_accuracy'] += adversary_metrics['identity_accuracy']
                total_metrics['budget_usage'] += budget_usage
                total_metrics['tradeoff_score'] += tradeoff_score
                
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return total_metrics
    
    def save_models(self, save_dir: str):
        """Save trained models"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save obfuscator
        torch.save(self.obfuscator.state_dict(), save_path / 'obfuscator.pth')
        
        # Save target model
        torch.save(self.target_model.state_dict(), save_path / 'target_model.pth')
        
        # Save adversary
        torch.save(self.adversary.state_dict(), save_path / 'adversary.pth')
        
        # Save adaptive loss
        torch.save(self.adaptive_loss.state_dict(), save_path / 'adaptive_loss.pth')
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save best obfuscator
        if self.best_obfuscator_state is not None:
            torch.save(self.best_obfuscator_state, save_path / 'best_obfuscator.pth')
        
        print(f"Models saved to {save_path}")
    
    def load_models(self, save_dir: str):
        """Load trained models"""
        save_path = Path(save_dir)
        
        # Load obfuscator
        self.obfuscator.load_state_dict(torch.load(save_path / 'obfuscator.pth'))
        
        # Load target model
        self.target_model.load_state_dict(torch.load(save_path / 'target_model.pth'))
        
        # Load adversary
        self.adversary.load_state_dict(torch.load(save_path / 'adversary.pth'))
        
        # Load adaptive loss
        self.adaptive_loss.load_state_dict(torch.load(save_path / 'adaptive_loss.pth'))
        
        print(f"Models loaded from {save_path}")
    
    def plot_training_history(self, save_dir: str):
        """Plot training history"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot losses
        axes[0, 0].plot(self.training_history['obfuscator_loss'], label='Obfuscator')
        axes[0, 0].plot(self.training_history['target_model_loss'], label='Target Model')
        axes[0, 0].plot(self.training_history['adversary_loss'], label='Adversary')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracies
        axes[0, 1].plot(self.training_history['utility_accuracy'], label='Utility')
        axes[0, 1].plot(self.training_history['privacy_accuracy'], label='Privacy')
        axes[0, 1].set_title('Model Accuracies')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot budget usage
        axes[0, 2].plot(self.training_history['budget_usage'], label='Budget Usage')
        axes[0, 2].set_title('Privacy Budget Usage')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Budget')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot tradeoff score
        axes[1, 0].plot(self.training_history['tradeoff_score'], label='Tradeoff Score')
        axes[1, 0].set_title('Privacy-Utility Tradeoff Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot privacy vs utility
        axes[1, 1].scatter(self.training_history['utility_accuracy'], 
                          self.training_history['privacy_accuracy'],
                          c=self.training_history['budget_usage'], 
                          cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('Privacy vs Utility Tradeoff')
        axes[1, 1].set_xlabel('Utility Accuracy')
        axes[1, 1].set_ylabel('Privacy Accuracy')
        axes[1, 1].grid(True)
        
        # Plot adaptive weights
        if hasattr(self.adaptive_loss, 'loss_history'):
            alpha_weights = [self.adaptive_loss.alpha_param.item()] * len(self.training_history['obfuscator_loss'])
            beta_weights = [self.adaptive_loss.beta_param.item()] * len(self.training_history['obfuscator_loss'])
            gamma_weights = [self.adaptive_loss.gamma_param.item()] * len(self.training_history['obfuscator_loss'])
            
            axes[1, 2].plot(alpha_weights, label='Alpha (Utility)')
            axes[1, 2].plot(beta_weights, label='Beta (Privacy)')
            axes[1, 2].plot(gamma_weights, label='Gamma (Budget)')
            axes[1, 2].set_title('Adaptive Loss Weights')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Weight')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plots saved to {save_path / 'training_history.png'}")

def main():
    """Example usage of the three-part trainer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize trainer
    trainer = ThreePartTrainer(
        device=device,
        num_identities=1000,  # Adjust based on your dataset
        num_utility_classes=2,  # Binary gender classification
        utility_task=UtilityTask.GENDER_CLASSIFICATION,
        target_utility_accuracy=0.9,
        target_privacy_accuracy=0.0
    )
    
    # Setup optimizers
    trainer.setup_optimizers(
        lr_obfuscator=1e-4,
        lr_target=1e-4,
        lr_adversary=1e-4
    )
    
    # Create dummy dataloader (replace with your actual dataloader)
    from torch.utils.data import TensorDataset, DataLoader
    
    # Dummy data
    images = torch.randn(100, 3, 160, 160)
    labels = torch.randint(0, 2, (100,))  # Binary labels
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train the system
    trainer.train_phase_1(dataloader, num_epochs=5)
    trainer.train_phase_2(dataloader, num_epochs=20)
    
    # Evaluate
    metrics = trainer.evaluate(dataloader)
    print(f"Final metrics: {metrics}")
    
    # Save models
    trainer.save_models('trained_models')
    
    # Plot training history
    trainer.plot_training_history('training_plots')

if __name__ == '__main__':
    main()
