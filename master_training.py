#!/usr/bin/env python3
"""
Master Training Script for Adaptive Privacy-Preserving Face Obfuscation

This script orchestrates the complete training pipeline for the three-part system:
1. Obfuscator (O) - Generates adaptive obfuscation masks
2. Target Model (T) - Performs utility tasks (gender classification)
3. Adversary (A) - Attempts to break privacy (identity classification)

The script implements the complete research framework for solving the
Utility-Privacy Trade-off problem in visual data.
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Import our modules
from models.obfuscator_mask import MaskOnlyObfuscator
from models.obfuscator_feature import FeatureSpaceObfuscator
from models.target_model import TargetModel, UtilityTask
from models.adversary import IdentityAdversary
from adaptive_loss import AdaptiveLoss, PrivacyBudgetController
from three_part_training import ThreePartTrainer
from adaptive_curve_analysis import AdaptiveCurveAnalyzer
from datasets.vggface2 import VGGFace2Dataset
from datasets.ucf101 import UCF101FramesDataset
from facenet_pytorch import MTCNN
import kornia

class MasterTrainer:
    """
    Master trainer that orchestrates the complete adaptive obfuscation system.
    
    This class implements the complete research framework for solving the
    Utility-Privacy Trade-off problem in visual data.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Initialize models
        if getattr(self.args, 'obfuscator_type', 'mask') == 'feature':
            self.obfuscator = FeatureSpaceObfuscator(base=getattr(self.args, 'obf_base', 64)).to(self.device)
        else:
            self.obfuscator = MaskOnlyObfuscator(base=getattr(self.args, 'obf_base', 32)).to(self.device)
        self.target_model = TargetModel(
            task=UtilityTask.GENDER_CLASSIFICATION,
            num_classes=args.num_utility_classes
        ).to(self.device)
        self.adversary = IdentityAdversary(
            num_identities=args.num_identities
        ).to(self.device)
        
        # Initialize face detector
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        
        # Initialize adaptive loss
        self.adaptive_loss = AdaptiveLoss(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            target_utility_accuracy=args.target_utility_accuracy,
            target_privacy_accuracy=args.target_privacy_accuracy
        ).to(self.device)
        
        # Initialize privacy budget controller
        self.budget_controller = PrivacyBudgetController(
            initial_budget=args.initial_budget,
            min_budget=args.min_budget,
            max_budget=args.max_budget
        )
        
        # Training history
        self.training_history = {
            'obfuscator_loss': [],
            'target_model_loss': [],
            'adversary_loss': [],
            'utility_accuracy': [],
            'privacy_accuracy': [],
            'privacy_score': [],
            'budget_usage': [],
            'tradeoff_score': [],
            'adaptive_weights': []
        }
        
        # Best model tracking
        self.best_tradeoff_score = 0.0
        self.best_obfuscator_state = None
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_optimizers(self):
        """Setup optimizers for all models"""
        
        # Obfuscator optimizer
        self.opt_obfuscator = torch.optim.Adam(
            self.obfuscator.parameters(),
            lr=self.args.lr_obfuscator,
            betas=(0.5, 0.999)
        )
        
        # Target model optimizer
        self.opt_target = torch.optim.Adam(
            self.target_model.parameters(),
            lr=self.args.lr_target,
            betas=(0.5, 0.999)
        )
        
        # Adversary optimizer
        self.opt_adversary = torch.optim.Adam(
            self.adversary.parameters(),
            lr=self.args.lr_adversary,
            betas=(0.5, 0.999)
        )
        
        # Adaptive loss optimizer (for adaptive weights)
        if self.adaptive_loss.adaptive_weights:
            self.opt_adaptive_loss = torch.optim.Adam(
                self.adaptive_loss.parameters(),
                lr=self.args.lr_adaptive_loss
            )
        else:
            self.opt_adaptive_loss = None
    
    def load_dataset(self):
        """Load the dataset"""
        if self.args.dataset == 'vggface2':
            dataset = VGGFace2Dataset(
                root=self.args.data_root,
                split=self.args.split,
                max_classes=self.args.max_classes,
                max_per_class=self.args.max_per_class,
                resize=(160, 160)
            )
        elif self.args.dataset == 'ucf101':
            dataset = UCF101FramesDataset(
                root=self.args.data_root,
                split=self.args.split,
                max_classes=self.args.max_classes,
                max_per_class=self.args.max_per_class,
                resize=(224, 224)
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def train_phase_1(self, dataloader):
        """Phase 1: Pre-train Target Model and Adversary"""
        print("Phase 1: Pre-training Target Model and Adversary...")
        
        # Pre-train target model
        self.target_model.train()
        for epoch in range(self.args.pretrain_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.opt_target.zero_grad()
                
                # Forward pass
                utility_logits = self.target_model(images)
                loss = nn.CrossEntropyLoss()(utility_logits, labels)
                
                # Backward pass
                loss.backward()
                self.opt_target.step()
                
                # Compute metrics
                with torch.no_grad():
                    predicted = torch.argmax(utility_logits, dim=1)
                    accuracy = (predicted == labels).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            print(f"Target Model Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")
        
        # Pre-train adversary
        self.adversary.train()
        for epoch in range(self.args.pretrain_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.opt_adversary.zero_grad()
                
                # Forward pass
                identity_logits = self.adversary(images)
                loss = nn.CrossEntropyLoss()(identity_logits, labels)
                
                # Backward pass
                loss.backward()
                self.opt_adversary.step()
                
                # Compute metrics
                with torch.no_grad():
                    predicted = torch.argmax(identity_logits, dim=1)
                    accuracy = (predicted == labels).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            print(f"Adversary Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")
        
        print("Phase 1 completed!")
    
    def train_phase_2(self, dataloader):
        """Phase 2: Joint training of all three models"""
        print("Phase 2: Joint training of all models...")
        
        for epoch in range(self.args.joint_epochs):
            # Optional beta scheduling (tapering privacy weight)
            if getattr(self.args, 'beta_schedule', None) and self.args.beta_schedule.lower() == 'linear':
                warmup = max(1, int(getattr(self.args, 'beta_warmup_epochs', 10)))
                beta_min = float(getattr(self.args, 'beta_min', self.args.beta))
                beta_max = float(getattr(self.args, 'beta_max', self.args.beta))
                if epoch < warmup:
                    curr_beta = beta_min + (beta_max - beta_min) * (epoch / warmup)
                else:
                    curr_beta = beta_max
                # Update adaptive loss beta weight
                if hasattr(self.adaptive_loss, 'beta_param'):
                    with torch.no_grad():
                        self.adaptive_loss.beta_param.data.fill_(curr_beta)

            epoch_metrics = self._train_epoch(dataloader, epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"O_Loss={epoch_metrics['obfuscator_loss']:.4f}, "
                  f"T_Loss={epoch_metrics['target_model_loss']:.4f}, "
                  f"A_Loss={epoch_metrics['adversary_loss']:.4f}, "
                  f"U_Acc={epoch_metrics['utility_accuracy']:.4f}, "
                  f"P_Acc={epoch_metrics['privacy_accuracy']:.4f}, "
                  f"P_Score={epoch_metrics['privacy_score']:.4f}, "
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
                self.args.target_utility_accuracy,
                self.args.target_privacy_accuracy
            )
            
            # Record training history
            for key, value in epoch_metrics.items():
                self.training_history[key].append(value)
            
            # Save periodic checkpoints
            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)
        
        print("Phase 2 completed!")
    
    def _train_epoch(self, dataloader, epoch):
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
            'privacy_score': 0.0,
            'budget_usage': 0.0,
            'tradeoff_score': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            utility_labels = labels.to(self.device)
            identity_labels = labels.to(self.device)
            
            # Generate obfuscated images and masks (handles both mask-only and feature-space obfuscators)
            with torch.no_grad():
                obfuscated_images, masks = self._obfuscate_batch(images)
            
            # 1. Train Target Model (T)
            self.opt_target.zero_grad()
            utility_logits = self.target_model(obfuscated_images)
            target_loss, target_metrics = self.target_model.compute_utility_loss(
                obfuscated_images, utility_labels, self.args.target_utility_accuracy
            )
            target_loss.backward()
            self.opt_target.step()
            
            # 2. Train Adversary (A) with optional multiple steps per batch
            adv_steps = max(1, int(getattr(self.args, 'adv_steps', 1)))
            adversary_loss = None
            adversary_metrics = {'identity_accuracy': 0.0}
            for _ in range(adv_steps):
                self.opt_adversary.zero_grad()
                privacy_logits = self.adversary(obfuscated_images)
                adversary_loss, adversary_metrics = self.adversary.compute_attack_loss(
                    obfuscated_images, identity_labels, self.args.target_privacy_accuracy
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
                self.opt_adaptive_loss.step()
            
            # Accumulate metrics
            epoch_metrics['obfuscator_loss'] += obfuscator_loss.item()
            epoch_metrics['target_model_loss'] += target_loss.item()
            epoch_metrics['adversary_loss'] += adversary_loss.item()
            epoch_metrics['utility_accuracy'] += target_metrics['utility_accuracy']
            epoch_metrics['privacy_accuracy'] += adversary_metrics['identity_accuracy']
            epoch_metrics['privacy_score'] += loss_components['privacy_score']
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
                
                # Apply obfuscation
                obfuscated_crop = self._apply_obfuscation_operator(face_crop)
                
                # Blend with mask
                blended_crop = (1 - mask_crop) * face_crop + mask_crop * obfuscated_crop
                
                # Paste back
                obfuscated_images[bi, :, y1:y2, x1:x2] = blended_crop[0]
        
        return obfuscated_images

    def _obfuscate_batch(self, images: torch.Tensor):
        """Unified obfuscation for different obfuscator types.
        Returns (obfuscated_images, masks).
        """
        # Feature-space obfuscator returns (img_obf, mask)
        result = self.obfuscator(images)
        if isinstance(result, tuple) and len(result) == 2:
            img_obf, mask = result
            # Full-frame blend: x' = (1 - m) * x + m * img_obf
            obfuscated = (1 - mask) * images + mask * img_obf
            return obfuscated, mask
        else:
            # Mask-only path: result is mask. Use localized operator pipeline
            mask = result
            obfuscated = self._apply_localized_obfuscation(images, mask)
            return obfuscated, mask
    
    def _apply_obfuscation_operator(self, face_crop: torch.Tensor) -> torch.Tensor:
        """Apply obfuscation operator to face crop"""
        H, W = face_crop.shape[-2], face_crop.shape[-1]
        sigma = self.args.blur_sigma
        
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
    
    def evaluate(self, dataloader):
        """Evaluate the trained system"""
        self.obfuscator.eval()
        self.target_model.eval()
        self.adversary.eval()
        
        total_metrics = {
            'utility_accuracy': 0.0,
            'privacy_accuracy': 0.0,
            'privacy_score': 0.0,
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
                obfuscated_images, masks = self._obfuscate_batch(images)
                
                # Evaluate target model
                utility_logits = self.target_model(obfuscated_images)
                _, target_metrics = self.target_model.compute_utility_loss(
                    obfuscated_images, utility_labels, self.args.target_utility_accuracy
                )
                
                # Evaluate adversary
                privacy_logits = self.adversary(obfuscated_images)
                _, adversary_metrics = self.adversary.compute_attack_loss(
                    obfuscated_images, identity_labels, self.args.target_privacy_accuracy
                )
                
                # Compute budget usage
                budget_usage = torch.mean(masks).item()
                
                # Compute tradeoff score
                privacy_score = 1.0 - adversary_metrics['identity_accuracy']
                utility_preservation = target_metrics['utility_accuracy'] / self.args.target_utility_accuracy
                tradeoff_score = privacy_score * utility_preservation
                
                # Accumulate metrics
                total_metrics['utility_accuracy'] += target_metrics['utility_accuracy']
                total_metrics['privacy_accuracy'] += adversary_metrics['identity_accuracy']
                total_metrics['privacy_score'] += privacy_score
                total_metrics['budget_usage'] += budget_usage
                total_metrics['tradeoff_score'] += tradeoff_score
                
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return total_metrics
    
    def generate_adaptive_curve(self, dataloader):
        """Generate adaptive curve analysis"""
        print("Generating adaptive curve analysis...")
        
        analyzer = AdaptiveCurveAnalyzer(
            device=self.device,
            num_identities=self.args.num_identities,
            num_utility_classes=self.args.num_utility_classes,
            utility_task=UtilityTask.GENDER_CLASSIFICATION,
            obf_base=getattr(self.args, 'obf_base', 32),
            obfuscator_type=getattr(self.args, 'obfuscator_type', 'mask')
        )
        
        # Copy trained models
        analyzer.obfuscator.load_state_dict(self.obfuscator.state_dict())
        analyzer.target_model.load_state_dict(self.target_model.state_dict())
        analyzer.adversary.load_state_dict(self.adversary.state_dict())
        
        # Generate curve
        analysis_summary = analyzer.generate_adaptive_curve(
            dataloader=dataloader,
            budget_range=(0.01, 0.3),
            num_budget_points=15,
            num_training_steps=50,
            save_dir=str(self.output_dir / 'adaptive_curve')
        )
        
        return analysis_summary
    
    def _save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'obfuscator_state_dict': self.obfuscator.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'adversary_state_dict': self.adversary.state_dict(),
            'adaptive_loss_state_dict': self.adaptive_loss.state_dict(),
            'optimizer_states': {
                'obfuscator': self.opt_obfuscator.state_dict(),
                'target': self.opt_target.state_dict(),
                'adversary': self.opt_adversary.state_dict()
            },
            'training_history': self.training_history,
            'best_tradeoff_score': self.best_tradeoff_score
        }
        
        torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    def save_final_models(self):
        """Save final trained models"""
        # Save obfuscator
        torch.save(self.obfuscator.state_dict(), self.output_dir / 'obfuscator.pth')
        
        # Save target model
        torch.save(self.target_model.state_dict(), self.output_dir / 'target_model.pth')
        
        # Save adversary
        torch.save(self.adversary.state_dict(), self.output_dir / 'adversary.pth')
        
        # Save adaptive loss
        torch.save(self.adaptive_loss.state_dict(), self.output_dir / 'adaptive_loss.pth')
        
        # Save best obfuscator
        if self.best_obfuscator_state is not None:
            torch.save(self.best_obfuscator_state, self.output_dir / 'best_obfuscator.pth')
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        print(f"Final models saved to {self.output_dir}")
    
    def plot_training_history(self):
        """Plot training history"""
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
        
        # Plot privacy score
        axes[0, 2].plot(self.training_history['privacy_score'], label='Privacy Score')
        axes[0, 2].set_title('Privacy Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot budget usage
        axes[1, 0].plot(self.training_history['budget_usage'], label='Budget Usage')
        axes[1, 0].set_title('Privacy Budget Usage')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Budget')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot tradeoff score
        axes[1, 1].plot(self.training_history['tradeoff_score'], label='Tradeoff Score')
        axes[1, 1].set_title('Privacy-Utility Tradeoff Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot privacy vs utility
        axes[1, 2].scatter(self.training_history['utility_accuracy'], 
                          self.training_history['privacy_accuracy'],
                          c=self.training_history['budget_usage'], 
                          cmap='viridis', alpha=0.7)
        axes[1, 2].set_title('Privacy vs Utility Tradeoff')
        axes[1, 2].set_xlabel('Utility Accuracy')
        axes[1, 2].set_ylabel('Privacy Accuracy')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plots saved to {self.output_dir / 'training_history.png'}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Master Training Script for Adaptive Privacy-Preserving Face Obfuscation')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='vggface2', choices=['vggface2', 'ucf101'])
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_classes', type=int, default=100)
    parser.add_argument('--max_per_class', type=int, default=50)
    
    # Model arguments
    parser.add_argument('--num_identities', type=int, default=8000, help='Number of identities for adversary')
    parser.add_argument('--num_utility_classes', type=int, default=2, help='Number of utility task classes')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--joint_epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Optimizer arguments
    parser.add_argument('--lr_obfuscator', type=float, default=1e-4)
    parser.add_argument('--lr_target', type=float, default=1e-4)
    parser.add_argument('--lr_adversary', type=float, default=1e-4)
    parser.add_argument('--lr_adaptive_loss', type=float, default=1e-5)
    
    # Loss function arguments
    parser.add_argument('--alpha', type=float, default=1.0, help='Utility weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Privacy weight')
    parser.add_argument('--gamma', type=float, default=0.1, help='Budget weight')
    parser.add_argument('--target_utility_accuracy', type=float, default=0.9)
    parser.add_argument('--target_privacy_accuracy', type=float, default=0.0)
    # Adversarial scheduling / sensitivity
    parser.add_argument('--beta_schedule', type=str, default=None, help='Beta scheduling strategy (e.g., linear)')
    parser.add_argument('--beta_min', type=float, default=None, help='Minimum beta during scheduling')
    parser.add_argument('--beta_max', type=float, default=None, help='Maximum beta during scheduling')
    parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Warmup epochs for beta schedule')
    parser.add_argument('--adv_steps', type=int, default=1, help='Number of adversary updates per batch')
    
    # Privacy budget arguments
    parser.add_argument('--initial_budget', type=float, default=0.1)
    parser.add_argument('--min_budget', type=float, default=0.01)
    parser.add_argument('--max_budget', type=float, default=0.5)
    
    # Obfuscation arguments
    parser.add_argument('--blur_sigma', type=float, default=8.0)
    parser.add_argument('--obf_base', type=int, default=32, help='Base channels for MaskOnlyObfuscator')
    parser.add_argument('--obfuscator_type', type=str, default='mask', choices=['mask','feature'], help='Obfuscator variant')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='master_training_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    print("Starting Master Training for Adaptive Privacy-Preserving Face Obfuscation")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Max classes: {args.max_classes}")
    print(f"Max per class: {args.max_per_class}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Initialize master trainer
    trainer = MasterTrainer(args)
    
    # Setup optimizers
    trainer.setup_optimizers()
    
    # Load dataset
    dataloader = trainer.load_dataset()
    print(f"Loaded dataset with {len(dataloader.dataset)} samples")
    
    # Phase 1: Pre-train models
    trainer.train_phase_1(dataloader)
    
    # Phase 2: Joint training
    trainer.train_phase_2(dataloader)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_metrics = trainer.evaluate(dataloader)
    print(f"Utility Accuracy: {final_metrics['utility_accuracy']:.4f}")
    print(f"Privacy Accuracy: {final_metrics['privacy_accuracy']:.4f}")
    print(f"Privacy Score: {final_metrics['privacy_score']:.4f}")
    print(f"Budget Usage: {final_metrics['budget_usage']:.4f}")
    print(f"Trade-off Score: {final_metrics['tradeoff_score']:.4f}")
    
    # Generate adaptive curve analysis
    if args.joint_epochs > 20:  # Only if we have enough training
        trainer.generate_adaptive_curve(dataloader)
    
    # Save final models
    trainer.save_final_models()
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
