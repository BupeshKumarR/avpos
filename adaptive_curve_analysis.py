"""
Adaptive Curve Analysis for Privacy-Utility Trade-off

This module generates comprehensive analysis of the privacy-utility trade-off
by varying the privacy budget parameter (γ) and measuring the resulting
performance on both privacy and utility tasks.

The analysis produces:
1. Privacy-Utility Trade-off Curve
2. Optimal Budget Point Analysis
3. Adaptive Performance Visualization
4. Comprehensive Evaluation Reports
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from models.obfuscator_mask import MaskOnlyObfuscator
from models.obfuscator_feature import FeatureSpaceObfuscator
from models.target_model import TargetModel, UtilityTask
from models.adversary import IdentityAdversary
from adaptive_loss import AdaptiveLoss, PrivacyBudgetController
from three_part_training import ThreePartTrainer
from facenet_pytorch import MTCNN

class AdaptiveCurveAnalyzer:
    """
    Analyzer for generating privacy-utility trade-off curves.
    
    This class systematically varies the privacy budget parameter (γ)
    and measures the resulting performance to understand the optimal
    trade-off point.
    """
    
    def __init__(self,
                 device: torch.device,
                 num_identities: int = 8000,
                 num_utility_classes: int = 2,
                 utility_task: UtilityTask = UtilityTask.GENDER_CLASSIFICATION,
                 obf_base: int = 32,
                 obfuscator_type: str = 'mask'):
        
        self.device = device
        self.num_identities = num_identities
        self.num_utility_classes = num_utility_classes
        self.utility_task = utility_task
        self.obfuscator_type = obfuscator_type
        
        # Initialize models based on obfuscator type
        if obfuscator_type == 'feature':
            self.obfuscator = FeatureSpaceObfuscator(base=obf_base).to(device)
        else:
            self.obfuscator = MaskOnlyObfuscator(base=obf_base).to(device)
        self.target_model = TargetModel(
            task=utility_task,
            num_classes=num_utility_classes
        ).to(device)
        self.adversary = IdentityAdversary(
            num_identities=num_identities
        ).to(device)
        
        # Initialize face detector
        self.mtcnn = MTCNN(keep_all=False, device=device)
        
        # Analysis results
        self.analysis_results = []
        self.optimal_points = {}
        
    def generate_adaptive_curve(self,
                               dataloader: torch.utils.data.DataLoader,
                               budget_range: Tuple[float, float] = (0.01, 0.5),
                               num_budget_points: int = 20,
                               num_training_steps: int = 100,
                               save_dir: str = 'adaptive_curve_results') -> Dict:
        """
        Generate the adaptive privacy-utility trade-off curve.
        
        Args:
            dataloader: DataLoader with test data
            budget_range: Range of budget values to test (min, max)
            num_budget_points: Number of budget points to test
            num_training_steps: Number of training steps per budget point
            save_dir: Directory to save results
            
        Returns:
            analysis_results: Dictionary containing all analysis results
        """
        print("Generating Adaptive Privacy-Utility Trade-off Curve...")
        print(f"Budget range: {budget_range}")
        print(f"Number of budget points: {num_budget_points}")
        
        # Generate budget points
        budget_points = np.linspace(budget_range[0], budget_range[1], num_budget_points)
        
        # Initialize results storage
        self.analysis_results = []
        
        # Test each budget point
        for i, budget in enumerate(tqdm(budget_points, desc="Testing budget points")):
            print(f"\nTesting budget point {i+1}/{num_budget_points}: γ = {budget:.4f}")
            
            # Train obfuscator with current budget
            obfuscator_metrics = self._train_obfuscator_with_budget(
                dataloader, budget, num_training_steps
            )
            
            # Evaluate performance
            evaluation_metrics = self._evaluate_budget_point(dataloader, budget)
            
            # Combine results
            budget_result = {
                'budget': budget,
                'training_metrics': obfuscator_metrics,
                'evaluation_metrics': evaluation_metrics,
                'timestamp': time.time()
            }
            
            self.analysis_results.append(budget_result)
            
            # Print progress
            print(f"  Utility Accuracy: {evaluation_metrics['utility_accuracy']:.4f}")
            print(f"  Privacy Accuracy: {evaluation_metrics['privacy_accuracy']:.4f}")
            print(f"  Privacy Score: {evaluation_metrics['privacy_score']:.4f}")
            print(f"  Trade-off Score: {evaluation_metrics['tradeoff_score']:.4f}")
        
        # Analyze results
        analysis_summary = self._analyze_results()
        
        # Generate visualizations
        self._generate_visualizations(save_dir)
        
        # Save results
        self._save_results(save_dir)
        
        return analysis_summary
    
    def _train_obfuscator_with_budget(self,
                                    dataloader: torch.utils.data.DataLoader,
                                    budget: float,
                                    num_steps: int) -> Dict[str, float]:
        """Train obfuscator with specific budget parameter"""
        
        # Initialize adaptive loss with specific budget weight
        adaptive_loss = AdaptiveLoss(
            alpha=1.0,  # Utility weight
            beta=1.0,   # Privacy weight
            gamma=budget,  # Budget weight (this is what we're varying)
            target_utility_accuracy=0.9,
            target_privacy_accuracy=0.0
        ).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.obfuscator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999)
        )
        
        # Training loop
        self.obfuscator.train()
        training_metrics = {
            'loss': [],
            'utility_accuracy': [],
            'privacy_accuracy': [],
            'budget_usage': []
        }
        
        for step in range(num_steps):
            # Sample batch
            try:
                images, labels = next(iter(dataloader))
            except StopIteration:
                continue
            
            images = images.to(self.device)
            utility_labels = labels.to(self.device)
            identity_labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Generate obfuscated images
            if self.obfuscator_type == 'feature':
                obfuscated_images, masks = self.obfuscator(images)
            else:
                masks = self.obfuscator(images)
                obfuscated_images = self._apply_localized_obfuscation(images, masks)
            
            # Forward pass through models
            utility_logits = self.target_model(obfuscated_images)
            privacy_logits = self.adversary(obfuscated_images)
            
            # Compute adaptive loss
            loss, loss_components = adaptive_loss(
                obfuscated_images, masks, utility_logits, utility_labels,
                privacy_logits, identity_labels, step
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record metrics
            training_metrics['loss'].append(loss.item())
            training_metrics['utility_accuracy'].append(loss_components['utility_accuracy'])
            training_metrics['privacy_accuracy'].append(loss_components['privacy_accuracy'])
            training_metrics['budget_usage'].append(loss_components['budget_usage'])
        
        # Return final metrics
        return {
            'final_loss': training_metrics['loss'][-1] if training_metrics['loss'] else 0.0,
            'final_utility_accuracy': training_metrics['utility_accuracy'][-1] if training_metrics['utility_accuracy'] else 0.0,
            'final_privacy_accuracy': training_metrics['privacy_accuracy'][-1] if training_metrics['privacy_accuracy'] else 0.0,
            'final_budget_usage': training_metrics['budget_usage'][-1] if training_metrics['budget_usage'] else 0.0,
            'convergence_steps': len(training_metrics['loss'])
        }
    
    def _evaluate_budget_point(self,
                              dataloader: torch.utils.data.DataLoader,
                              budget: float) -> Dict[str, float]:
        """Evaluate performance at specific budget point"""
        
        self.obfuscator.eval()
        self.target_model.eval()
        self.adversary.eval()
        
        total_metrics = {
            'utility_accuracy': 0.0,
            'privacy_accuracy': 0.0,
            'privacy_score': 0.0,
            'utility_preservation': 0.0,
            'budget_usage': 0.0,
            'tradeoff_score': 0.0,
            'mask_sparsity': 0.0,
            'mask_concentration': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                utility_labels = labels.to(self.device)
                identity_labels = labels.to(self.device)
                
                # Generate obfuscated images
                if self.obfuscator_type == 'feature':
                    obfuscated_images, masks = self.obfuscator(images)
                else:
                    masks = self.obfuscator(images)
                    obfuscated_images = self._apply_localized_obfuscation(images, masks)
                
                # Evaluate target model
                utility_logits = self.target_model(obfuscated_images)
                utility_predicted = torch.argmax(utility_logits, dim=1)
                utility_accuracy = (utility_predicted == utility_labels).float().mean()
                
                # Evaluate adversary
                privacy_logits = self.adversary(obfuscated_images)
                privacy_predicted = torch.argmax(privacy_logits, dim=1)
                privacy_accuracy = (privacy_predicted == identity_labels).float().mean()
                
                # Compute additional metrics
                privacy_score = 1.0 - privacy_accuracy
                utility_preservation = utility_accuracy / 0.9  # Assuming target utility of 0.9
                budget_usage = torch.mean(masks).item()
                tradeoff_score = privacy_score * utility_preservation
                
                # Mask analysis
                mask_sparsity = torch.mean((masks > 0.1).float()).item()
                mask_concentration = self._compute_mask_concentration(masks)
                
                # Accumulate metrics
                total_metrics['utility_accuracy'] += utility_accuracy.item()
                total_metrics['privacy_accuracy'] += privacy_accuracy.item()
                total_metrics['privacy_score'] += privacy_score.item()
                total_metrics['utility_preservation'] += utility_preservation.item()
                total_metrics['budget_usage'] += budget_usage
                total_metrics['tradeoff_score'] += tradeoff_score
                total_metrics['mask_sparsity'] += mask_sparsity
                total_metrics['mask_concentration'] += mask_concentration
                
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return total_metrics
    
    def _compute_mask_concentration(self, masks: torch.Tensor) -> float:
        """Compute how concentrated the mask is in the center"""
        B, C, H, W = masks.shape
        
        # Create center weight matrix
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, device=masks.device).float()
        x_coords = torch.arange(W, device=masks.device).float()
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Distance from center
        distance = torch.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
        max_distance = torch.sqrt(torch.tensor(center_y ** 2 + center_x ** 2, device=masks.device).float())
        center_weights = 1.0 - (distance / max_distance)
        
        # Compute concentration
        weighted_masks = masks * center_weights.unsqueeze(0).unsqueeze(0)
        concentration = torch.mean(weighted_masks).item()
        
        return concentration
    
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
    
    def _analyze_results(self) -> Dict:
        """Analyze the adaptive curve results"""
        
        # Extract data
        budgets = [result['budget'] for result in self.analysis_results]
        utility_accuracies = [result['evaluation_metrics']['utility_accuracy'] for result in self.analysis_results]
        privacy_accuracies = [result['evaluation_metrics']['privacy_accuracy'] for result in self.analysis_results]
        privacy_scores = [result['evaluation_metrics']['privacy_score'] for result in self.analysis_results]
        tradeoff_scores = [result['evaluation_metrics']['tradeoff_score'] for result in self.analysis_results]
        budget_usages = [result['evaluation_metrics']['budget_usage'] for result in self.analysis_results]
        
        # Find optimal points
        optimal_tradeoff_idx = np.argmax(tradeoff_scores)
        optimal_privacy_idx = np.argmax(privacy_scores)
        optimal_utility_idx = np.argmax(utility_accuracies)
        
        self.optimal_points = {
            'best_tradeoff': {
                'budget': budgets[optimal_tradeoff_idx],
                'utility_accuracy': utility_accuracies[optimal_tradeoff_idx],
                'privacy_accuracy': privacy_accuracies[optimal_tradeoff_idx],
                'privacy_score': privacy_scores[optimal_tradeoff_idx],
                'tradeoff_score': tradeoff_scores[optimal_tradeoff_idx],
                'budget_usage': budget_usages[optimal_tradeoff_idx]
            },
            'best_privacy': {
                'budget': budgets[optimal_privacy_idx],
                'utility_accuracy': utility_accuracies[optimal_privacy_idx],
                'privacy_accuracy': privacy_accuracies[optimal_privacy_idx],
                'privacy_score': privacy_scores[optimal_privacy_idx],
                'tradeoff_score': tradeoff_scores[optimal_privacy_idx],
                'budget_usage': budget_usages[optimal_privacy_idx]
            },
            'best_utility': {
                'budget': budgets[optimal_utility_idx],
                'utility_accuracy': utility_accuracies[optimal_utility_idx],
                'privacy_accuracy': privacy_accuracies[optimal_utility_idx],
                'privacy_score': privacy_scores[optimal_utility_idx],
                'tradeoff_score': tradeoff_scores[optimal_utility_idx],
                'budget_usage': budget_usages[optimal_utility_idx]
            }
        }
        
        # Compute statistics
        analysis_summary = {
            'budget_range': (min(budgets), max(budgets)),
            'num_budget_points': len(budgets),
            'optimal_points': self.optimal_points,
            'statistics': {
                'utility_accuracy_range': (min(utility_accuracies), max(utility_accuracies)),
                'privacy_accuracy_range': (min(privacy_accuracies), max(privacy_accuracies)),
                'privacy_score_range': (min(privacy_scores), max(privacy_scores)),
                'tradeoff_score_range': (min(tradeoff_scores), max(tradeoff_scores)),
                'budget_usage_range': (min(budget_usages), max(budget_usages))
            },
            'recommendations': self._generate_recommendations()
        }
        
        return analysis_summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        best_tradeoff = self.optimal_points['best_tradeoff']
        best_privacy = self.optimal_points['best_privacy']
        best_utility = self.optimal_points['best_utility']
        
        recommendations.append(f"Optimal trade-off point: Budget = {best_tradeoff['budget']:.4f}")
        recommendations.append(f"  - Utility Accuracy: {best_tradeoff['utility_accuracy']:.4f}")
        recommendations.append(f"  - Privacy Score: {best_tradeoff['privacy_score']:.4f}")
        recommendations.append(f"  - Trade-off Score: {best_tradeoff['tradeoff_score']:.4f}")
        
        recommendations.append(f"Best privacy point: Budget = {best_privacy['budget']:.4f}")
        recommendations.append(f"  - Privacy Score: {best_privacy['privacy_score']:.4f}")
        recommendations.append(f"  - Utility Accuracy: {best_privacy['utility_accuracy']:.4f}")
        
        recommendations.append(f"Best utility point: Budget = {best_utility['budget']:.4f}")
        recommendations.append(f"  - Utility Accuracy: {best_utility['utility_accuracy']:.4f}")
        recommendations.append(f"  - Privacy Score: {best_utility['privacy_score']:.4f}")
        
        # Analyze trade-off curve
        if best_tradeoff['tradeoff_score'] > 0.8:
            recommendations.append("Excellent trade-off achieved! System provides strong privacy with good utility.")
        elif best_tradeoff['tradeoff_score'] > 0.6:
            recommendations.append("Good trade-off achieved. Consider fine-tuning for better performance.")
        else:
            recommendations.append("Trade-off needs improvement. Consider adjusting loss weights or training strategy.")
        
        return recommendations
    
    def _generate_visualizations(self, save_dir: str):
        """Generate comprehensive visualizations"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        budgets = [result['budget'] for result in self.analysis_results]
        utility_accuracies = [result['evaluation_metrics']['utility_accuracy'] for result in self.analysis_results]
        privacy_accuracies = [result['evaluation_metrics']['privacy_accuracy'] for result in self.analysis_results]
        privacy_scores = [result['evaluation_metrics']['privacy_score'] for result in self.analysis_results]
        tradeoff_scores = [result['evaluation_metrics']['tradeoff_score'] for result in self.analysis_results]
        budget_usages = [result['evaluation_metrics']['budget_usage'] for result in self.analysis_results]
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Privacy-Utility Trade-off Curve
        scatter = axes[0, 0].scatter(utility_accuracies, privacy_accuracies, 
                                    c=budgets, cmap='viridis', s=100, alpha=0.7)
        axes[0, 0].set_xlabel('Utility Accuracy')
        axes[0, 0].set_ylabel('Privacy Accuracy')
        axes[0, 0].set_title('Privacy-Utility Trade-off Curve')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Budget (γ)')
        
        # Mark optimal points
        best_tradeoff = self.optimal_points['best_tradeoff']
        best_privacy = self.optimal_points['best_privacy']
        best_utility = self.optimal_points['best_utility']
        
        axes[0, 0].scatter(best_tradeoff['utility_accuracy'], best_tradeoff['privacy_accuracy'], 
                          c='red', s=200, marker='*', label='Best Trade-off')
        axes[0, 0].scatter(best_privacy['utility_accuracy'], best_privacy['privacy_accuracy'], 
                          c='blue', s=200, marker='^', label='Best Privacy')
        axes[0, 0].scatter(best_utility['utility_accuracy'], best_utility['privacy_accuracy'], 
                          c='green', s=200, marker='s', label='Best Utility')
        axes[0, 0].legend()
        
        # 2. Budget vs Performance
        axes[0, 1].plot(budgets, utility_accuracies, 'b-o', label='Utility Accuracy', alpha=0.7)
        axes[0, 1].plot(budgets, privacy_scores, 'r-o', label='Privacy Score', alpha=0.7)
        axes[0, 1].set_xlabel('Budget (γ)')
        axes[0, 1].set_ylabel('Performance')
        axes[0, 1].set_title('Budget vs Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Trade-off Score
        axes[0, 2].plot(budgets, tradeoff_scores, 'g-o', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Budget (γ)')
        axes[0, 2].set_ylabel('Trade-off Score')
        axes[0, 2].set_title('Trade-off Score vs Budget')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Mark optimal point
        axes[0, 2].scatter(best_tradeoff['budget'], best_tradeoff['tradeoff_score'], 
                          c='red', s=200, marker='*', zorder=5)
        
        # 4. Budget Usage
        axes[1, 0].plot(budgets, budget_usages, 'm-o', alpha=0.7)
        axes[1, 0].set_xlabel('Budget (γ)')
        axes[1, 0].set_ylabel('Actual Budget Usage')
        axes[1, 0].set_title('Budget Parameter vs Actual Usage')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Performance Heatmap
        budget_grid = np.linspace(min(budgets), max(budgets), 20)
        utility_grid = np.linspace(min(utility_accuracies), max(utility_accuracies), 20)
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(budgets, utility_accuracies, bins=[budget_grid, utility_grid])
        
        im = axes[1, 1].imshow(hist.T, extent=[min(budgets), max(budgets), 
                                              min(utility_accuracies), max(utility_accuracies)], 
                              aspect='auto', origin='lower', cmap='Blues')
        axes[1, 1].set_xlabel('Budget (γ)')
        axes[1, 1].set_ylabel('Utility Accuracy')
        axes[1, 1].set_title('Budget-Utility Distribution')
        plt.colorbar(im, ax=axes[1, 1], label='Frequency')
        
        # 6. Summary Statistics
        axes[1, 2].axis('off')
        
        # Create summary text
        summary_text = f"""
        Adaptive Curve Analysis Summary
        
        Budget Range: {min(budgets):.4f} - {max(budgets):.4f}
        Number of Points: {len(budgets)}
        
        Optimal Trade-off:
        Budget: {best_tradeoff['budget']:.4f}
        Utility: {best_tradeoff['utility_accuracy']:.4f}
        Privacy: {best_tradeoff['privacy_score']:.4f}
        Score: {best_tradeoff['tradeoff_score']:.4f}
        
        Best Privacy:
        Budget: {best_privacy['budget']:.4f}
        Privacy: {best_privacy['privacy_score']:.4f}
        
        Best Utility:
        Budget: {best_utility['budget']:.4f}
        Utility: {best_utility['utility_accuracy']:.4f}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path / 'adaptive_curve_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual plots
        self._create_individual_plots(save_path, budgets, utility_accuracies, 
                                    privacy_accuracies, privacy_scores, tradeoff_scores, budget_usages)
        
        print(f"Visualizations saved to {save_path}")
    
    def _create_individual_plots(self, save_path: Path, budgets: List[float], 
                               utility_accuracies: List[float], privacy_accuracies: List[float],
                               privacy_scores: List[float], tradeoff_scores: List[float], 
                               budget_usages: List[float]):
        """Create individual plots for detailed analysis"""
        
        # Privacy-Utility Trade-off Curve
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(utility_accuracies, privacy_accuracies, c=budgets, 
                            cmap='viridis', s=100, alpha=0.7)
        plt.xlabel('Utility Accuracy')
        plt.ylabel('Privacy Accuracy')
        plt.title('Privacy-Utility Trade-off Curve')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Budget (γ)')
        
        # Mark optimal points
        best_tradeoff = self.optimal_points['best_tradeoff']
        best_privacy = self.optimal_points['best_privacy']
        best_utility = self.optimal_points['best_utility']
        
        plt.scatter(best_tradeoff['utility_accuracy'], best_tradeoff['privacy_accuracy'], 
                   c='red', s=200, marker='*', label='Best Trade-off')
        plt.scatter(best_privacy['utility_accuracy'], best_privacy['privacy_accuracy'], 
                   c='blue', s=200, marker='^', label='Best Privacy')
        plt.scatter(best_utility['utility_accuracy'], best_utility['privacy_accuracy'], 
                   c='green', s=200, marker='s', label='Best Utility')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path / 'privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Trade-off Score
        plt.figure(figsize=(10, 6))
        plt.plot(budgets, tradeoff_scores, 'g-o', linewidth=2, markersize=8)
        plt.xlabel('Budget (γ)')
        plt.ylabel('Trade-off Score')
        plt.title('Trade-off Score vs Budget')
        plt.grid(True, alpha=0.3)
        
        # Mark optimal point
        plt.scatter(best_tradeoff['budget'], best_tradeoff['tradeoff_score'], 
                   c='red', s=200, marker='*', zorder=5)
        
        plt.tight_layout()
        plt.savefig(save_path / 'tradeoff_score.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, save_dir: str):
        """Save analysis results"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(save_path / 'analysis_results.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save optimal points
        with open(save_path / 'optimal_points.json', 'w') as f:
            json.dump(self.optimal_points, f, indent=2, default=str)
        
        # Save summary
        analysis_summary = self._analyze_results()
        with open(save_path / 'analysis_summary.json', 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        # Save CSV for easy analysis
        df_data = []
        for result in self.analysis_results:
            row = {
                'budget': result['budget'],
                **result['evaluation_metrics']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(save_path / 'analysis_results.csv', index=False)
        
        print(f"Results saved to {save_path}")

def main():
    """Example usage of the adaptive curve analyzer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize analyzer
    analyzer = AdaptiveCurveAnalyzer(
        device=device,
        num_identities=1000,  # Adjust based on your dataset
        num_utility_classes=2,  # Binary gender classification
        utility_task=UtilityTask.GENDER_CLASSIFICATION
    )
    
    # Create dummy dataloader (replace with your actual dataloader)
    from torch.utils.data import TensorDataset, DataLoader
    
    # Dummy data
    images = torch.randn(200, 3, 160, 160)
    labels = torch.randint(0, 2, (200,))  # Binary labels
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Generate adaptive curve
    analysis_summary = analyzer.generate_adaptive_curve(
        dataloader=dataloader,
        budget_range=(0.01, 0.3),
        num_budget_points=10,
        num_training_steps=50,
        save_dir='adaptive_curve_results'
    )
    
    print("\nAnalysis Summary:")
    print(json.dumps(analysis_summary, indent=2))

if __name__ == '__main__':
    main()
