#!/usr/bin/env python3
"""
Adaptive Obfuscation Mechanism

This module implements an adaptive obfuscation system that dynamically adjusts
the strength of obfuscation based on privacy and utility metrics.

The system:
1. Monitors identity similarity (privacy metric)
2. Monitors downstream task performance (utility metric)
3. Adjusts obfuscation parameters to maintain optimal privacy-utility tradeoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import kornia
from facenet_pytorch import MTCNN

class ObfuscationMethod(Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    PIXELATION = "pixelation"
    PATCH_SHUFFLE = "patch_shuffle"
    ADVERSARIAL_NOISE = "adversarial_noise"

@dataclass
class PrivacyUtilityMetrics:
    """Container for privacy and utility metrics"""
    identity_similarity: float
    privacy_score: float
    utility_accuracy: float
    utility_loss: float
    timestamp: float

@dataclass
class ObfuscationParams:
    """Container for obfuscation parameters"""
    method: ObfuscationMethod
    sigma: Optional[float] = None
    block_size: Optional[int] = None
    patch_size: Optional[int] = None
    noise_std: Optional[float] = None

class AdaptiveObfuscator:
    """Adaptive obfuscation system that adjusts parameters based on metrics"""
    
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 privacy_threshold: float = 0.3,
                 utility_threshold: float = 0.8,
                 adaptation_rate: float = 0.1):
        self.device = device
        self.privacy_threshold = privacy_threshold  # Max acceptable identity similarity
        self.utility_threshold = utility_threshold  # Min acceptable utility preservation
        self.adaptation_rate = adaptation_rate
        
        # Current obfuscation parameters
        self.current_params = {
            ObfuscationMethod.GAUSSIAN_BLUR: {'sigma': 8.0},
            ObfuscationMethod.PIXELATION: {'block_size': 16},
            ObfuscationMethod.PATCH_SHUFFLE: {'patch_size': 16},
            ObfuscationMethod.ADVERSARIAL_NOISE: {'noise_std': 0.1}
        }
        
        # Parameter ranges for adaptation
        self.param_ranges = {
            ObfuscationMethod.GAUSSIAN_BLUR: {'sigma': (2.0, 20.0)},
            ObfuscationMethod.PIXELATION: {'block_size': (4, 32)},
            ObfuscationMethod.PATCH_SHUFFLE: {'patch_size': (4, 32)},
            ObfuscationMethod.ADVERSARIAL_NOISE: {'noise_std': (0.01, 0.5)}
        }
        
        # Metrics history for trend analysis
        self.metrics_history: List[PrivacyUtilityMetrics] = []
        self.mtcnn = MTCNN(keep_all=False, device=device)
        
        # Adaptation state
        self.adaptation_enabled = True
        self.last_adaptation_step = 0
    
    def apply_obfuscation(self, image: torch.Tensor, method: ObfuscationMethod, **kwargs) -> torch.Tensor:
        """Apply specified obfuscation method to image"""
        H, W = image.shape[-2], image.shape[-1]
        
        if method == ObfuscationMethod.GAUSSIAN_BLUR:
            sigma = kwargs.get('sigma', 8.0)
            k_desired = max(3, int(2 * (3 * sigma) + 1))
            k_max = max(3, min(int(H), int(W)))
            k = min(k_desired, k_max)
            k = k if (k % 2 == 1) else (k - 1)
            k = max(3, k)
            
            sigma_eff = float(sigma)
            if k < k_desired:
                sigma_eff = max(0.5, (k - 1) / (2.0 * 3.0))
            
            gb = kornia.filters.GaussianBlur2d((k, k), (sigma_eff, sigma_eff))
            return gb(image)
        
        elif method == ObfuscationMethod.PIXELATION:
            block_size = kwargs.get('block_size', 16)
            img_small = F.avg_pool2d(image, kernel_size=block_size)
            return F.interpolate(img_small, size=image.shape[-2:], mode='nearest')
        
        elif method == ObfuscationMethod.PATCH_SHUFFLE:
            patch_size = kwargs.get('patch_size', 16)
            B, C, H, W = image.shape
            pad_h = (patch_size - (H % patch_size)) % patch_size
            pad_w = (patch_size - (W % patch_size)) % patch_size
            
            x = image
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
            HB, WB = x.shape[-2], x.shape[-1]
            gh, gw = HB // patch_size, WB // patch_size
            
            patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous()
            patches = patches.view(B, C, gh*gw, patch_size, patch_size)
            idx = torch.randperm(gh*gw, device=image.device)
            patches = patches[:, :, idx, :, :]
            patches = patches.view(B, C, gh, gw, patch_size, patch_size)
            
            rows = []
            for i in range(gh):
                row = torch.cat([patches[:, :, i, j, :, :] for j in range(gw)], dim=3)
                rows.append(row)
            xrec = torch.cat(rows, dim=2)
            return xrec[:, :, :H, :W]
        
        elif method == ObfuscationMethod.ADVERSARIAL_NOISE:
            noise_std = kwargs.get('noise_std', 0.1)
            noise = torch.randn_like(image) * noise_std
            return torch.clamp(image + noise, 0, 1)
        
        else:
            raise ValueError(f"Unknown obfuscation method: {method}")
    
    def apply_localized_obfuscation(self, image: torch.Tensor, method: ObfuscationMethod, **kwargs) -> torch.Tensor:
        """Apply obfuscation only to detected face regions"""
        B, C, H, W = image.shape
        result = image.clone()
        
        for bi in range(B):
            img_i = image[bi:bi+1]
            
            # Convert to numpy for MTCNN
            img_np = (img_i[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8')
            boxes, _ = self.mtcnn.detect([img_np])
            
            if boxes is None or len(boxes[0]) == 0:
                continue  # No face detected, keep original
            
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
            obfuscated_crop = self.apply_obfuscation(face_crop, method, **kwargs)
            
            # Paste back
            result[bi, :, y1:y2, x1:x2] = obfuscated_crop[0]
        
        return result
    
    def update_metrics(self, metrics: PrivacyUtilityMetrics):
        """Update metrics history and trigger adaptation if needed"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history (last 100 measurements)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Trigger adaptation if enabled
        if self.adaptation_enabled:
            self._adapt_parameters(metrics)
    
    def _adapt_parameters(self, metrics: PrivacyUtilityMetrics):
        """Adapt obfuscation parameters based on current metrics"""
        adaptation_needed = False
        
        # Check privacy threshold
        if metrics.identity_similarity > self.privacy_threshold:
            # Privacy insufficient, increase obfuscation strength
            self._increase_obfuscation_strength()
            adaptation_needed = True
        
        # Check utility threshold
        elif metrics.utility_accuracy < self.utility_threshold:
            # Utility insufficient, decrease obfuscation strength
            self._decrease_obfuscation_strength()
            adaptation_needed = True
        
        # Check for trend-based adaptation
        elif len(self.metrics_history) >= 5:
            # Analyze recent trends
            recent_metrics = self.metrics_history[-5:]
            privacy_trend = self._compute_trend([m.identity_similarity for m in recent_metrics])
            utility_trend = self._compute_trend([m.utility_accuracy for m in recent_metrics])
            
            if privacy_trend > 0.1:  # Privacy degrading
                self._increase_obfuscation_strength()
                adaptation_needed = True
            elif utility_trend < -0.1:  # Utility degrading
                self._decrease_obfuscation_strength()
                adaptation_needed = True
        
        if adaptation_needed:
            self.last_adaptation_step = len(self.metrics_history)
            print(f"Adapted obfuscation parameters at step {self.last_adaptation_step}")
            self._print_current_params()
    
    def _increase_obfuscation_strength(self):
        """Increase obfuscation strength across all methods"""
        for method, params in self.current_params.items():
            if method == ObfuscationMethod.GAUSSIAN_BLUR:
                params['sigma'] = min(params['sigma'] * (1 + self.adaptation_rate), 
                                    self.param_ranges[method]['sigma'][1])
            elif method == ObfuscationMethod.PIXELATION:
                params['block_size'] = min(int(params['block_size'] * (1 + self.adaptation_rate)), 
                                          self.param_ranges[method]['block_size'][1])
            elif method == ObfuscationMethod.PATCH_SHUFFLE:
                params['patch_size'] = min(int(params['patch_size'] * (1 + self.adaptation_rate)), 
                                          self.param_ranges[method]['patch_size'][1])
            elif method == ObfuscationMethod.ADVERSARIAL_NOISE:
                params['noise_std'] = min(params['noise_std'] * (1 + self.adaptation_rate), 
                                        self.param_ranges[method]['noise_std'][1])
    
    def _decrease_obfuscation_strength(self):
        """Decrease obfuscation strength across all methods"""
        for method, params in self.current_params.items():
            if method == ObfuscationMethod.GAUSSIAN_BLUR:
                params['sigma'] = max(params['sigma'] * (1 - self.adaptation_rate), 
                                    self.param_ranges[method]['sigma'][0])
            elif method == ObfuscationMethod.PIXELATION:
                params['block_size'] = max(int(params['block_size'] * (1 - self.adaptation_rate)), 
                                          self.param_ranges[method]['block_size'][0])
            elif method == ObfuscationMethod.PATCH_SHUFFLE:
                params['patch_size'] = max(int(params['patch_size'] * (1 - self.adaptation_rate)), 
                                          self.param_ranges[method]['patch_size'][0])
            elif method == ObfuscationMethod.ADVERSARIAL_NOISE:
                params['noise_std'] = max(params['noise_std'] * (1 - self.adaptation_rate), 
                                        self.param_ranges[method]['noise_std'][0])
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend (slope) of values over time"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _print_current_params(self):
        """Print current obfuscation parameters"""
        print("Current obfuscation parameters:")
        for method, params in self.current_params.items():
            print(f"  {method.value}: {params}")
    
    def get_current_params(self, method: ObfuscationMethod) -> Dict[str, Any]:
        """Get current parameters for a specific method"""
        return self.current_params[method].copy()
    
    def set_adaptation_enabled(self, enabled: bool):
        """Enable or disable adaptation"""
        self.adaptation_enabled = enabled
        print(f"Adaptation {'enabled' if enabled else 'disabled'}")
    
    def set_thresholds(self, privacy_threshold: float, utility_threshold: float):
        """Update adaptation thresholds"""
        self.privacy_threshold = privacy_threshold
        self.utility_threshold = utility_threshold
        print(f"Updated thresholds - Privacy: {privacy_threshold}, Utility: {utility_threshold}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            'avg_privacy_score': np.mean([m.privacy_score for m in recent_metrics]),
            'avg_utility_accuracy': np.mean([m.utility_accuracy for m in recent_metrics]),
            'avg_utility_loss': np.mean([m.utility_loss for m in recent_metrics]),
            'privacy_trend': self._compute_trend([m.privacy_score for m in recent_metrics]),
            'utility_trend': self._compute_trend([m.utility_accuracy for m in recent_metrics]),
            'total_measurements': len(self.metrics_history),
            'last_adaptation_step': self.last_adaptation_step
        }

# Example usage and testing
if __name__ == '__main__':
    # Initialize adaptive obfuscator
    adaptive_obf = AdaptiveObfuscator(
        privacy_threshold=0.3,
        utility_threshold=0.8,
        adaptation_rate=0.1
    )
    
    # Simulate some metrics updates
    import time
    
    for i in range(10):
        # Simulate varying metrics
        identity_sim = 0.2 + 0.1 * np.sin(i * 0.5) + np.random.normal(0, 0.05)
        utility_acc = 0.9 - 0.1 * np.sin(i * 0.3) + np.random.normal(0, 0.02)
        
        metrics = PrivacyUtilityMetrics(
            identity_similarity=identity_sim,
            privacy_score=1.0 - identity_sim,
            utility_accuracy=utility_acc,
            utility_loss=0.9 - utility_acc,
            timestamp=time.time()
        )
        
        adaptive_obf.update_metrics(metrics)
        print(f"Step {i}: Privacy={metrics.privacy_score:.3f}, Utility={metrics.utility_accuracy:.3f}")
    
    # Print final summary
    summary = adaptive_obf.get_metrics_summary()
    print("\nFinal Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
