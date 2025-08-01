#!/usr/bin/env python3
"""
MAC Operations Calculator for 3D Sparse Convolution Networks with Advanced Sparsity Modeling
Supports: MinkowskiNet, SPVNAS

Network configurations sourced from official repositories:
- MinkowskiNet: https://github.com/NVIDIA/MinkowskiEngine (Paper: https://arxiv.org/abs/1904.08755)
- SPVNAS: https://github.com/mit-han-lab/spvnas (Paper: https://arxiv.org/abs/2007.16100)

Advanced sparsity modeling includes:
1. Spatial sparsity: Non-zero voxel occupancy (typical LiDAR: 3-7%)
2. Channel sparsity: Pruned/inactive channels (typical: 20-50%)
3. Feature sparsity: Zero activations after ReLU (typical: 30-70%)
4. Structured sparsity: Block-wise or group-wise sparsity patterns
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from typing import Dict, List, Tuple, Union, Optional

class AdvancedSparseConv3DMACCalculator:
    """Calculate MAC operations for sparse 3D convolution operations with multiple sparsity types"""
    
    @staticmethod
    def sparse_conv3d_macs(num_active_voxels: int, kernel_size: int, 
                          in_channels: int, out_channels: int, 
                          spatial_sparsity: float = 0.05,
                          channel_sparsity: float = 0.3,
                          feature_sparsity: float = 0.5,
                          structured_sparsity: float = 0.0) -> Tuple[int, int]:
        """
        Calculate MACs for 3D sparse convolution with advanced sparsity modeling
        
        Args:
            num_active_voxels: Number of non-empty voxels in the sparse tensor
            kernel_size: Size of the 3D convolution kernel (assumes cubic)
            in_channels: Number of input channels
            out_channels: Number of output channels
            spatial_sparsity: Ratio of neighbors per output voxel that are non-zero (0.05 = 5% of kernel positions have data)
            channel_sparsity: Ratio of pruned channels (0.0-0.8)
            feature_sparsity: Ratio of zero activations after ReLU (0.0-0.8)
            structured_sparsity: Additional structured pruning (0.0-0.5)
        
        Returns:
            Tuple of (MAC operations, output_voxels)
        """
        kernel_volume = kernel_size ** 3
        
        # Effective channels after channel pruning
        effective_in_channels = int(in_channels * (1 - channel_sparsity))
        effective_out_channels = int(out_channels * (1 - channel_sparsity))
        
        # Output voxels = input voxels (no downsampling in this corrected version)
        output_voxels = num_active_voxels
        
        # Spatial sparsity affects the number of active kernel positions per output voxel
        # Each output voxel looks at kernel_volume positions, but only spatial_sparsity fraction have data
        active_kernel_positions_per_output = max(1, int(kernel_volume * spatial_sparsity)* 5)
        
        # Total MAC operations: output_voxels × active_kernel_positions × channel_pairs
        base_macs = output_voxels * active_kernel_positions_per_output * effective_in_channels * effective_out_channels
        
        # Apply feature sparsity (activations that are zero don't contribute)
        feature_effective_macs = int(base_macs * (1 - feature_sparsity))
        
        # Apply structured sparsity (additional pruning patterns)
        final_macs = int(feature_effective_macs * (1 - structured_sparsity))
        
        return final_macs, output_voxels
    
    @staticmethod
    def calculate_active_pairs(input_voxels: int, kernel_size: int, 
                              spatial_sparsity: float, stride: int = 1,
                              clustering_factor: float = 3.5) -> Tuple[int, int]:
        """
        Calculate the number of active input-output voxel pairs based on kernel overlap
        
        Args:
            input_voxels: Number of active input voxels (already sparse)
            kernel_size: Size of the convolution kernel
            spatial_sparsity: Fraction of kernel positions that have non-zero data per output voxel
            stride: Convolution stride (disabled for this correction)
            clustering_factor: Factor accounting for local clustering of points (1.0-10.0)
            
        Returns:
            Tuple of (active_pairs, output_voxels)
        """
        kernel_volume = kernel_size ** 3
        
        # Output voxels = input voxels (no downsampling)
        output_voxels = input_voxels
        
        # Active kernel positions per output voxel
        # spatial_sparsity determines how many of the kernel_volume positions actually have data
        avg_active_kernel_positions = max(1, int(kernel_volume * spatial_sparsity * clustering_factor))
        avg_active_kernel_positions = min(avg_active_kernel_positions, kernel_volume)
        
        # Total active pairs: each output voxel × average active kernel positions
        active_pairs = output_voxels * avg_active_kernel_positions
        
        return active_pairs, output_voxels
    
    @staticmethod
    def point_to_voxel_macs(num_points: int, voxel_size: float, 
                           point_features: int,
                           feature_sparsity: float = 0.3) -> int:
        """Calculate MACs for point-to-voxel conversion with feature sparsity"""
        base_macs = num_points * point_features
        return int(base_macs * (1 - feature_sparsity))
    
    @staticmethod
    def voxel_to_point_macs(num_voxels: int, num_points: int, 
                           voxel_features: int,
                           feature_sparsity: float = 0.3) -> int:
        """Calculate MACs for voxel-to-point feature propagation with sparsity"""
        base_macs = num_points * 8 * voxel_features  # trilinear interpolation
        return int(base_macs * (1 - feature_sparsity))
    
    @staticmethod
    def sparse_residual_block_macs(num_active_voxels: int, channels: int,
                                  spatial_sparsity: float = 0.05,
                                  channel_sparsity: float = 0.3,
                                  feature_sparsity: float = 0.5) -> Tuple[int, int]:
        """Calculate MACs for a sparse residual block with advanced sparsity"""
        # Conv1: 1x1x1 convolution
        conv1_macs, voxels_after_conv1 = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
            num_active_voxels, 1, channels, channels, 
            spatial_sparsity, channel_sparsity, feature_sparsity)
        
        # Conv2: 3x3x3 convolution
        conv2_macs, voxels_after_conv2 = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
            voxels_after_conv1, 3, channels, channels, 
            spatial_sparsity, channel_sparsity, feature_sparsity)
        
        total_macs = conv1_macs + conv2_macs
        return total_macs, voxels_after_conv2

class GEMMLayerGenerator:
    """Generate GEMM format data for sparse 3D convolution layers"""
    
    def __init__(self):
        self.layers = []
    
    def add_gemm_layer(self, layer_name: str, M: int, N: int, K: int, 
                      active_pairs: int = None, kernel_size: int = None):
        """
        Add a GEMM layer with dimensions M, N, K for sparse 3D convolution
        
        Args:
            layer_name: Name of the layer
            M: Number of output elements (active_output_voxels * output_channels)
            N: Batch size or number of parallel computations
            K: Number of input features per output element (input_channels for sparse conv)
            active_pairs: Number of active input-output voxel pairs (optional)
            kernel_size: Size of the convolution kernel (optional)
            
        Note: For sparse 3D convolution, the GEMM formulation is:
        - M: Total number of output activations that need to be computed
        - N: Batch size (typically 1 for point clouds)
        - K: Input feature dimension per output computation
        - The actual computation is much sparser than M×N×K due to spatial sparsity
        """
        layer_data = {
            'Layer': layer_name,
            'M': M,
            'N': N, 
            'K': K
        }
        
        # Add optional metadata for sparse convolution analysis
        if active_pairs is not None:
            layer_data['Active_Pairs'] = active_pairs
        if kernel_size is not None:
            layer_data['Kernel_Size'] = kernel_size
            
        self.layers.append(layer_data)
    
    def save_csv(self, filename: str):
        """Save GEMM data to CSV file"""
        filepath = os.path.join('/Users/ashriram/Desktop/minuet-tracer/mac-vs-time', filename)
        with open(filepath, 'w', newline='') as csvfile:
            if self.layers:
                # Include all keys that appear in any layer
                all_keys = set()
                for layer in self.layers:
                    all_keys.update(layer.keys())
                fieldnames = ['Layer', 'M', 'N', 'K'] + [k for k in sorted(all_keys) if k not in ['Layer', 'M', 'N', 'K']]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.layers)
        print(f"GEMM data saved to {filepath}")
    
    def clear(self):
        """Clear all layer data"""
        self.layers = []

class MinkowskiNet:
    """MinkowskiNet network configuration and MAC calculation with advanced sparsity modeling"""
    
    def __init__(self, num_points: int = 100000, voxel_size: float = 0.05,
                 input_channels: int = 4, num_classes: int = 20,
                 spatial_sparsity: float = 0.05,
                 channel_sparsity: float = 0.3,
                 feature_sparsity: float = 0.5,
                 clustering_factor: float = 3.0):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        # spatial_sparsity: fraction of kernel positions that have non-zero data per output voxel
        self.spatial_sparsity = spatial_sparsity
        self.channel_sparsity = channel_sparsity
        self.feature_sparsity = feature_sparsity
        # clustering_factor: how clustered the LiDAR points are (affects kernel matches)
        # 1.0 = uniform distribution, 3.0 = moderate clustering, 5.0+ = high clustering
        self.clustering_factor = clustering_factor
        
        # Since downsampling is disabled, each point corresponds to one active voxel
        # In sparse 3D convolution, we process each point as an active voxel
        self.num_voxels = num_points
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for MinkowskiNet with advanced sparsity - sparse conv layers only"""
        total_macs = 0
        layer_details = {}
        current_voxels = self.num_voxels
        current_channels = self.input_channels
        
        # Initial sparse convolution: input_channels -> 32 (following MinkUNet)
        macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
            current_voxels, 3, current_channels, 32, 
            self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
        total_macs += macs
        current_channels = 32
        layer_details['conv0'] = {
            'macs': macs, 'output_voxels': current_voxels, 'channels': current_channels
        }
        
        # Encoder stages - sparse conv only (NO DOWNSAMPLING)
        encoder_configs = [
            (32, 64, 1),   # stage1: no downsample
            (64, 128, 1),  # stage2: no downsample
            (128, 256, 1), # stage3: no downsample
            (256, 512, 1), # stage4: no downsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(encoder_configs):
            # No downsampling - voxels remain the same
            
            # Main convolution
            macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
                current_voxels, 3, in_ch, out_ch, 
                self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
            total_macs += macs
            current_channels = out_ch
            
            layer_details[f'encoder_stage_{i+1}'] = {
                'macs': macs, 'output_voxels': current_voxels, 'channels': current_channels
            }
        
        # Decoder stages - sparse conv only (NO UPSAMPLING)
        decoder_configs = [
            (512, 256, 1),  # stage1: no upsample
            (256, 128, 1),  # stage2: no upsample
            (128, 64, 1),   # stage3: no upsample
            (64, 32, 1),    # stage4: no upsample
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            # No upsampling - voxels remain the same
            
            # Convolution
            macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
                current_voxels, 3, in_ch, out_ch, 
                self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
            total_macs += macs
            current_channels = out_ch
            
            layer_details[f'decoder_stage_{i+1}'] = {
                'macs': macs, 'output_voxels': current_voxels, 'channels': current_channels
            }
        
        # Final classification head
        final_macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
            current_voxels, 1, current_channels, self.num_classes, 
            self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
        total_macs += final_macs
        layer_details['classification_head'] = {
            'macs': final_macs, 'output_voxels': current_voxels, 'channels': self.num_classes
        }
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'layer_details': layer_details,
            'final_output_voxels': current_voxels
        }
    
    def generate_gemm_data(self, gemm_generator: GEMMLayerGenerator, batch_size: int = 1):
        """Generate GEMM data for MinkowskiNet layers following Minuet sparse tensor GEMM formulation"""
        current_voxels = self.num_voxels
        current_channels = self.input_channels
        
        # Initial sparse convolution: input_channels -> 32
        kernel_size = 3
        kernel_volume = kernel_size ** 3
        active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
            current_voxels, kernel_size, self.spatial_sparsity, clustering_factor=self.clustering_factor)
        
        effective_in_channels = int(current_channels * (1 - self.channel_sparsity))
        effective_out_channels = int(32 * (1 - self.channel_sparsity))
        
        # GEMM formulation for sparse conv: 
        # M = number of active input-output element pairs (active elements to compute)
        # N = output channels
        # K = kernel_volume * input_channels (gathered input features per output)
        M = active_pairs  # Number of active input-output element pairs
        N = 32  # Output channels 
        K = kernel_volume * current_channels  # kernel_volume * input_channels = 27 * 4 = 108
        gemm_generator.add_gemm_layer('conv0', M, N, K, active_pairs, kernel_size)
        
        current_voxels = output_voxels
        current_channels = 32
        
        # Encoder stages - sparse conv only (NO DOWNSAMPLING)
        encoder_configs = [
            (32, 64, 1),   # stage1: no downsample
            (64, 128, 1),  # stage2: no downsample
            (128, 256, 1), # stage3: no downsample
            (256, 512, 1), # stage4: no downsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(encoder_configs):
            kernel_size = 3
            kernel_volume = kernel_size ** 3
            
            # No downsampling
            
            active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
                current_voxels, kernel_size, self.spatial_sparsity, stride)
            
            M = active_pairs  # Number of active input-output element pairs
            N = out_ch  # Output channels
            K = kernel_volume * in_ch  # kernel_volume * input_channels
            gemm_generator.add_gemm_layer(f'encoder_stage_{i+1}', M, N, K, active_pairs, kernel_size)
            
            current_voxels = output_voxels
            current_channels = out_ch
        
        # Decoder stages - sparse conv only (NO UPSAMPLING)
        decoder_configs = [
            (512, 256, 1),  # stage1: no upsample
            (256, 128, 1),  # stage2: no upsample
            (128, 64, 1),   # stage3: no upsample
            (64, 32, 1),    # stage4: no upsample
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            # No upsampling
            
            active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
                current_voxels, kernel_size, self.spatial_sparsity)
            
            M = active_pairs  # Number of active input-output element pairs
            N = out_ch  # Output channels
            K = kernel_volume * in_ch  # kernel_volume * input_channels
            gemm_generator.add_gemm_layer(f'decoder_stage_{i+1}', M, N, K, active_pairs, kernel_size)
            
            current_voxels = output_voxels
            current_channels = out_ch
        
        # Final classification head
        kernel_size = 1
        kernel_volume = 1
        
        M = current_voxels  # Number of voxels (1x1x1 conv, so active_pairs = voxels)
        N = self.num_classes  # Output channels (20 classes)
        K = kernel_volume * current_channels  # 1 * input_channels
        gemm_generator.add_gemm_layer('classification_head', M, N, K, current_voxels, kernel_size)

class SPVNAS:
    """SPVNAS network configuration with advanced sparsity modeling"""
    
    def __init__(self, num_points: int = 100000, voxel_size: float = 0.05,
                 input_channels: int = 4, num_classes: int = 20,
                 spatial_sparsity: float = 0.05,
                 channel_sparsity: float = 0.4,
                 feature_sparsity: float = 0.6,
                 clustering_factor: float = 4.0):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        # spatial_sparsity: fraction of kernel positions that have non-zero data per output voxel
        self.spatial_sparsity = spatial_sparsity
        self.channel_sparsity = channel_sparsity  # SPVNAS uses more aggressive pruning
        self.feature_sparsity = feature_sparsity
        # clustering_factor: SPVNAS tends to work better with more clustered data
        self.clustering_factor = clustering_factor
        
        # Since downsampling is disabled, each point corresponds to one active voxel
        # Use same calculation as MinkowskiNet for fair comparison
        self.num_voxels = num_points
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for SPVNAS with advanced sparsity - sparse conv layers only"""
        total_macs = 0
        layer_details = {}
        current_voxels = self.num_voxels
        current_channels = self.input_channels
        
        # SPVNAS stem: 4 -> 32 -> 32 (sparse conv only)
        stem_macs1, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
            current_voxels, 3, current_channels, 32, 
            self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
        stem_macs2, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
            current_voxels, 3, 32, 32, 
            self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
        
        total_macs += stem_macs1 + stem_macs2
        current_channels = 32
        layer_details['stem'] = {
            'macs': stem_macs1 + stem_macs2, 'output_voxels': current_voxels, 'channels': current_channels
        }
        
        # SPVNAS encoder blocks - sparse conv only (NO DOWNSAMPLING)
        spvnas_blocks = [
            (32, 48, 1),     # More efficient channel progression, no downsample
            (48, 64, 1),     # no downsample
            (64, 128, 1),    # no downsample
            (128, 256, 1),   # Fewer channels overall than MinkowskiNet, no downsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(spvnas_blocks):
            # No downsampling
            
            # Efficient convolution - single conv per stage (not residual blocks)
            kernel_size = 3  # Always use 3x3x3
            macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
                current_voxels, kernel_size, in_ch, out_ch, 
                self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
            
            total_macs += macs
            current_channels = out_ch
            
            layer_details[f'spvnas_block_{i+1}'] = {
                'macs': macs, 'output_voxels': current_voxels, 'channels': current_channels
            }
        
        # SPVNAS decoder - sparse conv only (NO UPSAMPLING)
        decoder_configs = [
            (256, 128, 1),  # Simpler decoder, no upsample
            (128, 64, 1),   # no upsample
            (64, 32, 1),    # no upsample
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            # No upsampling
            
            # Simple convolution
            macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
                current_voxels, 3, in_ch, out_ch, 
                self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
            
            total_macs += macs
            current_channels = out_ch
            
            layer_details[f'decoder_stage_{i+1}'] = {
                'macs': macs, 'output_voxels': current_voxels, 'channels': current_channels
            }
        
        # Final classification head
        point_classification_macs, _ = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
            current_voxels, 1, current_channels, self.num_classes, 
            self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
        total_macs += point_classification_macs
        layer_details['point_classification'] = {
            'macs': point_classification_macs, 'output_voxels': current_voxels, 'classes': self.num_classes
        }
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'layer_details': layer_details,
            'final_output_points': self.num_points
        }
    
    def generate_gemm_data(self, gemm_generator: GEMMLayerGenerator, batch_size: int = 1):
        """Generate GEMM data for SPVNAS layers following Minuet sparse tensor GEMM formulation"""
        current_voxels = self.num_voxels
        current_channels = self.input_channels
        
        # SPVNAS stem: 4 -> 32 -> 32 (sparse conv only)
        kernel_size = 3
        kernel_volume = kernel_size ** 3
        active_pairs, stem_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
            current_voxels, kernel_size, self.spatial_sparsity, clustering_factor=self.clustering_factor)
        
        effective_in_channels = int(current_channels * (1 - self.channel_sparsity))
        effective_out_channels = int(32 * (1 - self.channel_sparsity))
        
        # First stem conv
        M = active_pairs  # Number of active input-output element pairs
        N = 32  # Output channels (before sparsity - sparsity affects computation, not GEMM size)
        K = kernel_volume * current_channels  # kernel_volume * input_channels = 27 * 4 = 108
        gemm_generator.add_gemm_layer('stem_conv1', M, N, K, active_pairs, kernel_size)
        
        # Second stem conv
        active_pairs, stem_voxels2 = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
            stem_voxels, kernel_size, self.spatial_sparsity, clustering_factor=self.clustering_factor)
        
        M = active_pairs  # Number of active input-output element pairs
        N = 32  # Output channels
        K = kernel_volume * 32  # kernel_volume * input_channels = 27 * 32 = 864
        gemm_generator.add_gemm_layer('stem_conv2', M, N, K, active_pairs, kernel_size)
        
        current_voxels = stem_voxels2
        current_channels = 32
        
        # SPVNAS encoder blocks - sparse conv only (NO DOWNSAMPLING)
        spvnas_blocks = [
            (32, 48, 1),     # More efficient channel progression, no downsample
            (48, 64, 1),     # no downsample
            (64, 128, 1),    # no downsample
            (128, 256, 1),   # Fewer channels overall than MinkowskiNet, no downsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(spvnas_blocks):
            kernel_size = 3  # Always use 3x3x3
            kernel_volume = kernel_size ** 3
            
            # No downsampling
            
            active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
                current_voxels, kernel_size, self.spatial_sparsity, stride, self.clustering_factor)
            
            M = active_pairs  # Number of active input-output element pairs
            N = out_ch  # Output channels
            K = kernel_volume * in_ch  # kernel_volume * input_channels
            gemm_generator.add_gemm_layer(f'spvnas_block_{i+1}', M, N, K, active_pairs, kernel_size)
            
            current_voxels = output_voxels
            current_channels = out_ch
        
        # SPVNAS decoder - sparse conv only (NO UPSAMPLING)
        decoder_configs = [
            (256, 128, 1),  # Simpler decoder, no upsample
            (128, 64, 1),   # no upsample
            (64, 32, 1),    # no upsample
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            # No upsampling
            
            active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
                current_voxels, kernel_size, self.spatial_sparsity, clustering_factor=self.clustering_factor)
            print(f"Active pairs for decoder stage {i+1}: {active_pairs}, output voxels: {output_voxels}")
            M = active_pairs  # Number of active input-output element pairs
            N = out_ch  # Output channels
            K = kernel_volume * in_ch  # kernel_volume * input_channels
            gemm_generator.add_gemm_layer(f'decoder_stage_{i+1}', M, N, K, active_pairs, kernel_size)
            
            current_voxels = output_voxels
            current_channels = out_ch
        
        # Final classification head
        kernel_size = 1
        kernel_volume = 1
        
        M = current_voxels  # Number of voxels (1x1x1 conv, so active_pairs = voxels)
        N = self.num_classes  # Output channels (20 classes)
        K = kernel_volume * current_channels  # 1 * input_channels
        gemm_generator.add_gemm_layer('point_classification', M, N, K, current_voxels, kernel_size)

class GEMMPerformanceModel:
    """Performance model for sparse GEMM operations showing arithmetic intensity effects"""
    
    def __init__(self):
        # Hardware parameters (representative accelerator - A100-like)
        self.peak_flops = 312e12  # Peak FLOPs/s (312 TFLOPS for A100 Tensor)
        self.memory_bandwidth = 1555e9  # Memory bandwidth (1555 GB/s for A100 HBM2)
        self.cache_size = 40e6  # L2 cache size (40 MB)
        
        # Sparse operation overheads
        self.index_overhead_bytes = 4  # bytes per index (int32)
        self.gather_scatter_cycles = 10  # additional cycles for gather/scatter
        self.metadata_processing_cycles = 5  # cycles for processing sparsity metadata
        
        # Memory access costs
        self.dram_access_cycles = 400  # cycles to access DRAM
        self.cache_access_cycles = 10   # cycles to access cache
        
    def calculate_arithmetic_intensity(self, M: int, N: int, K: int, 
                                     sparsity_factor: float = 1.0,
                                     spatial_sparsity: float = 0.05) -> Dict[str, float]:
        """
        Calculate arithmetic intensity for sparse GEMM
        
        Args:
            M, N, K: GEMM dimensions
            sparsity_factor: Overall sparsity (fraction of non-zero elements)
            spatial_sparsity: Spatial sparsity (affects reuse patterns)
        
        Returns:
            Dictionary with arithmetic intensity metrics
        """
        # Dense GEMM arithmetic intensity baseline
        dense_flops = 2 * M * N * K  # 2 operations per MAC (multiply + add)
        dense_bytes = (M * K + K * N + M * N) * 4  # 4 bytes per float32
        dense_ai = dense_flops / dense_bytes
        
        # Sparse GEMM with reduced computation but increased metadata overhead
        sparse_flops = dense_flops * sparsity_factor
        
        # Memory overhead increases due to:
        # 1. Index arrays for sparse representation
        # 2. Irregular access patterns reducing cache efficiency
        # 3. Metadata for tracking non-zero positions
        
        # Effective non-zero elements
        nnz_elements = M * K * sparsity_factor
        
        # Memory footprint includes:
        # - Actual data (reduced by sparsity)
        # - Index arrays (overhead)
        # - Reduced cache efficiency (more cache misses)
        sparse_data_bytes = nnz_elements * 4  # actual non-zero data
        index_bytes = nnz_elements * self.index_overhead_bytes  # indices
        metadata_bytes = M * 4  # row pointers or similar metadata
        
        # Cache efficiency degradation due to irregular access
        cache_efficiency = min(1.0, spatial_sparsity * 10)  # spatial locality affects caching
        effective_cache_misses = 1.0 / cache_efficiency
        
        sparse_bytes = (sparse_data_bytes + index_bytes + metadata_bytes) * effective_cache_misses
        
        # Output remains dense in most cases
        output_bytes = M * N * 4
        total_sparse_bytes = sparse_bytes + output_bytes
        
        sparse_ai = sparse_flops / total_sparse_bytes
        
        return {
            'dense_ai': dense_ai,
            'sparse_ai': sparse_ai,
            'ai_ratio': sparse_ai / dense_ai,
            'flops_reduction': sparse_flops / dense_flops,
            'memory_overhead': total_sparse_bytes / dense_bytes,
            'roofline_knee': self.memory_bandwidth / self.peak_flops,  # AI at which we hit memory wall
            'is_memory_bound': sparse_ai < (self.memory_bandwidth / self.peak_flops)
        }
    
    def estimate_performance(self, M: int, N: int, K: int,
                           sparsity_factor: float = 1.0,
                           spatial_sparsity: float = 0.05,
                           channel_sparsity: float = 0.3,
                           feature_sparsity: float = 0.5) -> Dict[str, float]:
        """
        Estimate actual performance including overheads
        
        Args:
            M, N, K: GEMM dimensions 
            sparsity_factor: Overall computational sparsity
            spatial_sparsity: Spatial locality (affects cache behavior)
            channel_sparsity: Input/output channel pruning (affects M×K and K×N matrices)
            feature_sparsity: Zero activations after ReLU (affects K×N matrix)
        
        Returns:
            Performance metrics including effective throughput
        """
        # Calculate effective matrix dimensions after sparsity
        # M×K matrix: affected by channel sparsity (input channels pruned)
        effective_K_input = int(K * (1 - channel_sparsity))
        
        # K×N matrix: affected by both channel sparsity (output channels) and feature sparsity (zero activations)
        effective_K_output = int(K * (1 - channel_sparsity))
        effective_N = int(N * (1 - feature_sparsity))
        
        # Recalculate arithmetic intensity with proper sparsity effects
        ai_metrics = self.calculate_arithmetic_intensity(M, N, K, sparsity_factor, spatial_sparsity)
        
        # Base computation: reduced by channel and feature sparsity
        # Effective FLOPs = M × effective_N × effective_K × 2 (MAC operations)
        effective_flops = 2 * M * effective_N * effective_K_input
        
        # Memory requirements for sparse matrices
        # M×K matrix (input): reduced by channel sparsity
        mk_matrix_bytes = M * effective_K_input * 4  # 4 bytes per float32
        mk_index_bytes = M * effective_K_input * self.index_overhead_bytes  # indices for sparse representation
        
        # K×N matrix (weights/features): reduced by channel and feature sparsity  
        kn_matrix_bytes = effective_K_output * effective_N * 4
        kn_index_bytes = effective_K_output * effective_N * self.index_overhead_bytes
        
        # Output M×N matrix: usually remains dense, but can be affected by feature sparsity
        output_bytes = M * effective_N * 4
        
        # Total memory with sparsity overheads
        total_sparse_bytes = (mk_matrix_bytes + mk_index_bytes + 
                             kn_matrix_bytes + kn_index_bytes + 
                             output_bytes)
        
        # Cache efficiency degradation due to irregular access patterns
        # More severe with higher sparsity levels
        cache_efficiency = min(1.0, spatial_sparsity * 10 + (1 - channel_sparsity) * 0.5)
        effective_cache_misses = 1.0 / cache_efficiency
        
        adjusted_memory_bytes = total_sparse_bytes * effective_cache_misses
        
        # Calculate memory bandwidth requirement
        memory_time = adjusted_memory_bytes / self.memory_bandwidth
        
        # Add fixed overheads that become more significant with sparsity
        # Gather/scatter overhead: increases with sparsity due to irregular access
        gather_scatter_overhead = (M * effective_N * channel_sparsity * 2) * self.gather_scatter_cycles / self.peak_flops
        
        # Metadata processing: increases with number of sparse elements
        sparse_elements = M * effective_K_input + effective_K_output * effective_N
        metadata_overhead = sparse_elements * self.metadata_processing_cycles / self.peak_flops
        
        # Determine if memory-bound or compute-bound
        arithmetic_intensity = effective_flops / adjusted_memory_bytes
        roofline_knee = self.memory_bandwidth / self.peak_flops
        is_memory_bound = arithmetic_intensity < roofline_knee
        
        if is_memory_bound:
            # Memory-bound: performance limited by memory bandwidth
            total_time = memory_time + gather_scatter_overhead + metadata_overhead
            effective_flops_per_sec = effective_flops / total_time
            
        else:
            # Compute-bound: performance limited by peak FLOPs, but add sparse overhead
            compute_time = effective_flops / self.peak_flops
            
            # Sparse computation overhead increases with sparsity
            sparse_overhead_factor = 1.0 + 0.5 * (channel_sparsity + feature_sparsity)
            total_time = compute_time * sparse_overhead_factor + gather_scatter_overhead + metadata_overhead
            effective_flops_per_sec = effective_flops / total_time
        
        # Calculate efficiency metrics
        peak_utilization = effective_flops_per_sec / self.peak_flops
        memory_utilization = (adjusted_memory_bytes / total_time) / self.memory_bandwidth
        
        return {
            'effective_flops_per_sec': effective_flops_per_sec,
            'execution_time_ms': total_time * 1000,
            'peak_utilization': peak_utilization,
            'memory_utilization': memory_utilization,
            'bottleneck': 'memory' if is_memory_bound else 'compute',
            'arithmetic_intensity': arithmetic_intensity,
            'effective_flops': effective_flops,
            'effective_K_input': effective_K_input,
            'effective_N': effective_N
        }
    
    def analyze_sparsity_impact(self, gemm_layers: List[Dict]) -> Dict:
        """
        Analyze performance impact across different sparsity levels
        """
        sparsity_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        channel_sparsity_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
        feature_sparsity_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
        
        analysis_results = {
            'sparsity_sweep': [],
            'channel_sparsity_sweep': [],
            'feature_sparsity_sweep': [],
            'layer_analysis': []
        }
        
        # Analyze impact of overall sparsity
        if gemm_layers:
            sample_layer = gemm_layers[0]
            M, N, K = sample_layer['M'], sample_layer['N'], sample_layer['K']
            
            for sparsity in sparsity_levels:
                perf = self.estimate_performance(M, N, K, sparsity_factor=sparsity)
                ai_metrics = self.calculate_arithmetic_intensity(M, N, K, sparsity_factor=sparsity)
                
                analysis_results['sparsity_sweep'].append({
                    'sparsity_factor': sparsity,
                    'effective_flops_per_sec': perf['effective_flops_per_sec'],
                    'arithmetic_intensity': ai_metrics['sparse_ai'],
                    'is_memory_bound': ai_metrics['is_memory_bound'],
                    'peak_utilization': perf['peak_utilization']
                })
            
            # Analyze impact of channel sparsity (at 20% overall sparsity)
            for channel_sparsity in channel_sparsity_levels:
                perf = self.estimate_performance(M, N, K, sparsity_factor=0.2, 
                                               channel_sparsity=channel_sparsity, feature_sparsity=0.3)
                ai_metrics = self.calculate_arithmetic_intensity(M, N, K, sparsity_factor=0.2)
                
                analysis_results['channel_sparsity_sweep'].append({
                    'channel_sparsity': channel_sparsity,
                    'effective_flops_per_sec': perf['effective_flops_per_sec'],
                    'arithmetic_intensity': perf['arithmetic_intensity'],
                    'is_memory_bound': perf['bottleneck'] == 'memory',
                    'peak_utilization': perf['peak_utilization']
                })
            
            # Analyze impact of feature sparsity (at 20% overall sparsity, 30% channel sparsity)
            for feature_sparsity in feature_sparsity_levels:
                perf = self.estimate_performance(M, N, K, sparsity_factor=0.2,
                                               channel_sparsity=0.3, feature_sparsity=feature_sparsity)
                
                analysis_results['feature_sparsity_sweep'].append({
                    'feature_sparsity': feature_sparsity,
                    'effective_flops_per_sec': perf['effective_flops_per_sec'],
                    'arithmetic_intensity': perf['arithmetic_intensity'],
                    'is_memory_bound': perf['bottleneck'] == 'memory',
                    'peak_utilization': perf['peak_utilization']
                })
        
        # Analyze each layer with realistic sparsity values
        for layer in gemm_layers[:5]:  # Analyze first 5 layers
            M, N, K = layer['M'], layer['N'], layer['K']
            
            # Use realistic sparsity based on layer characteristics
            if 'Active_Pairs' in layer:
                estimated_sparsity = min(1.0, layer['Active_Pairs'] / (M * K))
            else:
                estimated_sparsity = 0.2  # Default assumption
            
            # Estimate channel and feature sparsity based on layer type
            # Early layers typically have less channel sparsity, later layers more
            layer_depth = len(analysis_results['layer_analysis'])
            estimated_channel_sparsity = min(0.6, layer_depth * 0.1)
            estimated_feature_sparsity = 0.4  # Typical ReLU sparsity
            
            perf = self.estimate_performance(M, N, K, sparsity_factor=estimated_sparsity,
                                           channel_sparsity=estimated_channel_sparsity,
                                           feature_sparsity=estimated_feature_sparsity)
            
            analysis_results['layer_analysis'].append({
                'layer_name': layer['Layer'],
                'M': M, 'N': N, 'K': K,
                'estimated_sparsity': estimated_sparsity,
                'channel_sparsity': estimated_channel_sparsity,
                'feature_sparsity': estimated_feature_sparsity,
                'arithmetic_intensity': perf['arithmetic_intensity'],
                'effective_flops_per_sec': perf['effective_flops_per_sec'],
                'is_memory_bound': perf['bottleneck'] == 'memory',
                'bottleneck': perf['bottleneck'],
                'peak_utilization': perf['peak_utilization'],
                'effective_flops': perf['effective_flops']
            })
        
        return analysis_results

def plot_performance_analysis(performance_model: GEMMPerformanceModel, gemm_data: List[Dict]):
    """Create comprehensive performance analysis plots"""
    
    # Get analysis results
    analysis = performance_model.analyze_sparsity_impact(gemm_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sparse GEMM Performance Analysis: Channel & Feature Sparsity Impact', fontsize=16, fontweight='bold')
    
    # Plot 1: Channel Sparsity vs Performance
    ax1 = axes[0, 0]
    channel_data = analysis['channel_sparsity_sweep']
    
    if channel_data:
        channel_sparsities = [d['channel_sparsity'] for d in channel_data]
        channel_flops = [d['effective_flops_per_sec'] / 1e12 for d in channel_data]
        channel_utils = [d['peak_utilization'] * 100 for d in channel_data]
        
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(channel_sparsities, channel_flops, 'bo-', linewidth=2, label='Performance (TFLOPS)')
        line2 = ax1_twin.plot(channel_sparsities, channel_utils, 'rs-', linewidth=2, label='Peak Utilization (%)')
        
        ax1.set_xlabel('Channel Sparsity Ratio')
        ax1.set_ylabel('Performance (TFLOPS)', color='blue')
        ax1_twin.set_ylabel('Peak Utilization (%)', color='red')
        ax1.set_title('Channel Sparsity Impact on M×K and K×N Matrices')
        ax1.grid(True, alpha=0.3)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 2: Feature Sparsity vs Performance  
    ax2 = axes[0, 1]
    feature_data = analysis['feature_sparsity_sweep']
    
    if feature_data:
        feature_sparsities = [d['feature_sparsity'] for d in feature_data]
        feature_flops = [d['effective_flops_per_sec'] / 1e12 for d in feature_data]
        feature_utils = [d['peak_utilization'] * 100 for d in feature_data]
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(feature_sparsities, feature_flops, 'go-', linewidth=2, label='Performance (TFLOPS)')
        line2 = ax2_twin.plot(feature_sparsities, feature_utils, 'ms-', linewidth=2, label='Peak Utilization (%)')
        
        ax2.set_xlabel('Feature Sparsity Ratio')
        ax2.set_ylabel('Performance (TFLOPS)', color='green')
        ax2_twin.set_ylabel('Peak Utilization (%)', color='magenta')
        ax2.set_title('Feature Sparsity Impact on K×N Matrix')
        ax2.grid(True, alpha=0.3)
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 3: Layer-wise Performance Analysis
    ax3 = axes[1, 0]
    layer_data = analysis['layer_analysis']
    
    if layer_data:
        layer_names = [d['layer_name'] for d in layer_data]
        layer_utils = [d['peak_utilization'] * 100 for d in layer_data]
        layer_bottleneck = [d['is_memory_bound'] for d in layer_data]
        
        # Bar plot with color coding for bottleneck type
        colors = ['red' if mb else 'blue' for mb in layer_bottleneck]
        bars = ax3.bar(range(len(layer_names)), layer_utils, color=colors, alpha=0.7)
        
        # Add sparsity information as text on bars
        for i, (bar, layer) in enumerate(zip(bars, layer_data)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"C:{layer['channel_sparsity']:.1f}\nF:{layer['feature_sparsity']:.1f}",
                    ha='center', va='bottom', fontsize=8)
        
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Peak Utilization (%)')
        ax3.set_title('Per-Layer Performance with Sparsity Breakdown')
        ax3.set_xticks(range(len(layer_names)))
        ax3.set_xticklabels(layer_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Memory-bound'),
                          Patch(facecolor='blue', alpha=0.7, label='Compute-bound')]
        ax3.legend(handles=legend_elements)
    
    # Plot 4: Arithmetic Intensity vs Sparsity
    ax4 = axes[1, 1]
    
    if layer_data:
        # Create scatter plot showing relationship between sparsity and arithmetic intensity
        total_sparsities = []
        ais = []
        peak_utils = []
        
        for layer in layer_data:
            total_sparsity = layer['channel_sparsity'] + layer['feature_sparsity']
            total_sparsities.append(total_sparsity)
            ais.append(layer['arithmetic_intensity'])
            peak_utils.append(layer['peak_utilization'] * 100)
        
        # Color by peak utilization
        scatter = ax4.scatter(total_sparsities, ais, c=peak_utils, s=100, alpha=0.7, cmap='RdYlBu_r')
        
        # Add roofline knee
        roofline_knee = performance_model.memory_bandwidth / performance_model.peak_flops
        ax4.axhline(y=roofline_knee, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Memory Wall (AI={roofline_knee:.3f})')
        
        ax4.set_xlabel('Total Sparsity (Channel + Feature)')
        ax4.set_ylabel('Arithmetic Intensity (FLOPs/byte)')
        ax4.set_title('Sparsity vs Arithmetic Intensity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Peak Utilization (%)')
    
    plt.tight_layout()
    return fig, analysis

def generate_performance_analysis():
    """Generate and save performance analysis for both networks"""
    print("\nGenerating GEMM Performance Analysis...")
    
    # Load GEMM data for both networks
    networks = ['MinkowskiNet', 'SPVNAS']
    performance_model = GEMMPerformanceModel()
    
    for network_name in networks:
        try:
            # Load GEMM data
            filename = f"{network_name.lower()}_gemm_layers.csv"
            filepath = os.path.join('/Users/ashriram/Desktop/minuet-tracer/mac-vs-time', filename)
            
            gemm_data = []
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert string values to integers
                    gemm_data.append({
                        'Layer': row['Layer'],
                        'M': int(row['M']),
                        'N': int(row['N']),
                        'K': int(row['K']),
                        'Active_Pairs': int(row.get('Active_Pairs', row['M'])),
                        'Kernel_Size': int(row.get('Kernel_Size', 3))
                    })
            
            # Generate performance analysis
            fig, analysis = plot_performance_analysis(performance_model, gemm_data)
            
            # Save plot
            plot_filename = f"{network_name.lower()}_performance_analysis.png"
            plt.savefig(os.path.join('/Users/ashriram/Desktop/minuet-tracer/mac-vs-time', plot_filename), 
                       dpi=300, bbox_inches='tight')
            print(f"Performance analysis saved as {plot_filename}")
            
            # Print summary
            print(f"\n{network_name} Performance Summary:")
            print("-" * 50)
            layer_analysis = analysis['layer_analysis']
            if layer_analysis:
                memory_bound_layers = sum(1 for layer in layer_analysis if layer['is_memory_bound'])
                avg_utilization = np.mean([layer['peak_utilization'] for layer in layer_analysis])
                print(f"Memory-bound layers: {memory_bound_layers}/{len(layer_analysis)}")
                print(f"Average peak utilization: {avg_utilization:.1%}")
                
                # Show most problematic layers
                sorted_layers = sorted(layer_analysis, key=lambda x: x['peak_utilization'])
                print("Most impacted layers:")
                for layer in sorted_layers[:3]:
                    print(f"  {layer['layer_name']}: {layer['peak_utilization']:.1%} utilization, "
                          f"AI={layer['arithmetic_intensity']:.3f} ({'memory' if layer['is_memory_bound'] else 'compute'}-bound)")
            
            plt.close(fig)  # Close to save memory
            
        except FileNotFoundError:
            print(f"Warning: Could not find GEMM data file for {network_name}")
        except Exception as e:
            print(f"Error analyzing {network_name}: {e}")
    
    print("\nPerformance analysis complete!")

def generate_gemm_csv_files():
    """Generate GEMM CSV files for each network"""
    print("\nGenerating GEMM CSV files for sparse 3D networks...")
    
    # Standard configurations
    configs = {
        'MinkowskiNet': {
            'num_points': 100000, 'voxel_size': 0.05, 'input_channels': 4, 
            'num_classes': 20, 'spatial_sparsity': 0.05,
            'channel_sparsity': 0.3, 'feature_sparsity': 0.5
        },
        'SPVNAS': {
            'num_points': 100000, 'voxel_size': 0.05, 'input_channels': 4, 
            'num_classes': 20, 'spatial_sparsity': 0.05,
            'channel_sparsity': 0.4, 'feature_sparsity': 0.6
        }
    }
    
    # Create GEMM generator
    gemm_generator = GEMMLayerGenerator()
    
    for network_name, config in configs.items():
        print(f"Generating GEMM data for {network_name}...")
        
        # Create network instance
        if network_name == 'MinkowskiNet':
            network = MinkowskiNet(**config)
        elif network_name == 'SPVNAS':
            network = SPVNAS(**config)
        
        # Clear previous data and generate GEMM data
        gemm_generator.clear()
        network.generate_gemm_data(gemm_generator, batch_size=1)
        
        # Save CSV file
        filename = f"{network_name.lower()}_gemm_layers.csv"
        gemm_generator.save_csv(filename)
        print(f"Saved {filename}")
    
    print("GEMM CSV generation complete!")

def plot_sparsity_analysis():
    """Create comprehensive sparsity analysis plots"""
    
    # Configuration for analysis
    base_config = {
        'num_points': 100000, 'voxel_size': 0.05, 'input_channels': 4, 
        'num_classes': 20, 'spatial_sparsity': 0.05
    }
    
    # 1. Individual sparsity impact
    sparsity_types = ['channel_sparsity', 'feature_sparsity']
    sparsity_ranges = {
        'channel_sparsity': np.linspace(0.0, 0.7, 15),
        'feature_sparsity': np.linspace(0.0, 0.8, 17)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('3D Sparse Networks: Advanced Sparsity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Channel Sparsity Impact
    ax1 = axes[0, 0]
    for network_name in ['MinkowskiNet', 'SPVNAS']:
        macs_values = []
        for cs in sparsity_ranges['channel_sparsity']:
            config = base_config.copy()
            config.update({'channel_sparsity': cs, 'feature_sparsity': 0.3})
            
            if network_name == 'MinkowskiNet':
                network = MinkowskiNet(**config)
            else:
                network = SPVNAS(**config)
            
            result = network.calculate_macs()
            macs_values.append(result['total_macs_millions'])
        
        ax1.plot(sparsity_ranges['channel_sparsity'], macs_values, 
                marker='o', linewidth=2, label=network_name)
    
    ax1.set_xlabel('Channel Sparsity Ratio')
    ax1.set_ylabel('MACs (Millions)')
    ax1.set_title('Impact of Channel Sparsity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature Sparsity Impact
    ax2 = axes[0, 1]
    for network_name in ['MinkowskiNet', 'SPVNAS']:
        macs_values = []
        for fs in sparsity_ranges['feature_sparsity']:
            config = base_config.copy()
            config.update({'channel_sparsity': 0.3, 'feature_sparsity': fs})
            
            if network_name == 'MinkowskiNet':
                network = MinkowskiNet(**config)
            else:
                network = SPVNAS(**config)
            
            result = network.calculate_macs()
            macs_values.append(result['total_macs_millions'])
        
        ax2.plot(sparsity_ranges['feature_sparsity'], macs_values, 
                marker='s', linewidth=2, label=network_name)
    
    ax2.set_xlabel('Feature Sparsity Ratio')
    ax2.set_ylabel('MACs (Millions)')
    ax2.set_title('Impact of Feature Sparsity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined Sparsity Heatmap for MinkowskiNet
    ax3 = axes[1, 0]
    channel_vals = np.linspace(0.0, 0.6, 10)
    feature_vals = np.linspace(0.0, 0.7, 10)
    macs_grid = np.zeros((len(feature_vals), len(channel_vals)))
    
    for i, fs in enumerate(feature_vals):
        for j, cs in enumerate(channel_vals):
            config = base_config.copy()
            config.update({'channel_sparsity': cs, 'feature_sparsity': fs})
            network = MinkowskiNet(**config)
            result = network.calculate_macs()
            macs_grid[i, j] = result['total_macs_millions']
    
    im3 = ax3.imshow(macs_grid, aspect='auto', origin='lower', cmap='viridis')
    ax3.set_xlabel('Channel Sparsity')
    ax3.set_ylabel('Feature Sparsity')
    ax3.set_title('MinkowskiNet: Combined Sparsity Impact')
    ax3.set_xticks(np.arange(0, len(channel_vals), 2))
    ax3.set_xticklabels([f'{x:.1f}' for x in channel_vals[::2]])
    ax3.set_yticks(np.arange(0, len(feature_vals), 2))
    ax3.set_yticklabels([f'{x:.1f}' for x in feature_vals[::2]])
    plt.colorbar(im3, ax=ax3, label='MACs (Millions)')
    
    # Plot 4: Efficiency Comparison
    ax4 = axes[1, 1]
    
    # Different sparsity scenarios
    scenarios = {
        'Dense (No Sparsity)': {'channel_sparsity': 0.0, 'feature_sparsity': 0.0},
        'Channel Pruned': {'channel_sparsity': 0.4, 'feature_sparsity': 0.0},
        'Feature Sparse': {'channel_sparsity': 0.0, 'feature_sparsity': 0.5},
        'Fully Optimized': {'channel_sparsity': 0.4, 'feature_sparsity': 0.6},
    }
    
    networks = ['MinkowskiNet', 'SPVNAS']
    scenario_names = list(scenarios.keys())
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    for i, network_name in enumerate(networks):
        macs_values = []
        for scenario_config in scenarios.values():
            config = base_config.copy()
            config.update(scenario_config)
            
            if network_name == 'MinkowskiNet':
                network = MinkowskiNet(**config)
            else:
                network = SPVNAS(**config)
            
            result = network.calculate_macs()
            macs_values.append(result['total_macs_millions'])
        
        ax4.bar(x + i * width, macs_values, width, label=network_name, alpha=0.8)
    
    ax4.set_xlabel('Sparsity Scenarios')
    ax4.set_ylabel('MACs (Millions)')
    ax4.set_title('Network Efficiency Comparison')
    ax4.set_xticks(x + width / 2)
    ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to calculate and compare MAC operations with advanced sparsity"""
    print("3D Sparse Convolution Networks - Advanced Sparsity MAC Calculator")
    print("=" * 80)
    
    # Generate GEMM CSV files for compute layers
    generate_gemm_csv_files()
    print("\n" + "=" * 80)
    
    # Generate performance analysis
    generate_performance_analysis()
    print("\n" + "=" * 80)
    
    # Standard configurations for LiDAR point clouds
    configs = {
        'MinkowskiNet (Dense)': {
            'num_points': 100000, 'voxel_size': 0.05, 'input_channels': 4, 
            'num_classes': 20, 'spatial_sparsity': 0.05,
            'channel_sparsity': 0.0, 'feature_sparsity': 0.0
        },
        'MinkowskiNet (Optimized)': {
            'num_points': 100000, 'voxel_size': 0.05, 'input_channels': 4, 
            'num_classes': 20, 'spatial_sparsity': 0.05,
            'channel_sparsity': 0.3, 'feature_sparsity': 0.5
        },
        'SPVNAS (Dense)': {
            'num_points': 100000, 'voxel_size': 0.05, 'input_channels': 4, 
            'num_classes': 20, 'spatial_sparsity': 0.05,
            'channel_sparsity': 0.0, 'feature_sparsity': 0.0
        },
        'SPVNAS (Optimized)': {
            'num_points': 100000, 'voxel_size': 0.05, 'input_channels': 4, 
            'num_classes': 20, 'spatial_sparsity': 0.05,
            'channel_sparsity': 0.4, 'feature_sparsity': 0.6
        }
    }
    
    results = {}
    
    # Calculate for each network configuration
    for network_name, config in configs.items():
        print(f"\n{network_name}")
        print("-" * 50)
        print(f"Input points: {config['num_points']:,}")
        print(f"Spatial sparsity: {config['spatial_sparsity']}")
        print(f"Channel sparsity: {config['channel_sparsity']}")
        print(f"Feature sparsity: {config['feature_sparsity']}")
        
        if 'MinkowskiNet' in network_name:
            network = MinkowskiNet(**config)
        else:
            network = SPVNAS(**config)
        
        result = network.calculate_macs()
        results[network_name] = result
        
        print(f"Total MACs: {result['total_macs']:,}")
        print(f"Total MACs (millions): {result['total_macs_millions']:.2f}M")
        
        if 'final_output_voxels' in result:
            print(f"Final output voxels: {result['final_output_voxels']:,}")
        if 'final_output_points' in result:
            print(f"Final output points: {result['final_output_points']:,}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    sorted_networks = sorted(results.items(), key=lambda x: x[1]['total_macs'])
    
    for network_name, result in sorted_networks:
        print(f"{network_name:<25}: {result['total_macs_millions']:>8.2f}M MACs")
    
    # Efficiency improvements
    print("\nSparsity Benefits:")
    dense_mink = results['MinkowskiNet (Dense)']['total_macs']
    opt_mink = results['MinkowskiNet (Optimized)']['total_macs']
    dense_spv = results['SPVNAS (Dense)']['total_macs']
    opt_spv = results['SPVNAS (Optimized)']['total_macs']
    
    mink_reduction = (dense_mink - opt_mink) / dense_mink * 100
    spv_reduction = (dense_spv - opt_spv) / dense_spv * 100
    
    print(f"MinkowskiNet sparsity reduction: {mink_reduction:.1f}%")
    print(f"SPVNAS sparsity reduction: {spv_reduction:.1f}%")
    
    # Generate plots
    print("\nGenerating sparsity analysis plots...")
    fig = plot_sparsity_analysis()
    
    # Save plot
    plt.savefig('/Users/ashriram/Desktop/minuet-tracer/mac-vs-time/sparsity_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("Plots saved as 'sparsity_analysis.png'")
    
    plt.show()

if __name__ == "__main__":
    main()
    plt.show()

if __name__ == "__main__":
    main()
