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
            spatial_sparsity: Ratio of occupied voxels (0.05 = 5% of voxels are non-zero)
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
        
        # For sparse convolution, computation is only performed on active voxels
        # Output voxels depend on how kernel overlaps with active input voxels
        # Each active input voxel can contribute to multiple output positions
        
        # Estimate output active voxels: depends on kernel size and local density
        # For stride=1: output typically has similar or slightly more active voxels
        # The kernel_size factor accounts for the expansion of active regions
        output_sparsity_factor = min(1.0, spatial_sparsity * kernel_size)
        output_voxels = int(num_active_voxels * output_sparsity_factor)
        
        # Active computation pairs: actual number of multiply-accumulate operations
        # This is the number of active input voxels × kernel positions × channel pairs
        active_kernel_positions = min(kernel_volume, int(kernel_volume * spatial_sparsity * 3))
        base_macs = num_active_voxels * active_kernel_positions * effective_in_channels * effective_out_channels
        
        # Apply feature sparsity (activations that are zero don't contribute)
        feature_effective_macs = int(base_macs * (1 - feature_sparsity))
        
        # Apply structured sparsity (additional pruning patterns)
        final_macs = int(feature_effective_macs * (1 - structured_sparsity))
        
        return final_macs, output_voxels
    
    @staticmethod
    def calculate_active_pairs(input_voxels: int, kernel_size: int, 
                              spatial_sparsity: float, stride: int = 1,
                              clustering_factor: float = 3.0) -> Tuple[int, int]:
        """
        Calculate the number of active input-output voxel pairs based on kernel overlap
        
        Args:
            input_voxels: Number of active input voxels (already sparse)
            kernel_size: Size of the convolution kernel
            spatial_sparsity: Fraction of voxels that are non-zero (e.g., 0.05 = 5% occupied)
            stride: Convolution stride
            clustering_factor: Factor accounting for local clustering of points (1.0-10.0)
                              Higher values = more clustered points = more kernel matches per output
            
        Returns:
            Tuple of (active_pairs, output_voxels)
        """
        kernel_volume = kernel_size ** 3
        
        # For sparse convolution:
        # - We only compute on positions where input voxels are active
        # - Each active input voxel affects a local neighborhood in the output
        # - The actual computation depends on kernel overlap patterns
        
        if stride > 1:
            # Downsampling: output has fewer voxels, roughly divided by stride^3
            # But we maintain some active voxels based on local density
            base_output_voxels = max(1, input_voxels // (stride ** 3))
            # Account for the fact that some regions might still be active after downsampling
            output_voxels = int(base_output_voxels * (1 + spatial_sparsity))
        else:
            # Same resolution: output active voxels depend on kernel expansion
            # Each input active voxel can create output activations in kernel neighborhood
            expansion_factor = min(2.0, 1.0 + (kernel_size - 1) * spatial_sparsity)
            output_voxels = int(input_voxels * expansion_factor)
        
        # Active pairs: number of actual multiply-accumulate operations
        # Each output voxel is computed from multiple input voxels within kernel range
        # But in sparse tensors, many kernel positions have zero input
        
        # Average number of active kernel positions per output voxel
        # This depends on:
        # 1. spatial_sparsity: how dense the input is
        # 2. clustering_factor: how clustered the points are (LiDAR points tend to cluster)
        # 3. kernel_volume: maximum possible matches
        
        theoretical_matches = kernel_volume * spatial_sparsity * clustering_factor
        avg_active_kernel_positions = min(kernel_volume, int(theoretical_matches))
        
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
        # spatial_sparsity: fraction of voxels that are occupied (non-zero)
        # For LiDAR: typically 3-7% of the 3D grid contains actual points
        self.spatial_sparsity = spatial_sparsity
        self.channel_sparsity = channel_sparsity
        self.feature_sparsity = feature_sparsity
        # clustering_factor: how clustered the LiDAR points are (affects kernel matches)
        # 1.0 = uniform distribution, 3.0 = moderate clustering, 5.0+ = high clustering
        self.clustering_factor = clustering_factor
        
        # Estimate number of active voxels from point cloud
        # This represents the actual occupied voxels in the sparse tensor
        # For a fair comparison, both networks should start with similar voxel counts
        estimated_total_voxels = int(num_points / (spatial_sparsity * 10))  # Conservative estimation
        self.num_voxels = int(estimated_total_voxels * spatial_sparsity)
        
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
        
        # Encoder stages - sparse conv only
        encoder_configs = [
            (32, 64, 2),   # stage1: downsample
            (64, 128, 2),  # stage2: downsample
            (128, 256, 2), # stage3: downsample
            (256, 512, 2), # stage4: downsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(encoder_configs):
            # Downsampling convolution
            if stride > 1:
                current_voxels = current_voxels // (stride ** 3)
            
            # Main convolution
            macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
                current_voxels, 3, in_ch, out_ch, 
                self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
            total_macs += macs
            current_channels = out_ch
            
            layer_details[f'encoder_stage_{i+1}'] = {
                'macs': macs, 'output_voxels': current_voxels, 'channels': current_channels
            }
        
        # Decoder stages - sparse conv only
        decoder_configs = [
            (512, 256, 2),  # stage1: upsample
            (256, 128, 2),  # stage2: upsample
            (128, 64, 2),   # stage3: upsample
            (64, 32, 2),    # stage4: upsample
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            # Upsampling increases voxel count
            current_voxels = current_voxels * (upsample_factor ** 3)
            
            # Transpose convolution for upsampling
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
        
        # Encoder stages - sparse conv only
        encoder_configs = [
            (32, 64, 2),   # stage1: downsample
            (64, 128, 2),  # stage2: downsample
            (128, 256, 2), # stage3: downsample
            (256, 512, 2), # stage4: downsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(encoder_configs):
            kernel_size = 3
            kernel_volume = kernel_size ** 3
            
            # Downsampling convolution
            if stride > 1:
                current_voxels = current_voxels // (stride ** 3)
            
            active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
                current_voxels, kernel_size, self.spatial_sparsity, stride)
            
            M = active_pairs  # Number of active input-output element pairs
            N = out_ch  # Output channels
            K = kernel_volume * in_ch  # kernel_volume * input_channels
            gemm_generator.add_gemm_layer(f'encoder_stage_{i+1}', M, N, K, active_pairs, kernel_size)
            
            current_voxels = output_voxels
            current_channels = out_ch
        
        # Decoder stages - sparse conv only
        decoder_configs = [
            (512, 256, 2),  # stage1: upsample
            (256, 128, 2),  # stage2: upsample
            (128, 64, 2),   # stage3: upsample
            (64, 32, 2),    # stage4: upsample
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            kernel_size = 3
            kernel_volume = kernel_size ** 3
            
            # Upsampling increases voxel count
            current_voxels = current_voxels * (upsample_factor ** 3)
            
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
        # spatial_sparsity: fraction of voxels that contain actual data
        # SPVNAS uses more aggressive sparsity exploitation
        self.spatial_sparsity = spatial_sparsity
        self.channel_sparsity = channel_sparsity  # SPVNAS uses more aggressive pruning
        self.feature_sparsity = feature_sparsity
        # clustering_factor: SPVNAS tends to work better with more clustered data
        self.clustering_factor = clustering_factor
        
        # Estimate active voxels: Use same estimation as MinkowskiNet for fair comparison
        estimated_total_voxels = int(num_points / (spatial_sparsity * 10))  # Same as MinkowskiNet
        self.num_voxels = int(estimated_total_voxels * spatial_sparsity)
        
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
        
        # SPVNAS encoder blocks - sparse conv only (simpler than MinkowskiNet)
        spvnas_blocks = [
            (32, 48, 2),     # More efficient channel progression
            (48, 64, 2),     
            (64, 128, 2),    
            (128, 256, 2),   # Fewer channels overall than MinkowskiNet
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(spvnas_blocks):
            if stride > 1:
                current_voxels = current_voxels // (stride ** 3)
            
            # Efficient convolution - single conv per stage (not residual blocks)
            kernel_size = 2 if stride > 1 else 3
            macs, current_voxels = AdvancedSparseConv3DMACCalculator.sparse_conv3d_macs(
                current_voxels, kernel_size, in_ch, out_ch, 
                self.spatial_sparsity, self.channel_sparsity, self.feature_sparsity)
            
            total_macs += macs
            current_channels = out_ch
            
            layer_details[f'spvnas_block_{i+1}'] = {
                'macs': macs, 'output_voxels': current_voxels, 'channels': current_channels
            }
        
        # SPVNAS decoder - sparse conv only (simpler than MinkowskiNet)
        decoder_configs = [
            (256, 128, 2),  # Simpler decoder
            (128, 64, 2),
            (64, 32, 2),
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            current_voxels = current_voxels * (upsample_factor ** 3)
            
            # Simple deconvolution
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
        
        # SPVNAS encoder blocks - sparse conv only (simpler than MinkowskiNet)
        spvnas_blocks = [
            (32, 48, 2),     # More efficient channel progression
            (48, 64, 2),     
            (64, 128, 2),    
            (128, 256, 2),   # Fewer channels overall than MinkowskiNet
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(spvnas_blocks):
            kernel_size = 2 if stride > 1 else 3
            kernel_volume = kernel_size ** 3
            
            if stride > 1:
                current_voxels = current_voxels // (stride ** 3)
            
            active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
                current_voxels, kernel_size, self.spatial_sparsity, stride, self.clustering_factor)
            
            M = active_pairs  # Number of active input-output element pairs
            N = out_ch  # Output channels
            K = kernel_volume * in_ch  # kernel_volume * input_channels
            gemm_generator.add_gemm_layer(f'spvnas_block_{i+1}', M, N, K, active_pairs, kernel_size)
            
            current_voxels = output_voxels
            current_channels = out_ch
        
        # SPVNAS decoder - sparse conv only (simpler than MinkowskiNet)
        decoder_configs = [
            (256, 128, 2),  # Simpler decoder
            (128, 64, 2),
            (64, 32, 2),
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            kernel_size = 3
            kernel_volume = kernel_size ** 3
            
            current_voxels = current_voxels * (upsample_factor ** 3)
            
            active_pairs, output_voxels = AdvancedSparseConv3DMACCalculator.calculate_active_pairs(
                current_voxels, kernel_size, self.spatial_sparsity, clustering_factor=self.clustering_factor)
            
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
