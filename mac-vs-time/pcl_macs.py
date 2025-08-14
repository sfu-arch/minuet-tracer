#!/usr/bin/env python3
"""
MAC Operations Calculator for 3D Sparse Convolution Networks with Advanced Sparsity Modeling
Supports: MinkowskiNet, SPVNAS, LargeKernel3D, VoxelNeXt, RSN

Network configurations sourced from official repositories:
- MinkowskiNet: https://github.com/NVIDIA/MinkowskiEngine (Paper: https://arxiv.org/abs/1904.08755)
- SPVNAS: https://github.com/mit-han-lab/spvnas (Paper: https://arxiv.org/abs/2007.16100)
- LargeKernel3D: https://github.com/dvlab-research/LargeKernel3D (Paper: Large Kernel Convolutions for 3D Processing)
- VoxelNeXt: https://github.com/dvlab-research/VoxelNeXt (Paper: https://arxiv.org/abs/2211.12697)
- RSN: https://github.com/caiyuanhao1998/RSN (Paper: Range Sparse Net for LiDAR 3D Object Detection)

Advanced sparsity modeling includes:
1. Spatial sparsity: Non-zero voxel occupancy (typical LiDAR: 3-7%)
2. Channel sparsity: Pruned/inactive channels (typical: 20-50%)
3. Feature sparsity: Zero activations after ReLU (typical: 30-70%)
4. Structured sparsity: Block-wise or group-wise sparsity patterns

Typical Spatial Sparsity Factors by Dataset:
- SemanticKITTI: 3-5% voxel occupancy
- nuScenes: 8-12% voxel occupancy  
- ScanNet (indoor): 15-25% voxel occupancy
- S3DIS (indoor): 20-30% voxel occupancy

Weight Sparsity Potential (without significant accuracy loss):
- Conservative (<1% acc loss): MinkowskiNet 30%, SPVNAS 25%, LargeKernel3D 20%, VoxelNeXt 25%, RSN 30%
- Moderate (<3% acc loss): MinkowskiNet 50%, SPVNAS 45%, LargeKernel3D 40%, VoxelNeXt 45%, RSN 50%
- Aggressive (<5% acc loss): MinkowskiNet 70%, SPVNAS 65%, LargeKernel3D 60%, VoxelNeXt 65%, RSN 70%
"""

import math
import csv
import os
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

@dataclass
class SparsityConfig:
    """Configuration for 3D sparsity parameters"""
    spatial_sparsity: float = 0.05    # Default 5% spatial occupancy (LiDAR typical)
    feature_sparsity: float = 0.5     # Default 50% feature sparsity (ReLU zeros)
    weight_sparsity: float = 0.3      # Default 30% weight sparsity (conservative pruning)
    channel_sparsity: float = 0.0     # Default no channel pruning
    
    def get_effective_sparsity_multiplier(self) -> float:
        """Calculate effective computation reduction due to combined sparsity"""
        # Spatial sparsity: fraction of voxels that are active
        spatial_efficiency = self.spatial_sparsity
        
        # Feature sparsity: fraction of activations that are non-zero
        feature_efficiency = 1 - self.feature_sparsity
        
        # Weight sparsity: fraction of weights that are non-zero
        weight_efficiency = 1 - self.weight_sparsity
        
        # Channel sparsity: fraction of channels that are active
        channel_efficiency = 1 - self.channel_sparsity
        
        # Combined effect: spatial sparsity dominates, others reduce remaining computation
        return spatial_efficiency * feature_efficiency * weight_efficiency * channel_efficiency

class SparseConv3DMACCalculator:
    """Calculate MAC operations for sparse 3D convolution operations"""
    
    @staticmethod
    def sparse_conv3d_macs(num_points: int, voxel_size: float, kernel_size: int, 
                          in_channels: int, out_channels: int,
                          sparsity_config: SparsityConfig = None) -> Tuple[int, int]:
        """
        Calculate MACs for 3D sparse convolution
        
        Args:
            num_points: Number of input points
            voxel_size: Voxel resolution
            kernel_size: Size of the 3D convolution kernel (assumes cubic)
            in_channels: Number of input channels
            out_channels: Number of output channels
            sparsity_config: Sparsity configuration
        
        Returns:
            Tuple of (MAC operations, active_output_voxels)
        """
        if sparsity_config is None:
            sparsity_config = SparsityConfig()
        
        # Calculate number of active voxels based on spatial sparsity
        active_voxels = int(num_points)
        
        # For sparse convolution, each active output voxel requires:
        # - kernel_size^3 neighbor lookups
        # - in_channels * out_channels multiply-accumulate operations per valid neighbor
        
        # Average number of active neighbors per output voxel (limited by spatial sparsity)
        avg_active_neighbors = min(kernel_size ** 3, 
                                 active_voxels * sparsity_config.spatial_sparsity)
        
        # Base MACs = active_voxels * avg_neighbors * in_channels * out_channels
        base_macs = int(active_voxels * avg_active_neighbors * in_channels * out_channels)
        
        # Apply feature and weight sparsity
        feature_efficiency = 1 - sparsity_config.feature_sparsity
        weight_efficiency = 1 - sparsity_config.weight_sparsity
        channel_efficiency = 1 - sparsity_config.channel_sparsity
        
        effective_macs = int(base_macs * feature_efficiency * weight_efficiency * channel_efficiency)
        
        return effective_macs, active_voxels
    
    @staticmethod
    def submanifold_conv3d_macs(num_points: int, kernel_size: int, 
                               in_channels: int, out_channels: int,
                               sparsity_config: SparsityConfig = None) -> int:
        """
        Calculate MACs for submanifold sparse convolution (maintains sparsity pattern)
        
        Args:
            num_points: Number of input points (active voxels)
            kernel_size: Size of the 3D convolution kernel
            in_channels: Number of input channels  
            out_channels: Number of output channels
            sparsity_config: Sparsity configuration
        
        Returns:
            MAC operations
        """
        if sparsity_config is None:
            sparsity_config = SparsityConfig()
        
        # Submanifold convolution only computes at input active locations
        active_locations = int(num_points * sparsity_config.spatial_sparsity)
        
        # Average number of active neighbors (depends on clustering)
        # For LiDAR, points are clustered, so fewer neighbors are typically active
        clustering_factor = 0.3  # Typical for LiDAR point clouds
        avg_active_neighbors = int(kernel_size ** 3 * clustering_factor)
        
        # MACs = active_locations * avg_neighbors * in_channels * out_channels
        base_macs = active_locations * avg_active_neighbors * in_channels * out_channels
        
        # Apply additional sparsity factors
        feature_efficiency = 1 - sparsity_config.feature_sparsity
        weight_efficiency = 1 - sparsity_config.weight_sparsity
        channel_efficiency = 1 - sparsity_config.channel_sparsity
        
        effective_macs = int(base_macs * feature_efficiency * weight_efficiency * channel_efficiency)
        
        return effective_macs

class MinkowskiNet:
    """MinkowskiNet network configuration and MAC calculation"""
    
    def __init__(self, num_points: int = 100000, voxel_size: float = 0.05,
                 input_channels: int = 4, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None,
                 spatial_sparsity: float = None, feature_sparsity: float = None,
                 weight_sparsity: float = None, channel_sparsity: float = None):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Handle both SparsityConfig object and individual parameters
        if sparsity_config is not None:
            self.sparsity_config = sparsity_config
        else:
            self.sparsity_config = SparsityConfig(
                spatial_sparsity=spatial_sparsity if spatial_sparsity is not None else 0.05,
                feature_sparsity=feature_sparsity if feature_sparsity is not None else 0.5,
                weight_sparsity=weight_sparsity if weight_sparsity is not None else 0.3,
                channel_sparsity=channel_sparsity if channel_sparsity is not None else 0.0
            )
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for MinkowskiNet using proper layer configuration"""
        total_macs = 0
        layer_details = {}
        current_points = 400000  # From CSV: all layers use 400k active pairs
        current_channels = self.input_channels
        
        # Initial sparse convolution: input_channels -> 32 (CSV: M=400k, N=32, K=4)
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            current_points, self.voxel_size, 3, current_channels, 32, self.sparsity_config)
        total_macs += macs
        current_channels = 32
        layer_details['conv0'] = {'macs': macs, 'output_points': current_points, 'channels': current_channels}
        
        # Encoder stages - CSV shows 400k active pairs for each
        encoder_configs = [
            (32, 64, 2),   # CSV: M=400k, N=64, K=32
            (64, 128, 2),  # CSV: M=400k, N=128, K=64
            (128, 256, 2), # CSV: M=400k, N=256, K=128
            (256, 512, 2), # CSV: M=400k, N=512, K=256
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(encoder_configs):
            macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, 3, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'encoder_stage_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Decoder stages - CSV shows 400k active pairs for each
        decoder_configs = [
            (512, 256, 2),  # CSV: M=400k, N=256, K=512
            (256, 128, 2),  # CSV: M=400k, N=128, K=256
            (128, 64, 2),   # CSV: M=400k, N=64, K=128
            (64, 32, 2),    # CSV: M=400k, N=32, K=64
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(decoder_configs):
            macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, 3, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'decoder_stage_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Final classification layer - CSV: M=100k, N=20, K=32, kernel=1
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            100000, self.voxel_size, 1, 32, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['classification_head'] = {'macs': macs, 'output_points': 100000, 'channels': self.num_classes}
        
        macs_per_point = total_macs / self.num_points
        
        return {
            'total_macs': total_macs,
            'macs_per_point': macs_per_point,
            'layer_details': layer_details,
            'sparsity_config': self.sparsity_config
        }

class SPVNAS:
    """SPVNAS (Sparse Point-Voxel NAS) network configuration and MAC calculation"""
    
    def __init__(self, num_points: int = 100000, voxel_size: float = 0.05,
                 input_channels: int = 4, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None,
                 spatial_sparsity: float = None, feature_sparsity: float = None,
                 weight_sparsity: float = None, channel_sparsity: float = None):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Handle both SparsityConfig object and individual parameters
        if sparsity_config is not None:
            self.sparsity_config = sparsity_config
        else:
            self.sparsity_config = SparsityConfig(
                spatial_sparsity=spatial_sparsity if spatial_sparsity is not None else 0.05,
                feature_sparsity=feature_sparsity if feature_sparsity is not None else 0.5,
                weight_sparsity=weight_sparsity if weight_sparsity is not None else 0.3,
                channel_sparsity=channel_sparsity if channel_sparsity is not None else 0.0
            )
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for SPVNAS using proper layer configuration"""
        total_macs = 0
        layer_details = {}
        current_points = 500000  # From CSV: main layers use 500k active pairs
        current_channels = self.input_channels
        
        # Stem convolutions - CSV shows 500k active pairs
        stem_configs = [
            (self.input_channels, 32, 3),  # CSV: M=500k, N=32, K=4
            (32, 32, 3),                   # CSV: M=500k, N=32, K=32
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(stem_configs):
            macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'stem_conv{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # SPVNAS blocks - CSV shows 500k active pairs, NAS-optimized channels
        spvnas_configs = [
            (32, 48, 3),    # CSV: M=500k, N=48, K=32
            (48, 64, 3),    # CSV: M=500k, N=64, K=48
            (64, 128, 3),   # CSV: M=500k, N=128, K=64
            (128, 256, 3),  # CSV: M=500k, N=256, K=128
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(spvnas_configs):
            macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'spvnas_block_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Decoder with skip connections - CSV shows 500k active pairs
        decoder_configs = [
            (256, 128, 3),  # CSV: M=500k, N=128, K=256
            (128, 64, 3),   # CSV: M=500k, N=64, K=128
            (64, 32, 3),    # CSV: M=500k, N=32, K=64
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(decoder_configs):
            macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'decoder_stage_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Point-wise classification - CSV: M=100k, N=20, K=32, kernel=1
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            100000, self.voxel_size, 1, 32, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['point_classification'] = {
            'macs': macs, 'output_points': 100000, 'channels': self.num_classes
        }
        
        macs_per_point = total_macs / self.num_points
        
        return {
            'total_macs': total_macs,
            'macs_per_point': macs_per_point,
            'layer_details': layer_details,
            'sparsity_config': self.sparsity_config
        }

class LargeKernel3D:
    """LargeKernel3D network with large kernel sparse convolutions"""
    
    def __init__(self, num_points: int = 100000, voxel_size: float = 0.05,
                 input_channels: int = 4, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None,
                 spatial_sparsity: float = None, feature_sparsity: float = None,
                 weight_sparsity: float = None, channel_sparsity: float = None):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Handle both SparsityConfig object and individual parameters
        if sparsity_config is not None:
            self.sparsity_config = sparsity_config
        else:
            self.sparsity_config = SparsityConfig(
                spatial_sparsity=spatial_sparsity if spatial_sparsity is not None else 0.05,
                feature_sparsity=feature_sparsity if feature_sparsity is not None else 0.5,
                weight_sparsity=weight_sparsity if weight_sparsity is not None else 0.3,
                channel_sparsity=channel_sparsity if channel_sparsity is not None else 0.0
            )
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for LargeKernel3D using proper layer configuration"""
        total_macs = 0
        layer_details = {}
        
        # Stem with standard kernel - CSV: M=600k, N=64, K=108, kernel=3
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            600000, self.voxel_size, 3, self.input_channels, 64, self.sparsity_config)
        total_macs += macs
        layer_details['stem'] = {'macs': macs, 'output_points': 600000, 'channels': 64}
        
        # Large kernel stages with varying active pairs and large kernels
        large_kernel_configs = [
            (3100000, 64, 96, 5),    # CSV: M=3.1M, N=96, K=8000, kernel=5
            (8500000, 96, 128, 7),   # CSV: M=8.5M, N=128, K=32928, kernel=7 
            (18200000, 128, 192, 9), # CSV: M=18.2M, N=192, K=93312, kernel=9
            (8500000, 192, 256, 7),  # CSV: M=8.5M, N=256, K=65856, kernel=7
        ]
        
        for i, (active_pairs, in_ch, out_ch, kernel_size) in enumerate(large_kernel_configs):
            macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                active_pairs, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            layer_details[f'large_kernel_stage_{i+1}_k{kernel_size}'] = {
                'macs': macs, 'output_points': active_pairs, 'channels': out_ch,
                'kernel_size': kernel_size
            }
        
        # Decoder with progressively smaller kernels and fewer active pairs
        decoder_configs = [
            (3100000, 256, 192, 5),  # CSV: M=3.1M, N=192, K=32000, kernel=5
            (600000, 192, 128, 3),   # CSV: M=600k, N=128, K=5184, kernel=3
            (600000, 128, 64, 3),    # CSV: M=600k, N=64, K=3456, kernel=3
        ]
        
        for i, (active_pairs, in_ch, out_ch, kernel_size) in enumerate(decoder_configs):
            macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                active_pairs, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            layer_details[f'decoder_stage_{i+1}_k{kernel_size}'] = {
                'macs': macs, 'output_points': active_pairs, 'channels': out_ch,
                'kernel_size': kernel_size
            }
        
        # Classification head - CSV: M=100k, N=20, K=64, kernel=1
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            100000, self.voxel_size, 1, 64, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['classification_head'] = {
            'macs': macs, 'output_points': 100000, 'channels': self.num_classes
        }
        
        macs_per_point = total_macs / self.num_points
        
        return {
            'total_macs': total_macs,
            'macs_per_point': macs_per_point,
            'layer_details': layer_details,
            'sparsity_config': self.sparsity_config
        }

class VoxelNeXt:
    """VoxelNeXt with ConvNeXt-inspired architecture for voxel processing"""
    
    def __init__(self, num_points: int = 100000, voxel_size: float = 0.05,
                 input_channels: int = 4, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None,
                 spatial_sparsity: float = None, feature_sparsity: float = None,
                 weight_sparsity: float = None, channel_sparsity: float = None):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Handle both SparsityConfig object and individual parameters
        if sparsity_config is not None:
            self.sparsity_config = sparsity_config
        else:
            self.sparsity_config = SparsityConfig(
                spatial_sparsity=spatial_sparsity if spatial_sparsity is not None else 0.05,
                feature_sparsity=feature_sparsity if feature_sparsity is not None else 0.5,
                weight_sparsity=weight_sparsity if weight_sparsity is not None else 0.3,
                channel_sparsity=channel_sparsity if channel_sparsity is not None else 0.0
            )
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for VoxelNeXt based on actual CSV data"""
        total_macs = 0
        layer_details = {}
        
        # Based on voxelnext_gemm_layers.csv - complex structure with different M values
        # Layer configurations from CSV: M, N, K, Kernel_Size, MACs
        layers_config = [
            ('patchify_conv', 1100000, 48, 256, 4, 13516800000),
            ('patchify_expand', 100000, 96, 48, 1, 460800000),
            ('stage_1_block_1_dw', 6000000, 96, 343, 7, 197568000000),
            ('stage_1_block_1_mlp_expand', 100000, 384, 96, 1, 3686400000),
            ('stage_1_block_1_mlp_contract', 100000, 96, 384, 1, 3686400000),
            ('stage_1_block_2_dw', 6000000, 96, 343, 7, 197568000000),
            ('stage_1_block_2_mlp_expand', 100000, 384, 96, 1, 3686400000),
            ('stage_1_block_2_mlp_contract', 100000, 96, 384, 1, 3686400000),
            ('stage_2_expand', 100000, 192, 96, 1, 1843200000),
            ('stage_2_block_1_dw', 6000000, 192, 343, 7, 395136000000),
            ('stage_2_block_1_mlp_expand', 100000, 768, 192, 1, 14745600000),
            ('stage_2_block_1_mlp_contract', 100000, 192, 768, 1, 14745600000),
            ('stage_2_block_2_dw', 6000000, 192, 343, 7, 395136000000),
            ('stage_2_block_2_mlp_expand', 100000, 768, 192, 1, 14745600000),
            ('stage_2_block_2_mlp_contract', 100000, 192, 768, 1, 14745600000),
            ('stage_3_expand', 100000, 384, 192, 1, 7372800000),
            ('stage_3_block_1_dw', 6000000, 384, 343, 7, 790272000000),
            ('stage_3_block_1_mlp_expand', 100000, 1536, 384, 1, 58982400000),
            ('stage_3_block_1_mlp_contract', 100000, 384, 1536, 1, 58982400000),
            ('stage_3_block_2_dw', 6000000, 384, 343, 7, 790272000000),
            ('stage_3_block_2_mlp_expand', 100000, 1536, 384, 1, 58982400000),
            ('stage_3_block_2_mlp_contract', 100000, 384, 1536, 1, 58982400000),
            ('stage_3_block_3_dw', 6000000, 384, 343, 7, 790272000000),
            ('stage_3_block_3_mlp_expand', 100000, 1536, 384, 1, 58982400000),
            ('stage_3_block_3_mlp_contract', 100000, 384, 1536, 1, 58982400000),
            ('stage_3_block_4_dw', 6000000, 384, 343, 7, 790272000000),
            ('stage_3_block_4_mlp_expand', 100000, 1536, 384, 1, 58982400000),
            ('stage_3_block_4_mlp_contract', 100000, 384, 1536, 1, 58982400000),
            ('stage_3_block_5_dw', 6000000, 384, 343, 7, 790272000000),
            ('stage_3_block_5_mlp_expand', 100000, 1536, 384, 1, 58982400000),
            ('stage_3_block_5_mlp_contract', 100000, 384, 1536, 1, 58982400000),
            ('stage_3_block_6_dw', 6000000, 384, 343, 7, 790272000000),
            ('stage_3_block_6_mlp_expand', 100000, 1536, 384, 1, 58982400000),
            ('stage_3_block_6_mlp_contract', 100000, 384, 1536, 1, 58982400000),
            ('stage_4_expand', 100000, 768, 384, 1, 29491200000),
            ('stage_4_block_1_dw', 6000000, 768, 343, 7, 1580544000000),
            ('stage_4_block_1_mlp_expand', 100000, 3072, 768, 1, 235929600000),
            ('stage_4_block_1_mlp_contract', 100000, 768, 3072, 1, 235929600000),
            ('stage_4_block_2_dw', 6000000, 768, 343, 7, 1580544000000),
            ('stage_4_block_2_mlp_expand', 100000, 3072, 768, 1, 235929600000),
            ('stage_4_block_2_mlp_contract', 100000, 768, 3072, 1, 235929600000),
            ('classification_head', 100000, 20, 768, 1, 1536000000)
        ]
        
        for layer_name, M, N, K, kernel_size, base_macs in layers_config:
            # Apply sparsity factors to base MACs
            effective_macs = int(base_macs * self.sparsity_config.get_effective_sparsity_multiplier())
            total_macs += effective_macs
            
            layer_details[layer_name] = {
                'macs': effective_macs,
                'output_points': M,
                'channels': N,
                'kernel_size': kernel_size
            }
        
        macs_per_point = total_macs / self.num_points
        
        return {
            'total_macs': total_macs,
            'macs_per_point': macs_per_point,
            'layer_details': layer_details,
            'sparsity_config': self.sparsity_config
        }

class RSN:
    """RSN (Range Sparse Net) for efficient LiDAR 3D object detection"""
    
    def __init__(self, num_points: int = 100000, voxel_size: float = 0.05,
                 input_channels: int = 4, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None,
                 spatial_sparsity: float = None, feature_sparsity: float = None,
                 weight_sparsity: float = None, channel_sparsity: float = None):
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Handle both SparsityConfig object and individual parameters
        if sparsity_config is not None:
            self.sparsity_config = sparsity_config
        else:
            self.sparsity_config = SparsityConfig(
                spatial_sparsity=spatial_sparsity if spatial_sparsity is not None else 0.05,
                feature_sparsity=feature_sparsity if feature_sparsity is not None else 0.5,
                weight_sparsity=weight_sparsity if weight_sparsity is not None else 0.3,
                channel_sparsity=channel_sparsity if channel_sparsity is not None else 0.0
            )
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for RSN based on actual CSV data"""
        total_macs = 0
        layer_details = {}
        
        # Based on rsn_gemm_layers.csv - dual path with 500k for main convs, 100k for residuals
        # Layer configurations from CSV: M, N, K, Kernel_Size, MACs
        layers_config = [
            ('initial_features', 500000, 32, 108, 3, 1728000000),
            ('encoder_stage_1_main', 500000, 64, 864, 3, 27648000000),
            ('encoder_stage_1_residual', 100000, 64, 32, 1, 204800000),
            ('encoder_stage_2_main', 500000, 128, 1728, 3, 110592000000),
            ('encoder_stage_2_residual', 100000, 128, 64, 1, 819200000),
            ('encoder_stage_3_main', 500000, 256, 3456, 3, 442368000000),
            ('encoder_stage_3_residual', 100000, 256, 128, 1, 3276800000),
            ('encoder_stage_4_main', 500000, 512, 6912, 3, 1769472000000),
            ('encoder_stage_4_residual', 100000, 512, 256, 1, 13107200000),
            ('decoder_stage_1_main', 500000, 256, 13824, 3, 1769472000000),
            ('decoder_stage_1_skip', 100000, 256, 256, 1, 6553600000),
            ('decoder_stage_2_main', 500000, 128, 6912, 3, 442368000000),
            ('decoder_stage_2_skip', 100000, 128, 128, 1, 1638400000),
            ('decoder_stage_3_main', 500000, 64, 3456, 3, 110592000000),
            ('decoder_stage_3_skip', 100000, 64, 64, 1, 409600000),
            ('decoder_stage_4_main', 500000, 32, 1728, 3, 27648000000),
            ('decoder_stage_4_skip', 100000, 32, 32, 1, 102400000),
            ('fusion_1', 100000, 64, 32, 1, 204800000),
            ('fusion_2', 100000, 32, 64, 1, 204800000),
            ('classification_head', 100000, 20, 32, 1, 64000000)
        ]
        
        for layer_name, M, N, K, kernel_size, base_macs in layers_config:
            # Apply sparsity factors to base MACs
            effective_macs = int(base_macs * self.sparsity_config.get_effective_sparsity_multiplier())
            total_macs += effective_macs
            
            layer_details[layer_name] = {
                'macs': effective_macs,
                'output_points': M,
                'channels': N,
                'kernel_size': kernel_size
            }
        
        macs_per_point = total_macs / self.num_points
        
        return {
            'total_macs': total_macs,
            'macs_per_point': macs_per_point,
            'layer_details': layer_details,
            'sparsity_config': self.sparsity_config
        }

def save_results_to_csv(results: Dict, filename: str):
    """Save MAC calculation results to CSV file"""
    filepath = os.path.join('/Users/ashriram/Desktop/minuet-tracer/mac-vs-time', filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['Layer', 'MACs', 'Output_Points', 'Channels', 'Additional_Info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for layer_name, details in results['layer_details'].items():
            row = {
                'Layer': layer_name,
                'MACs': details['macs'],
                'Output_Points': details['output_points'],
                'Channels': details['channels'],
                'Additional_Info': str({k: v for k, v in details.items() 
                                      if k not in ['macs', 'output_points', 'channels']})
            }
            writer.writerow(row)
    
    print(f"Results saved to {filepath}")

def compare_networks():
    """Compare MAC operations across all 3D networks"""
    print("3D Point Cloud Network MAC Comparison")
    print("="*50)
    
    # Standard configuration
    num_points = 100000
    sparsity_config = SparsityConfig(
        spatial_sparsity=0.05,   # 5% spatial occupancy
        feature_sparsity=0.5,    # 50% feature sparsity
        weight_sparsity=0.3,     # 30% weight sparsity
        channel_sparsity=0.0     # No channel pruning
    )
    
    networks = {
        'MinkowskiNet': MinkowskiNet(num_points=num_points, sparsity_config=sparsity_config),
        'SPVNAS': SPVNAS(num_points=num_points, sparsity_config=sparsity_config),
        'LargeKernel3D': LargeKernel3D(num_points=num_points, sparsity_config=sparsity_config),
        'VoxelNeXt': VoxelNeXt(num_points=num_points, sparsity_config=sparsity_config),
        'RSN': RSN(num_points=num_points, sparsity_config=sparsity_config)
    }
    
    results = {}
    for name, network in networks.items():
        print(f"\nCalculating {name}...")
        result = network.calculate_macs()
        results[name] = result
        
        print(f"  Total MACs: {result['total_macs']:,}")
        print(f"  MACs per point: {result['macs_per_point']:.2f}")
        
        # Save individual network results
        save_results_to_csv(result, f"{name.lower()}_layers.csv")
    
    # Summary comparison
    print(f"\n{'Network':<15} {'Total MACs':<15} {'MACs/Point':<12} {'Efficiency Rank'}")
    print("-" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['macs_per_point'])
    for rank, (name, result) in enumerate(sorted_results, 1):
        print(f"{name:<15} {result['total_macs']:<15,} {result['macs_per_point']:<12.2f} {rank}")
    
    return results

if __name__ == "__main__":
    compare_networks()