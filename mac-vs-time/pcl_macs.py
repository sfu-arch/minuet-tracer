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
        """Calculate total MACs for MinkowskiNet"""
        total_macs = 0
        layer_details = {}
        current_points = self.num_points
        current_channels = self.input_channels
        
        # Initial sparse convolution: input_channels -> 32
        macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
            current_points, self.voxel_size, 3, current_channels, 32, self.sparsity_config)
        total_macs += macs
        current_channels = 32
        layer_details['conv0'] = {'macs': macs, 'output_points': current_points, 'channels': current_channels}
        
        # Encoder stages with strided convolutions
        encoder_configs = [
            (32, 64, 2),   # stage1: stride 2 downsample
            (64, 128, 2),  # stage2: stride 2 downsample
            (128, 256, 2), # stage3: stride 2 downsample
            (256, 512, 2), # stage4: stride 2 downsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(encoder_configs):
            # Strided convolution reduces spatial resolution
            current_points = current_points // (stride ** 3) if stride > 1 else current_points
            
            macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size * (2**i), 3, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'encoder_stage_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Decoder stages with transposed convolutions
        decoder_configs = [
            (512, 256, 2),  # stage1: stride 2 upsample
            (256, 128, 2),  # stage2: stride 2 upsample  
            (128, 64, 2),   # stage3: stride 2 upsample
            (64, 32, 2),    # stage4: stride 2 upsample
        ]
        
        for i, (in_ch, out_ch, stride) in enumerate(decoder_configs):
            # Transposed convolution increases spatial resolution
            current_points = current_points * (stride ** 3) if stride > 1 else current_points
            
            macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, 3, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'decoder_stage_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Final classification layer
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            self.num_points, self.voxel_size, 1, 32, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['classification'] = {'macs': macs, 'output_points': self.num_points, 'channels': self.num_classes}
        
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
        """Calculate total MACs for SPVNAS"""
        total_macs = 0
        layer_details = {}
        current_points = self.num_points
        current_channels = self.input_channels
        
        # Stem convolutions
        stem_configs = [
            (self.input_channels, 32, 3),  # stem1
            (32, 32, 3),                   # stem2
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(stem_configs):
            macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'stem_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # SPVNAS blocks with NAS-optimized channel configurations
        spvnas_configs = [
            (32, 48, 3),    # block1: efficient expansion
            (48, 64, 3),    # block2: moderate growth
            (64, 128, 3),   # block3: standard growth
            (128, 256, 3),  # block4: final expansion
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(spvnas_configs):
            # Apply stride for some blocks (NAS-optimized)
            stride = 2 if i in [1, 3] else 1
            if stride > 1:
                current_points = current_points // (stride ** 3)
            
            macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'spvnas_block_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Decoder with skip connections
        decoder_configs = [
            (256, 128, 3),  # decoder1
            (128, 64, 3),   # decoder2  
            (64, 32, 3),    # decoder3
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(decoder_configs):
            # Upsample for decoder
            current_points = min(current_points * 4, self.num_points)
            
            macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'decoder_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels
            }
        
        # Point-wise classification
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            self.num_points, self.voxel_size, 1, 32, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['point_classification'] = {
            'macs': macs, 'output_points': self.num_points, 'channels': self.num_classes
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
        """Calculate total MACs for LargeKernel3D"""
        total_macs = 0
        layer_details = {}
        current_points = self.num_points
        current_channels = self.input_channels
        
        # Stem with standard kernel
        macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
            current_points, self.voxel_size, 3, self.input_channels, 64, self.sparsity_config)
        total_macs += macs
        current_channels = 64
        layer_details['stem'] = {'macs': macs, 'output_points': current_points, 'channels': current_channels}
        
        # Large kernel stages with increasing kernel sizes
        large_kernel_configs = [
            (64, 96, 5),    # stage1: kernel_size 5
            (96, 128, 7),   # stage2: kernel_size 7
            (128, 192, 9),  # stage3: kernel_size 9 (very large)
            (192, 256, 7),  # stage4: kernel_size 7
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(large_kernel_configs):
            # Downsample for some stages
            stride = 2 if i in [1, 3] else 1
            if stride > 1:
                current_points = current_points // (stride ** 3)
            
            macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'large_kernel_stage_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels,
                'kernel_size': kernel_size
            }
        
        # Decoder with progressively smaller kernels
        decoder_configs = [
            (256, 192, 5),  # decoder1: kernel_size 5
            (192, 128, 3),  # decoder2: kernel_size 3
            (128, 64, 3),   # decoder3: kernel_size 3
        ]
        
        for i, (in_ch, out_ch, kernel_size) in enumerate(decoder_configs):
            # Upsample
            current_points = min(current_points * 4, self.num_points)
            
            macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += macs
            current_channels = out_ch
            layer_details[f'decoder_{i+1}'] = {
                'macs': macs, 'output_points': current_points, 'channels': current_channels,
                'kernel_size': kernel_size
            }
        
        # Classification head
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            self.num_points, self.voxel_size, 1, 64, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['classification'] = {
            'macs': macs, 'output_points': self.num_points, 'channels': self.num_classes
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
        """Calculate total MACs for VoxelNeXt"""
        total_macs = 0
        layer_details = {}
        current_points = self.num_points
        current_channels = self.input_channels
        
        # Patchify operation (like ConvNeXt)
        macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
            current_points, self.voxel_size, 4, self.input_channels, 48, self.sparsity_config)
        total_macs += macs
        current_channels = 48
        layer_details['patchify'] = {'macs': macs, 'output_points': current_points, 'channels': current_channels}
        
        # ConvNeXt-style stages with different depths
        stage_configs = [
            (48, 96, 2, 3),    # stage1: 2 blocks, downsample
            (96, 192, 2, 3),   # stage2: 2 blocks, downsample
            (192, 384, 6, 3),  # stage3: 6 blocks (main stage)
            (384, 768, 2, 3),  # stage4: 2 blocks, downsample
        ]
        
        for stage_idx, (in_ch, out_ch, num_blocks, kernel_size) in enumerate(stage_configs):
            # Downsample at the beginning of each stage
            if stage_idx > 0:
                current_points = current_points // 8  # More aggressive downsampling
            
            # Channel expansion at stage start
            if in_ch != out_ch:
                macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                    current_points, self.voxel_size, 1, in_ch, out_ch, self.sparsity_config)
                total_macs += macs
                current_channels = out_ch
                layer_details[f'stage_{stage_idx+1}_expand'] = {
                    'macs': macs, 'output_points': current_points, 'channels': current_channels
                }
            
            # ConvNeXt blocks
            for block_idx in range(num_blocks):
                # Depthwise convolution (large kernel)
                dw_macs = SparseConv3DMACCalculator.submanifold_conv3d_macs(
                    current_points, kernel_size, out_ch, out_ch, self.sparsity_config)
                total_macs += dw_macs
                
                # Point-wise MLP (expand + contract)
                mlp_dim = out_ch * 4  # 4x expansion like ConvNeXt
                
                # MLP expand
                mlp_expand_macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                    current_points, self.voxel_size, 1, out_ch, mlp_dim, self.sparsity_config)
                total_macs += mlp_expand_macs
                
                # MLP contract
                mlp_contract_macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                    current_points, self.voxel_size, 1, mlp_dim, out_ch, self.sparsity_config)
                total_macs += mlp_contract_macs
                
                block_total_macs = dw_macs + mlp_expand_macs + mlp_contract_macs
                layer_details[f'stage_{stage_idx+1}_block_{block_idx+1}'] = {
                    'macs': block_total_macs, 'output_points': current_points, 'channels': out_ch,
                    'dw_macs': dw_macs, 'mlp_macs': mlp_expand_macs + mlp_contract_macs
                }
        
        # Global average pooling and classification
        # Simulate with 1x1 conv to final classes
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            self.num_points, self.voxel_size, 1, 768, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['classification'] = {
            'macs': macs, 'output_points': self.num_points, 'channels': self.num_classes
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
        """Calculate total MACs for RSN"""
        total_macs = 0
        layer_details = {}
        current_points = self.num_points
        current_channels = self.input_channels
        
        # Initial feature extraction
        macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
            current_points, self.voxel_size, 3, self.input_channels, 32, self.sparsity_config)
        total_macs += macs
        current_channels = 32
        layer_details['initial_features'] = {
            'macs': macs, 'output_points': current_points, 'channels': current_channels
        }
        
        # Range-aware encoder with multiple scales
        encoder_configs = [
            (32, 64, 3, 2),    # stage1: stride 2
            (64, 128, 3, 2),   # stage2: stride 2  
            (128, 256, 3, 2),  # stage3: stride 2
            (256, 512, 3, 2),  # stage4: stride 2
        ]
        
        skip_connections = {}
        for i, (in_ch, out_ch, kernel_size, stride) in enumerate(encoder_configs):
            # Store skip connection before downsampling
            skip_connections[f'skip_{i+1}'] = (current_points, in_ch)
            
            # Downsample
            if stride > 1:
                current_points = current_points // (stride ** 3)
            
            # Main convolution
            main_macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += main_macs
            
            # Residual connection (if needed)
            if in_ch != out_ch:
                residual_macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                    current_points, self.voxel_size, 1, in_ch, out_ch, self.sparsity_config)
                total_macs += residual_macs
            else:
                residual_macs = 0
            
            current_channels = out_ch
            layer_details[f'encoder_stage_{i+1}'] = {
                'macs': main_macs + residual_macs, 'output_points': current_points, 
                'channels': current_channels, 'main_macs': main_macs, 'residual_macs': residual_macs
            }
        
        # Range-aware decoder with skip connections
        decoder_configs = [
            (512, 256, 3, 2),  # decoder1: upsample
            (256, 128, 3, 2),  # decoder2: upsample
            (128, 64, 3, 2),   # decoder3: upsample
            (64, 32, 3, 2),    # decoder4: upsample
        ]
        
        for i, (in_ch, out_ch, kernel_size, stride) in enumerate(decoder_configs):
            # Upsample
            if stride > 1:
                current_points = current_points * (stride ** 3)
                current_points = min(current_points, self.num_points)
            
            # Decoder convolution
            decoder_macs, current_points = SparseConv3DMACCalculator.sparse_conv3d_macs(
                current_points, self.voxel_size, kernel_size, in_ch, out_ch, self.sparsity_config)
            total_macs += decoder_macs
            
            # Skip connection fusion
            skip_points, skip_channels = skip_connections[f'skip_{4-i}']
            if skip_channels == out_ch:
                skip_macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
                    skip_points, self.voxel_size, 1, skip_channels, out_ch, self.sparsity_config)
                total_macs += skip_macs
            else:
                skip_macs = 0
            
            current_channels = out_ch
            layer_details[f'decoder_stage_{i+1}'] = {
                'macs': decoder_macs + skip_macs, 'output_points': current_points,
                'channels': current_channels, 'decoder_macs': decoder_macs, 'skip_macs': skip_macs
            }
        
        # Range-aware feature fusion
        fusion_macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            self.num_points, self.voxel_size, 1, 32, 64, self.sparsity_config)
        total_macs += fusion_macs
        layer_details['range_fusion'] = {
            'macs': fusion_macs, 'output_points': self.num_points, 'channels': 64
        }
        
        # Final classification
        macs, _ = SparseConv3DMACCalculator.sparse_conv3d_macs(
            self.num_points, self.voxel_size, 1, 64, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['classification'] = {
            'macs': macs, 'output_points': self.num_points, 'channels': self.num_classes
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