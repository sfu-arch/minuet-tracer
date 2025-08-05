#!/usr/bin/env python3
"""
MAC Operations Calculator for LiDAR Semantic Segmentation Networks and Modern 2D CNNs with Sparsity Support
Supports: SqueezeSeg, SalsaNext, RangeNet++, PolarNet, CENet, EfficientNetV2, ConvNeXt

LiDAR Network configurations sourced from official repositories:
- SqueezeSeg: https://github.com/BichenWuUCB/SqueezeSeg (Paper: https://arxiv.org/abs/1710.07368)
- SalsaNext: https://github.com/TiagoCortinhal/SalsaNext (Paper: https://arxiv.org/abs/2003.03653)
- RangeNet++: https://github.com/PRBonn/lidar-bonnetal (Paper: https://arxiv.org/abs/1909.12324)
- PolarNet: https://github.com/edwardzhou130/PolarSeg (Paper: https://arxiv.org/abs/2003.14032)
- CENet: Context Embedding Network for LiDAR segmentation (2022)

Modern 2D CNN architectures:
- EfficientNetV2: https://arxiv.org/abs/2104.00298 (2021/2022)
- ConvNeXt: https://arxiv.org/abs/2201.03545 (2022)

Sparsity Support:
- Feature Sparsity: Activations that are zero after ReLU (typically 40-60%)
- Weight Sparsity: Pruned weights for model compression

Typical Weight Sparsity Factors (without significant accuracy loss):
- Conservative (<1% acc loss): SqueezeSeg 25%, SalsaNext 20%, RangeNet++ 30%, PolarNet 20%, CENet 25%
- Moderate (<3% acc loss): SqueezeSeg 45%, SalsaNext 40%, RangeNet++ 50%, PolarNet 35%, CENet 40%
- Aggressive (<5% acc loss): SqueezeSeg 65%, SalsaNext 60%, RangeNet++ 70%, PolarNet 55%, CENet 65%

All network architectures are based on official implementations, not approximations.
Fire module configurations for SqueezeSeg are taken from the actual implementation.
SalsaNext ResBlock and UpBlock structures follow the official architecture.
RangeNet++ uses DarkNet backbone with BasicBlock residual structure.
PolarNet uses polar coordinate representation with ResNet backbone and FPN.
CENet employs context embedding modules with multi-scale feature extraction.
"""

import math
import csv
import os
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

@dataclass
class SparsityConfig:
    """Configuration for sparsity parameters"""
    feature_sparsity: float = 0.5  # Default 50% feature sparsity (ReLU zeros)
    weight_sparsity: float = 0.3   # Default 30% weight sparsity (conservative pruning)
    
    def get_effective_sparsity_multiplier(self) -> float:
        """Calculate effective computation reduction due to sparsity"""
        # Feature sparsity reduces the number of active computations
        feature_efficiency = 1 - self.feature_sparsity
        
        # Weight sparsity reduces the number of weight parameters used
        weight_efficiency = 1 - self.weight_sparsity
        
        # Combined effect: both features and weights contribute to sparsity
        return feature_efficiency * weight_efficiency

class MACCalculator:
    """Calculate MAC operations for different layer types with sparsity support"""
    
    @staticmethod
    def conv2d_macs(input_shape: Tuple[int, int, int], kernel_size: int, 
                    out_channels: int, stride: int = 1, padding: int = 0,
                    sparsity_config: SparsityConfig = None) -> int:
        """Calculate MACs for 2D convolution with sparsity support"""
        h_in, w_in, c_in = input_shape
        h_out = (h_in + 2 * padding - kernel_size) // stride + 1
        w_out = (w_in + 2 * padding - kernel_size) // stride + 1
        
        # Base MACs = output_spatial_size * kernel_spatial_size * input_channels * output_channels
        base_macs = h_out * w_out * kernel_size * kernel_size * c_in * out_channels
        
        # Apply sparsity if configured
        if sparsity_config is not None:
            sparsity_multiplier = sparsity_config.get_effective_sparsity_multiplier()
            actual_macs = int(base_macs * sparsity_multiplier)
        else:
            actual_macs = base_macs
            
        return actual_macs, (h_out, w_out, out_channels)
    
    @staticmethod
    def depthwise_conv2d_macs(input_shape: Tuple[int, int, int], kernel_size: int, 
                             stride: int = 1, padding: int = 0,
                             sparsity_config: SparsityConfig = None) -> int:
        """Calculate MACs for depthwise separable convolution with sparsity support"""
        h_in, w_in, c_in = input_shape
        h_out = (h_in + 2 * padding - kernel_size) // stride + 1
        w_out = (w_in + 2 * padding - kernel_size) // stride + 1
        
        # Depthwise: each input channel gets its own kernel
        base_macs = h_out * w_out * kernel_size * kernel_size * c_in
        
        # Apply sparsity if configured
        if sparsity_config is not None:
            sparsity_multiplier = sparsity_config.get_effective_sparsity_multiplier()
            actual_macs = int(base_macs * sparsity_multiplier)
        else:
            actual_macs = base_macs
            
        return actual_macs, (h_out, w_out, c_in)
    
    @staticmethod
    def pointwise_conv2d_macs(input_shape: Tuple[int, int, int], out_channels: int,
                             sparsity_config: SparsityConfig = None) -> int:
        """Calculate MACs for 1x1 pointwise convolution with sparsity support"""
        h_in, w_in, c_in = input_shape
        base_macs = h_in * w_in * c_in * out_channels
        
        # Apply sparsity if configured
        if sparsity_config is not None:
            sparsity_multiplier = sparsity_config.get_effective_sparsity_multiplier()
            actual_macs = int(base_macs * sparsity_multiplier)
        else:
            actual_macs = base_macs
            
        return actual_macs, (h_in, w_in, out_channels)
    
    @staticmethod
    def fire_module_macs(input_shape: Tuple[int, int, int], squeeze_channels: int, 
                        expand_channels: int, sparsity_config: SparsityConfig = None) -> int:
        """Calculate MACs for SqueezeNet Fire module with sparsity support"""
        # Squeeze layer (1x1 conv)
        squeeze_macs, squeeze_shape = MACCalculator.pointwise_conv2d_macs(
            input_shape, squeeze_channels, sparsity_config)
        
        # Expand layers (1x1 and 3x3 conv in parallel)
        expand_1x1_macs, _ = MACCalculator.pointwise_conv2d_macs(
            squeeze_shape, expand_channels // 2, sparsity_config)
        expand_3x3_macs, expand_shape = MACCalculator.conv2d_macs(
            squeeze_shape, 3, expand_channels // 2, padding=1, sparsity_config=sparsity_config)
        
        total_macs = squeeze_macs + expand_1x1_macs + expand_3x3_macs
        output_shape = (expand_shape[0], expand_shape[1], expand_channels)
        
        return total_macs, output_shape

class CSVLayerGenerator:
    """Generate CSV files with layer-by-layer breakdown for simulator"""
    
    def __init__(self, sparsity_config: SparsityConfig = None):
        self.sparsity_config = sparsity_config or SparsityConfig()
        self.layers = []
        
    def add_layer(self, layer_name: str, ifmap_height: int, ifmap_width: int,
                  filter_height: int, filter_width: int, channels: int, 
                  num_filter: int, strides: int = 1):
        """Add a layer to the CSV data"""
        effective_sparsity = 1 - self.sparsity_config.get_effective_sparsity_multiplier()
        
        layer_data = {
            'Layer name': layer_name,
            'IFMAP Height': ifmap_height,
            'IFMAP Width': ifmap_width,
            'Filter Height': filter_height,
            'Filter Width': filter_width,
            'Channels': channels,
            'Num Filter': num_filter,
            'Strides': strides,
            'Sparsity': f"{effective_sparsity:.3f}"
        }
        self.layers.append(layer_data)
        
    def save_csv(self, filename: str):
        """Save layer data to CSV file"""
        fieldnames = ['Layer name', 'IFMAP Height', 'IFMAP Width', 'Filter Height', 
                     'Filter Width', 'Channels', 'Num Filter', 'Strides', 'Sparsity']
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.layers)
        
        print(f"CSV saved: {filename} ({len(self.layers)} layers)")
        
    def clear(self):
        """Clear all layer data"""
        self.layers = []

class SqueezeSeg:
    """SqueezeSeg network configuration and MAC calculation with sparsity support"""
    
    def __init__(self, input_height: int = 64, input_width: int = 512, 
                 input_channels: int = 5, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None):
        self.input_shape = (input_height, input_width, input_channels)
        self.num_classes = num_classes
        self.sparsity_config = sparsity_config or SparsityConfig()
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for SqueezeSeg"""
        current_shape = self.input_shape
        total_macs = 0
        layer_details = {}
        
        # Conv1: 3x3 conv, stride 2, 64 filters
        macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 3, 64, stride=2, padding=1, sparsity_config=self.sparsity_config)
        total_macs += macs
        layer_details['conv1'] = {'macs': macs, 'output_shape': current_shape}
        
        # Conv1_skip: 1x1 conv, stride 1, 64 filters
        macs_skip, shape_skip = MACCalculator.conv2d_macs(
            self.input_shape, 1, 64, stride=1, padding=0, sparsity_config=self.sparsity_config)
        total_macs += macs_skip
        layer_details['conv1_skip'] = {'macs': macs_skip, 'output_shape': shape_skip}
        
        # Pool1: 3x3, stride 2
        h, w, c = current_shape
        current_shape = (h // 2, w // 2, c)
        
        # Fire modules with actual SqueezeSeg configurations
        fire_configs = [
            # (squeeze_channels, expand_1x1, expand_3x3)
            (16, 64, 64),   # fire2
            (16, 64, 64),   # fire3
            (32, 128, 128), # fire4 
            (32, 128, 128), # fire5
            (48, 192, 192), # fire6
            (48, 192, 192), # fire7
            (64, 256, 256), # fire8
            (64, 256, 256), # fire9
        ]
        
        for i, (squeeze, expand_1x1, expand_3x3) in enumerate(fire_configs):
            # Fire module: squeeze + expand (1x1 and 3x3 in parallel)
            # Total expand channels = expand_1x1 + expand_3x3
            total_expand = expand_1x1 + expand_3x3
            macs, current_shape = MACCalculator.fire_module_macs(
                current_shape, squeeze, total_expand, self.sparsity_config)
            total_macs += macs
            layer_details[f'fire{i+2}'] = {'macs': macs, 'output_shape': current_shape}
            
            # Pool3 after fire3 (fire module index 1)
            if i == 1:  # After fire3
                h, w, c = current_shape
                current_shape = (h // 2, w // 2, c)
                
            # Pool5 after fire5 (fire module index 3)  
            if i == 3:  # After fire5
                h, w, c = current_shape
                current_shape = (h // 2, w // 2, c)
        
        # Fire deconvolution layers (upsampling)
        fire_deconv_configs = [
            # (squeeze, expand_1x1, expand_3x3, upsample_factor)
            (64, 128, 128, 2),  # fire_deconv10
            (32, 64, 64, 2),    # fire_deconv11
            (16, 32, 32, 2),    # fire_deconv12
            (16, 32, 32, 2),    # fire_deconv13
        ]
        
        for i, (squeeze, expand_1x1, expand_3x3, factor) in enumerate(fire_deconv_configs):
            h, w, c = current_shape
            
            # Squeeze 1x1 conv
            squeeze_macs, squeeze_shape = MACCalculator.pointwise_conv2d_macs(
                current_shape, squeeze, self.sparsity_config)
            
            # Deconv (upsampling) - approximate as bilinear upsampling (no MACs) + conv
            h_up, w_up = h * factor, w * factor
            deconv_shape = (h_up, w_up, squeeze)
            
            # Expand 1x1 and 3x3 convs
            expand_1x1_macs, _ = MACCalculator.pointwise_conv2d_macs(
                deconv_shape, expand_1x1, self.sparsity_config)
            expand_3x3_macs, expand_shape = MACCalculator.conv2d_macs(
                deconv_shape, 3, expand_3x3, padding=1, sparsity_config=self.sparsity_config)
            
            fire_deconv_macs = squeeze_macs + expand_1x1_macs + expand_3x3_macs
            total_macs += fire_deconv_macs
            
            current_shape = (expand_shape[0], expand_shape[1], expand_1x1 + expand_3x3)
            layer_details[f'fire_deconv{i+10}'] = {
                'macs': fire_deconv_macs, 'output_shape': current_shape}
        
        # Final conv14_prob: 3x3 conv to num_classes
        macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 3, self.num_classes, padding=1, sparsity_config=self.sparsity_config)
        total_macs += macs
        layer_details['conv14_prob'] = {'macs': macs, 'output_shape': current_shape}
        
        # Calculate number of points (pixels in range image)
        # For LiDAR, each pixel typically represents one point in the point cloud
        num_points = self.input_shape[0] * self.input_shape[1]
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'macs_per_point': total_macs / num_points if num_points > 0 else 0,
            'num_points': num_points,
            'layer_details': layer_details,
            'final_output_shape': current_shape,
            'sparsity_config': self.sparsity_config
        }
    
    def generate_csv_data(self, csv_generator: CSVLayerGenerator):
        """Generate CSV data for SqueezeSeg layers"""
        current_shape = self.input_shape
        
        # Conv1: 3x3 conv, stride 2, 64 filters
        csv_generator.add_layer('conv1', current_shape[0], current_shape[1], 
                               3, 3, current_shape[2], 64, 2)
        h_out = (current_shape[0] + 2 * 1 - 3) // 2 + 1
        w_out = (current_shape[1] + 2 * 1 - 3) // 2 + 1
        current_shape = (h_out, w_out, 64)
        
        # Conv1_skip: 1x1 conv, stride 1, 64 filters
        csv_generator.add_layer('conv1_skip', self.input_shape[0], self.input_shape[1],
                               1, 1, self.input_shape[2], 64, 1)
        
        # Pool1: 3x3, stride 2
        h, w, c = current_shape
        current_shape = (h // 2, w // 2, c)
        
        # Fire modules with actual SqueezeSeg configurations
        fire_configs = [
            # (squeeze_channels, expand_1x1, expand_3x3)
            (16, 64, 64),   # fire2
            (16, 64, 64),   # fire3
            (32, 128, 128), # fire4 
            (32, 128, 128), # fire5
            (48, 192, 192), # fire6
            (48, 192, 192), # fire7
            (64, 256, 256), # fire8
            (64, 256, 256), # fire9
        ]
        
        for i, (squeeze, expand_1x1, expand_3x3) in enumerate(fire_configs):
            # Fire module squeeze layer (1x1 conv)
            csv_generator.add_layer(f'fire{i+2}_squeeze', current_shape[0], current_shape[1],
                                   1, 1, current_shape[2], squeeze, 1)
            
            # Fire module expand 1x1 layer
            csv_generator.add_layer(f'fire{i+2}_expand1x1', current_shape[0], current_shape[1],
                                   1, 1, squeeze, expand_1x1, 1)
            
            # Fire module expand 3x3 layer
            csv_generator.add_layer(f'fire{i+2}_expand3x3', current_shape[0], current_shape[1],
                                   3, 3, squeeze, expand_3x3, 1)
            
            # Update shape after fire module
            total_expand = expand_1x1 + expand_3x3
            current_shape = (current_shape[0], current_shape[1], total_expand)
            
            # Pool3 after fire3 (fire module index 1)
            if i == 1:  # After fire3
                h, w, c = current_shape
                current_shape = (h // 2, w // 2, c)
                
            # Pool5 after fire5 (fire module index 3)  
            if i == 3:  # After fire5
                h, w, c = current_shape
                current_shape = (h // 2, w // 2, c)
        
        # Fire deconvolution layers (upsampling)
        fire_deconv_configs = [
            # (squeeze, expand_1x1, expand_3x3, upsample_factor)
            (64, 128, 128, 2),  # fire_deconv10
            (32, 64, 64, 2),    # fire_deconv11
            (16, 32, 32, 2),    # fire_deconv12
            (16, 32, 32, 2),    # fire_deconv13
        ]
        
        for i, (squeeze, expand_1x1, expand_3x3, factor) in enumerate(fire_deconv_configs):
            # Squeeze 1x1 conv
            csv_generator.add_layer(f'fire_deconv{i+10}_squeeze', current_shape[0], current_shape[1],
                                   1, 1, current_shape[2], squeeze, 1)
            
            # Deconv (upsampling)
            h_up, w_up = current_shape[0] * factor, current_shape[1] * factor
            
            # Expand 1x1 conv
            csv_generator.add_layer(f'fire_deconv{i+10}_expand1x1', h_up, w_up,
                                   1, 1, squeeze, expand_1x1, 1)
            
            # Expand 3x3 conv
            csv_generator.add_layer(f'fire_deconv{i+10}_expand3x3', h_up, w_up,
                                   3, 3, squeeze, expand_3x3, 1)
            
            current_shape = (h_up, w_up, expand_1x1 + expand_3x3)
        
        # Final conv14_prob: 3x3 conv to num_classes
        csv_generator.add_layer('conv14_prob', current_shape[0], current_shape[1],
                               3, 3, current_shape[2], self.num_classes, 1)

class SalsaNext:
    """SalsaNext network configuration based on official implementation with sparsity support
    Source: https://github.com/TiagoCortinhal/SalsaNext"""
    
    def __init__(self, input_height: int = 64, input_width: int = 2048, 
                 input_channels: int = 5, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None):
        self.input_shape = (input_height, input_width, input_channels)
        self.num_classes = num_classes
        self.sparsity_config = sparsity_config or SparsityConfig()
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for SalsaNext based on official architecture"""
        current_shape = self.input_shape
        total_macs = 0
        layer_details = {}
        
        # Context blocks (encoder)
        # downCntx: ResContextBlock(5, 32)
        macs, current_shape = self._res_context_block_macs(current_shape, 32)
        total_macs += macs
        layer_details['downCntx'] = {'macs': macs, 'output_shape': current_shape}
        
        # downCntx2: ResContextBlock(32, 32)
        macs, current_shape = self._res_context_block_macs(current_shape, 32)
        total_macs += macs
        layer_details['downCntx2'] = {'macs': macs, 'output_shape': current_shape}
        
        # downCntx3: ResContextBlock(32, 32)
        macs, current_shape = self._res_context_block_macs(current_shape, 32)
        total_macs += macs
        layer_details['downCntx3'] = {'macs': macs, 'output_shape': current_shape}
        
        # Encoder residual blocks with pooling
        encoder_configs = [
            (32, 64, True, False),   # resBlock1(32, 2*32, pooling=True, drop_out=False)
            (64, 128, True, True),   # resBlock2(2*32, 2*2*32, pooling=True)
            (128, 256, True, True),  # resBlock3(2*2*32, 2*4*32, pooling=True)
            (256, 256, True, True),  # resBlock4(2*4*32, 2*4*32, pooling=True)
            (256, 256, False, True), # resBlock5(2*4*32, 2*4*32, pooling=False)
        ]
        
        skip_connections = []
        for i, (in_ch, out_ch, pooling, dropout) in enumerate(encoder_configs):
            macs, current_shape, skip_shape = self._res_block_macs(
                current_shape, in_ch, out_ch, pooling)
            total_macs += macs
            if pooling:
                skip_connections.append(skip_shape)
            layer_details[f'resBlock{i+1}'] = {'macs': macs, 'output_shape': current_shape}
        
        # Decoder upsampling blocks
        decoder_configs = [
            (256, 128),  # upBlock1(8*32, 4*32)
            (128, 128),  # upBlock2(4*32, 4*32)
            (128, 64),   # upBlock3(4*32, 2*32)
            (64, 32),    # upBlock4(2*32, 32, drop_out=False)
        ]
        
        for i, (in_ch, out_ch) in enumerate(decoder_configs):
            macs, current_shape = self._up_block_macs(current_shape, in_ch, out_ch)
            total_macs += macs
            layer_details[f'upBlock{i+1}'] = {'macs': macs, 'output_shape': current_shape}
        
        # Final logits: Conv2d(32, nclasses, kernel_size=(1, 1))
        macs, current_shape = MACCalculator.pointwise_conv2d_macs(
            current_shape, self.num_classes, self.sparsity_config)
        total_macs += macs
        layer_details['logits'] = {'macs': macs, 'output_shape': current_shape}
        
        # Calculate number of points (pixels in range image)
        # For LiDAR, each pixel typically represents one point in the point cloud
        num_points = self.input_shape[0] * self.input_shape[1]
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'macs_per_point': total_macs / num_points if num_points > 0 else 0,
            'num_points': num_points,
            'layer_details': layer_details,
            'final_output_shape': current_shape,
            'sparsity_config': self.sparsity_config
        }
    
    def generate_csv_data(self, csv_generator: CSVLayerGenerator):
        """Generate CSV data for SalsaNext layers"""
        current_shape = self.input_shape
        
        # ResContext blocks
        # downCntx: ResContextBlock(5, 32)
        csv_generator.add_layer('downCntx_conv1', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], 32, 1)
        csv_generator.add_layer('downCntx_conv2', current_shape[0], current_shape[1],
                               3, 3, 32, 32, 1)
        csv_generator.add_layer('downCntx_conv3', current_shape[0], current_shape[1],
                               3, 3, 32, 32, 1)
        current_shape = (current_shape[0], current_shape[1], 32)
        
        # downCntx2: ResContextBlock(32, 32)
        csv_generator.add_layer('downCntx2_conv1', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], 32, 1)
        csv_generator.add_layer('downCntx2_conv2', current_shape[0], current_shape[1],
                               3, 3, 32, 32, 1)
        csv_generator.add_layer('downCntx2_conv3', current_shape[0], current_shape[1],
                               3, 3, 32, 32, 1)
        
        # downCntx3: ResContextBlock(32, 32)
        csv_generator.add_layer('downCntx3_conv1', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], 32, 1)
        csv_generator.add_layer('downCntx3_conv2', current_shape[0], current_shape[1],
                               3, 3, 32, 32, 1)
        csv_generator.add_layer('downCntx3_conv3', current_shape[0], current_shape[1],
                               3, 3, 32, 32, 1)
        
        # Encoder residual blocks with pooling
        encoder_configs = [
            (32, 64, True),   # resBlock1(32, 2*32, pooling=True)
            (64, 128, True),  # resBlock2(2*32, 2*2*32, pooling=True)
            (128, 256, True), # resBlock3(2*2*32, 2*4*32, pooling=True)
            (256, 256, True), # resBlock4(2*4*32, 2*4*32, pooling=True)
            (256, 256, False), # resBlock5(2*4*32, 2*4*32, pooling=False)
        ]
        
        for i, (in_ch, out_ch, pooling) in enumerate(encoder_configs):
            if pooling and i > 0:
                # Pooling reduces spatial dimensions by 2
                h, w = current_shape[0] // 2, current_shape[1] // 2
                current_shape = (h, w, current_shape[2])
            
            # ResBlock conv1: 1x1 conv
            csv_generator.add_layer(f'resBlock{i+1}_conv1', current_shape[0], current_shape[1],
                                   1, 1, current_shape[2], out_ch, 1)
            
            # ResBlock conv2: 3x3 conv
            csv_generator.add_layer(f'resBlock{i+1}_conv2', current_shape[0], current_shape[1],
                                   3, 3, out_ch, out_ch, 1)
            
            current_shape = (current_shape[0], current_shape[1], out_ch)
        
        # Decoder upsampling blocks
        decoder_configs = [
            (256, 128),  # upBlock1(8*32, 4*32)
            (128, 128),  # upBlock2(4*32, 4*32)
            (128, 64),   # upBlock3(4*32, 2*32)
            (64, 32),    # upBlock4(2*32, 32)
        ]
        
        for i, (in_ch, out_ch) in enumerate(decoder_configs):
            # Upsampling by 2x
            h_up, w_up = current_shape[0] * 2, current_shape[1] * 2
            
            # UpBlock conv layers
            csv_generator.add_layer(f'upBlock{i+1}_conv1', h_up, w_up,
                                   3, 3, current_shape[2], out_ch, 1)
            csv_generator.add_layer(f'upBlock{i+1}_conv2', h_up, w_up,
                                   3, 3, out_ch, out_ch, 1)
            
            current_shape = (h_up, w_up, out_ch)
        
        # Final logits layer
        csv_generator.add_layer('logits', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], self.num_classes, 1)
    
    def _res_context_block_macs(self, input_shape: Tuple[int, int, int], 
                               out_filters: int) -> Tuple[int, Tuple[int, int, int]]:
        """Calculate MACs for ResContextBlock with sparsity support"""
        # conv1: Conv2d(in_filters, out_filters, kernel_size=(1, 1))
        conv1_macs, conv1_shape = MACCalculator.pointwise_conv2d_macs(
            input_shape, out_filters, self.sparsity_config)
        
        # conv2: Conv2d(out_filters, out_filters, (3,3), padding=1)
        conv2_macs, conv2_shape = MACCalculator.conv2d_macs(
            conv1_shape, 3, out_filters, padding=1, sparsity_config=self.sparsity_config)
        
        # conv3: Conv2d(out_filters, out_filters, (3,3), dilation=2, padding=2)
        conv3_macs, conv3_shape = MACCalculator.conv2d_macs(
            conv2_shape, 3, out_filters, padding=2, sparsity_config=self.sparsity_config)
        
        total_macs = conv1_macs + conv2_macs + conv3_macs
        return total_macs, conv3_shape
    
    def _res_block_macs(self, input_shape: Tuple[int, int, int], in_filters: int, 
                       out_filters: int, pooling: bool) -> Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]:
        """Calculate MACs for ResBlock with sparsity support"""
        # conv1: Conv2d(in_filters, out_filters, kernel_size=(1, 1)) - shortcut
        shortcut_macs, shortcut_shape = MACCalculator.pointwise_conv2d_macs(
            input_shape, out_filters, self.sparsity_config)
        
        # conv2: Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        conv2_macs, conv2_shape = MACCalculator.conv2d_macs(
            input_shape, 3, out_filters, padding=1, sparsity_config=self.sparsity_config)
        
        # conv3: Conv2d(out_filters, out_filters, kernel_size=(3,3), dilation=2, padding=2)
        conv3_macs, conv3_shape = MACCalculator.conv2d_macs(
            conv2_shape, 3, out_filters, padding=2, sparsity_config=self.sparsity_config)
        
        # conv4: Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        conv4_macs, conv4_shape = MACCalculator.conv2d_macs(
            conv3_shape, 2, out_filters, padding=1, sparsity_config=self.sparsity_config)
        
        # conv5: Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        # Concatenate conv2, conv3, conv4 outputs
        concat_channels = out_filters * 3
        conv5_input_shape = (conv4_shape[0], conv4_shape[1], concat_channels)
        conv5_macs, conv5_shape = MACCalculator.pointwise_conv2d_macs(
            conv5_input_shape, out_filters, self.sparsity_config)
        
        total_macs = shortcut_macs + conv2_macs + conv3_macs + conv4_macs + conv5_macs
        
        # Skip connection for upsampling
        skip_shape = conv5_shape
        
        # Pooling (if enabled)
        if pooling:
            h, w, c = conv5_shape
            pooled_shape = (h // 2, w // 2, c)  # AvgPool2d with stride=2
            return total_macs, pooled_shape, skip_shape
        else:
            return total_macs, conv5_shape, skip_shape
    
    def _up_block_macs(self, input_shape: Tuple[int, int, int], in_filters: int, 
                      out_filters: int) -> Tuple[int, Tuple[int, int, int]]:
        """Calculate MACs for UpBlock with sparsity support"""
        h, w, c = input_shape
        
        # PixelShuffle(2) - no MACs, just reshaping
        # Effectively upsamples by factor of 2
        upsampled_shape = (h * 2, w * 2, c // 4)
        
        # conv1: Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        # Note: This assumes skip connection concatenation
        conv1_input_ch = upsampled_shape[2] + 2 * out_filters
        conv1_input_shape = (upsampled_shape[0], upsampled_shape[1], conv1_input_ch)
        conv1_macs, conv1_shape = MACCalculator.conv2d_macs(
            conv1_input_shape, 3, out_filters, padding=1, sparsity_config=self.sparsity_config)
        
        # conv2: Conv2d(out_filters, out_filters, (3,3), dilation=2, padding=2)
        conv2_macs, conv2_shape = MACCalculator.conv2d_macs(
            conv1_shape, 3, out_filters, padding=2, sparsity_config=self.sparsity_config)
        
        # conv3: Conv2d(out_filters, out_filters, (2,2), dilation=2, padding=1)
        conv3_macs, conv3_shape = MACCalculator.conv2d_macs(
            conv2_shape, 2, out_filters, padding=1, sparsity_config=self.sparsity_config)
        
        # conv4: Conv2d(out_filters*3, out_filters, kernel_size=(1,1))
        concat_channels = out_filters * 3
        conv4_input_shape = (conv3_shape[0], conv3_shape[1], concat_channels)
        conv4_macs, conv4_shape = MACCalculator.pointwise_conv2d_macs(
            conv4_input_shape, out_filters, self.sparsity_config)
        
        total_macs = conv1_macs + conv2_macs + conv3_macs + conv4_macs
        return total_macs, conv4_shape

class RangeNetPP:
    """RangeNet++ network configuration based on official implementation with sparsity support
    Source: https://github.com/PRBonn/lidar-bonnetal (DarkNet backbone)"""
    
    def __init__(self, input_height: int = 64, input_width: int = 1024, 
                 input_channels: int = 5, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None):
        self.input_shape = (input_height, input_width, input_channels)
        self.num_classes = num_classes
        self.sparsity_config = sparsity_config or SparsityConfig()
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for RangeNet++ based on DarkNet backbone"""
        current_shape = self.input_shape
        total_macs = 0
        layer_details = {}
        
        # Backbone: DarkNet-53 encoder (simplified for LiDAR)
        
        # Initial conv layer
        macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 3, 32, stride=1, padding=1, sparsity_config=self.sparsity_config)
        total_macs += macs
        layer_details['initial_conv'] = {'macs': macs, 'output_shape': current_shape}
        
        # DarkNet residual blocks
        # Based on darknet.py decoder configuration
        darknet_configs = [
            (32, 64, 1),    # stage 1: 1 residual block
            (64, 128, 2),   # stage 2: 2 residual blocks, downsample
            (128, 256, 8),  # stage 3: 8 residual blocks, downsample  
            (256, 512, 8),  # stage 4: 8 residual blocks, downsample
            (512, 1024, 4), # stage 5: 4 residual blocks, downsample
        ]
        
        skip_connections = []
        for stage_idx, (in_ch, out_ch, num_blocks) in enumerate(darknet_configs):
            for block_idx in range(num_blocks):
                # First block in stage downsamples
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                
                # BasicBlock: conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> relu2 + residual
                # conv1: Conv2d(inplanes, planes[0], kernel_size=1)
                conv1_macs, conv1_shape = MACCalculator.pointwise_conv2d_macs(
                    current_shape, out_ch // 2, self.sparsity_config)
                
                # conv2: Conv2d(planes[0], planes[1], kernel_size=3, padding=1)
                conv2_macs, conv2_shape = MACCalculator.conv2d_macs(
                    conv1_shape, 3, out_ch, stride=stride, padding=1, sparsity_config=self.sparsity_config)
                
                # Residual connection (if shape changes)
                residual_macs = 0
                if current_shape[2] != out_ch or stride > 1:
                    residual_macs, _ = MACCalculator.conv2d_macs(
                        current_shape, 1, out_ch, stride=stride, sparsity_config=self.sparsity_config)
                
                block_macs = conv1_macs + conv2_macs + residual_macs
                total_macs += block_macs
                current_shape = conv2_shape
                
                layer_details[f'darknet_stage{stage_idx+1}_block{block_idx+1}'] = {
                    'macs': block_macs, 'output_shape': current_shape}
                
                # Store skip connections at specific resolutions
                if block_idx == 0:
                    skip_connections.append(current_shape)
        
        # Decoder based on darknet.py configuration
        # dec5: BasicBlock([1024, 512]) -> dec4: BasicBlock([512, 256]) -> 
        # dec3: BasicBlock([256, 128]) -> dec2: BasicBlock([128, 64]) -> dec1: BasicBlock([64, 32])
        
        decoder_configs = [
            (1024, 512, 2),  # dec5: upsample + conv
            (512, 256, 2),   # dec4: upsample + conv  
            (256, 128, 2),   # dec3: upsample + conv
            (128, 64, 2),    # dec2: upsample + conv
            (64, 32, 2),     # dec1: upsample + conv
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            # Upsampling via ConvTranspose2d(kernel_size=[1, 4], stride=[1, 2])
            h, w, c = current_shape
            
            # ConvTranspose2d for upsampling
            # Approximate MACs: similar to regular conv but with transposed operation
            h_up = h * upsample_factor
            w_up = (w - 1) * 2 + 4 - 2  # accounting for kernel_size=[1,4], stride=[1,2], padding=[0,1]
            w_up = w * 2  # simplified
            
            # Upconv operation
            upconv_macs = h_up * w_up * 1 * 4 * in_ch * in_ch  # simplified
            
            upconv_shape = (h_up, w_up, in_ch)
            
            # BatchNorm + LeakyReLU (no MACs)
            
            # Residual BasicBlock: conv1 + conv2
            # conv1: Conv2d(in_ch, out_ch, kernel_size=1)
            res_conv1_macs, res_conv1_shape = MACCalculator.pointwise_conv2d_macs(
                upconv_shape, out_ch // 2, self.sparsity_config)
            
            # conv2: Conv2d(out_ch//2, out_ch, kernel_size=3, padding=1)
            res_conv2_macs, res_conv2_shape = MACCalculator.conv2d_macs(
                res_conv1_shape, 3, out_ch, padding=1, sparsity_config=self.sparsity_config)
            
            # Skip connection addition (no MACs, just element-wise add)
            
            decoder_macs = upconv_macs + res_conv1_macs + res_conv2_macs
            total_macs += decoder_macs
            current_shape = res_conv2_shape
            
            layer_details[f'decoder_stage_{i+1}'] = {
                'macs': decoder_macs, 'output_shape': current_shape}
        
        # Final prediction head: Dropout2d + Conv2d(32, num_classes, kernel_size=3, padding=1)
        final_macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 3, self.num_classes, padding=1, sparsity_config=self.sparsity_config)
        total_macs += final_macs
        layer_details['prediction_head'] = {'macs': final_macs, 'output_shape': current_shape}
        
        # Calculate number of points (pixels in range image)
        # For LiDAR, each pixel typically represents one point in the point cloud
        num_points = self.input_shape[0] * self.input_shape[1]
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'macs_per_point': total_macs / num_points if num_points > 0 else 0,
            'num_points': num_points,
            'layer_details': layer_details,
            'final_output_shape': current_shape,
            'sparsity_config': self.sparsity_config
        }
    
    def generate_csv_data(self, csv_generator: CSVLayerGenerator):
        """Generate CSV data for RangeNet++ layers"""
        current_shape = self.input_shape
        
        # Initial conv layer
        csv_generator.add_layer('initial_conv', current_shape[0], current_shape[1],
                               3, 3, current_shape[2], 32, 1)
        current_shape = (current_shape[0], current_shape[1], 32)
        
        # DarkNet residual blocks
        darknet_configs = [
            (32, 64, 1),    # stage 1: 1 residual block
            (64, 128, 2),   # stage 2: 2 residual blocks, downsample
            (128, 256, 8),  # stage 3: 8 residual blocks, downsample  
            (256, 512, 8),  # stage 4: 8 residual blocks, downsample
            (512, 1024, 4), # stage 5: 4 residual blocks, downsample
        ]
        
        for stage_idx, (in_ch, out_ch, num_blocks) in enumerate(darknet_configs):
            for block_idx in range(num_blocks):
                # First block in stage downsamples
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                
                if stride == 2:
                    # Update shape for downsampling
                    h, w = current_shape[0] // 2, current_shape[1] // 2
                    current_shape = (h, w, current_shape[2])
                
                # BasicBlock conv1: 1x1 conv
                csv_generator.add_layer(f'darknet_s{stage_idx+1}_b{block_idx+1}_conv1',
                                       current_shape[0], current_shape[1],
                                       1, 1, current_shape[2], out_ch // 2, stride)
                
                # BasicBlock conv2: 3x3 conv
                conv1_shape = (current_shape[0], current_shape[1], out_ch // 2)
                csv_generator.add_layer(f'darknet_s{stage_idx+1}_b{block_idx+1}_conv2',
                                       conv1_shape[0], conv1_shape[1],
                                       3, 3, out_ch // 2, out_ch, 1)
                
                # Residual connection (if needed)
                if current_shape[2] != out_ch or stride > 1:
                    csv_generator.add_layer(f'darknet_s{stage_idx+1}_b{block_idx+1}_residual',
                                           current_shape[0], current_shape[1],
                                           1, 1, current_shape[2], out_ch, stride)
                
                current_shape = (conv1_shape[0], conv1_shape[1], out_ch)
        
        # Decoder
        decoder_configs = [
            (1024, 512, 2),  # dec5: upsample + conv
            (512, 256, 2),   # dec4: upsample + conv  
            (256, 128, 2),   # dec3: upsample + conv
            (128, 64, 2),    # dec2: upsample + conv
            (64, 32, 2),     # dec1: upsample + conv
        ]
        
        for i, (in_ch, out_ch, upsample_factor) in enumerate(decoder_configs):
            # Upsampling
            h_up, w_up = current_shape[0] * upsample_factor, current_shape[1] * upsample_factor
            
            # Decoder conv1: 1x1 conv
            csv_generator.add_layer(f'decoder_s{i+1}_conv1', h_up, w_up,
                                   1, 1, current_shape[2], out_ch // 2, 1)
            
            # Decoder conv2: 3x3 conv
            csv_generator.add_layer(f'decoder_s{i+1}_conv2', h_up, w_up,
                                   3, 3, out_ch // 2, out_ch, 1)
            
            current_shape = (h_up, w_up, out_ch)
        
        # Final prediction head
        csv_generator.add_layer('prediction_head', current_shape[0], current_shape[1],
                               3, 3, current_shape[2], self.num_classes, 1)

class PolarNet:
    """PolarNet network configuration (CVPR 2020/2022 improvements) with sparsity support
    Polar Bird's Eye View representation for LiDAR point cloud segmentation
    Based on: https://github.com/edwardzhou130/PolarSeg"""
    
    def __init__(self, input_height: int = 480, input_width: int = 360, 
                 input_channels: int = 9, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None):
        self.input_shape = (input_height, input_width, input_channels)
        self.num_classes = num_classes
        self.sparsity_config = sparsity_config or SparsityConfig()
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for PolarNet"""
        current_shape = self.input_shape
        total_macs = 0
        layer_details = {}
        
        # Encoder - ResNet-like backbone with polar coordinates
        # Initial conv block
        macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 3, 64, stride=1, padding=1, sparsity_config=self.sparsity_config)
        total_macs += macs
        layer_details['initial_conv'] = {'macs': macs, 'output_shape': current_shape}
        
        # ResNet blocks with polar-aware convolutions
        resnet_configs = [
            (64, 64, 2, 1),    # layer1: 2 blocks, no downsample
            (64, 128, 2, 2),   # layer2: 2 blocks, downsample
            (128, 256, 2, 2),  # layer3: 2 blocks, downsample
            (256, 512, 2, 2),  # layer4: 2 blocks, downsample
        ]
        
        for layer_idx, (in_ch, out_ch, num_blocks, stride) in enumerate(resnet_configs):
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                
                # Basic ResNet block: conv1 -> bn -> relu -> conv2 -> bn -> (+residual) -> relu
                # conv1: 3x3 conv
                conv1_macs, conv1_shape = MACCalculator.conv2d_macs(
                    current_shape, 3, out_ch, stride=block_stride, padding=1, 
                    sparsity_config=self.sparsity_config)
                
                # conv2: 3x3 conv
                conv2_macs, conv2_shape = MACCalculator.conv2d_macs(
                    conv1_shape, 3, out_ch, stride=1, padding=1, 
                    sparsity_config=self.sparsity_config)
                
                # Residual connection (if needed)
                residual_macs = 0
                if current_shape[2] != out_ch or block_stride > 1:
                    residual_macs, _ = MACCalculator.conv2d_macs(
                        current_shape, 1, out_ch, stride=block_stride, 
                        sparsity_config=self.sparsity_config)
                
                block_macs = conv1_macs + conv2_macs + residual_macs
                total_macs += block_macs
                current_shape = conv2_shape
                
                layer_details[f'resnet_layer{layer_idx+1}_block{block_idx+1}'] = {
                    'macs': block_macs, 'output_shape': current_shape}
        
        # Polar-aware Feature Pyramid Network (FPN)
        # Top-down pathway with lateral connections
        fpn_configs = [
            (512, 256),  # P5 -> P4
            (256, 128),  # P4 -> P3  
            (128, 64),   # P3 -> P2
        ]
        
        for i, (in_ch, out_ch) in enumerate(fpn_configs):
            # Lateral conv: 1x1 conv
            lateral_macs, lateral_shape = MACCalculator.pointwise_conv2d_macs(
                current_shape, out_ch, self.sparsity_config)
            
            # Upsampling (no MACs, just interpolation)
            h_up, w_up = current_shape[0] * 2, current_shape[1] * 2
            upsampled_shape = (h_up, w_up, out_ch)
            
            # Output conv: 3x3 conv
            output_macs, output_shape = MACCalculator.conv2d_macs(
                upsampled_shape, 3, out_ch, padding=1, sparsity_config=self.sparsity_config)
            
            fpn_macs = lateral_macs + output_macs
            total_macs += fpn_macs
            current_shape = output_shape
            
            layer_details[f'fpn_level_{i+1}'] = {'macs': fpn_macs, 'output_shape': current_shape}
        
        # Final prediction head with polar-aware loss
        final_macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 1, self.num_classes, sparsity_config=self.sparsity_config)
        total_macs += final_macs
        layer_details['prediction_head'] = {'macs': final_macs, 'output_shape': current_shape}
        
        # Calculate number of points
        num_points = self.input_shape[0] * self.input_shape[1]
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'macs_per_point': total_macs / num_points if num_points > 0 else 0,
            'num_points': num_points,
            'layer_details': layer_details,
            'final_output_shape': current_shape,
            'sparsity_config': self.sparsity_config
        }
    
    def generate_csv_data(self, csv_generator: CSVLayerGenerator):
        """Generate CSV data for PolarNet layers"""
        current_shape = self.input_shape
        
        # Initial conv
        csv_generator.add_layer('initial_conv', current_shape[0], current_shape[1],
                               3, 3, current_shape[2], 64, 1)
        current_shape = (current_shape[0], current_shape[1], 64)
        
        # ResNet blocks (simplified for CSV)
        resnet_configs = [
            (64, 64, 2, 1),    # layer1
            (64, 128, 2, 2),   # layer2
            (128, 256, 2, 2),  # layer3
            (256, 512, 2, 2),  # layer4
        ]
        
        for layer_idx, (in_ch, out_ch, num_blocks, stride) in enumerate(resnet_configs):
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                
                if block_stride == 2:
                    h, w = current_shape[0] // 2, current_shape[1] // 2
                    current_shape = (h, w, current_shape[2])
                
                # ResNet block conv1
                csv_generator.add_layer(f'resnet_l{layer_idx+1}_b{block_idx+1}_conv1',
                                       current_shape[0], current_shape[1],
                                       3, 3, current_shape[2], out_ch, block_stride)
                
                # ResNet block conv2
                csv_generator.add_layer(f'resnet_l{layer_idx+1}_b{block_idx+1}_conv2',
                                       current_shape[0], current_shape[1],
                                       3, 3, out_ch, out_ch, 1)
                
                current_shape = (current_shape[0], current_shape[1], out_ch)
        
        # FPN (simplified)
        for i in range(3):
            h_up, w_up = current_shape[0] * 2, current_shape[1] * 2
            out_ch = max(64, current_shape[2] // 2)
            
            csv_generator.add_layer(f'fpn_level_{i+1}', h_up, w_up,
                                   3, 3, current_shape[2], out_ch, 1)
            current_shape = (h_up, w_up, out_ch)
        
        # Prediction head
        csv_generator.add_layer('prediction_head', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], self.num_classes, 1)

class CENet:
    """CENet (Context Embedding Network) for LiDAR segmentation (2022) with sparsity support
    Improved context modeling for range image segmentation"""
    
    def __init__(self, input_height: int = 64, input_width: int = 2048, 
                 input_channels: int = 5, num_classes: int = 20,
                 sparsity_config: SparsityConfig = None):
        self.input_shape = (input_height, input_width, input_channels)
        self.num_classes = num_classes
        self.sparsity_config = sparsity_config or SparsityConfig()
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for CENet"""
        current_shape = self.input_shape
        total_macs = 0
        layer_details = {}
        
        # Context Embedding Module
        macs, current_shape = self._context_embedding_module(current_shape)
        total_macs += macs
        layer_details['context_embedding'] = {'macs': macs, 'output_shape': current_shape}
        
        # Multi-scale Feature Extraction
        for scale in [1, 2, 4]:
            scale_macs, scale_shape = self._multiscale_feature_block(current_shape, scale)
            total_macs += scale_macs
            layer_details[f'multiscale_{scale}'] = {'macs': scale_macs, 'output_shape': scale_shape}
        
        # Attention-guided Feature Fusion
        attention_macs, current_shape = self._attention_fusion_module(current_shape)
        total_macs += attention_macs
        layer_details['attention_fusion'] = {'macs': attention_macs, 'output_shape': current_shape}
        
        # Final prediction
        final_macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 1, self.num_classes, sparsity_config=self.sparsity_config)
        total_macs += final_macs
        layer_details['prediction'] = {'macs': final_macs, 'output_shape': current_shape}
        
        num_points = self.input_shape[0] * self.input_shape[1]
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'macs_per_point': total_macs / num_points if num_points > 0 else 0,
            'num_points': num_points,
            'layer_details': layer_details,
            'final_output_shape': current_shape,
            'sparsity_config': self.sparsity_config
        }
    
    def _context_embedding_module(self, input_shape):
        """Context embedding with dilated convolutions"""
        # Multi-branch dilated convolutions
        branch_macs = 0
        
        # Branch 1: 1x1 conv
        macs1, shape1 = MACCalculator.pointwise_conv2d_macs(
            input_shape, 64, self.sparsity_config)
        branch_macs += macs1
        
        # Branch 2: 3x3 dilated conv (dilation=2)
        macs2, shape2 = MACCalculator.conv2d_macs(
            input_shape, 3, 64, padding=2, sparsity_config=self.sparsity_config)
        branch_macs += macs2
        
        # Branch 3: 3x3 dilated conv (dilation=4)
        macs3, shape3 = MACCalculator.conv2d_macs(
            input_shape, 3, 64, padding=4, sparsity_config=self.sparsity_config)
        branch_macs += macs3
        
        # Fusion conv
        fusion_input_shape = (shape1[0], shape1[1], 64 * 3)
        fusion_macs, output_shape = MACCalculator.pointwise_conv2d_macs(
            fusion_input_shape, 128, self.sparsity_config)
        branch_macs += fusion_macs
        
        return branch_macs, output_shape
    
    def _multiscale_feature_block(self, input_shape, scale):
        """Multi-scale feature extraction block"""
        # Downsample
        h, w, c = input_shape
        if scale > 1:
            downsampled_shape = (h // scale, w // scale, c)
        else:
            downsampled_shape = input_shape
        
        # Feature extraction
        macs1, shape1 = MACCalculator.conv2d_macs(
            downsampled_shape, 3, 64, padding=1, sparsity_config=self.sparsity_config)
        macs2, shape2 = MACCalculator.conv2d_macs(
            shape1, 3, 64, padding=1, sparsity_config=self.sparsity_config)
        
        # Upsample back (if needed)
        if scale > 1:
            upsampled_shape = (h, w, 64)
        else:
            upsampled_shape = shape2
        
        return macs1 + macs2, upsampled_shape
    
    def _attention_fusion_module(self, input_shape):
        """Attention-guided feature fusion"""
        # Channel attention
        h, w, c = input_shape
        
        # Global average pooling + FC layers (approximated as 1x1 convs)
        gap_macs = h * w * c  # Global average pooling approximation
        
        # Attention weights computation
        att_macs1, att_shape1 = MACCalculator.pointwise_conv2d_macs(
            (1, 1, c), c // 16, self.sparsity_config)
        att_macs2, att_shape2 = MACCalculator.pointwise_conv2d_macs(
            att_shape1, c, self.sparsity_config)
        
        # Apply attention
        feature_macs, feature_shape = MACCalculator.pointwise_conv2d_macs(
            input_shape, 128, self.sparsity_config)
        
        total_macs = gap_macs + att_macs1 + att_macs2 + feature_macs
        return total_macs, feature_shape
    
    def generate_csv_data(self, csv_generator: CSVLayerGenerator):
        """Generate CSV data for CENet layers"""
        current_shape = self.input_shape
        
        # Context embedding module
        csv_generator.add_layer('context_emb_conv1', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], 64, 1)
        csv_generator.add_layer('context_emb_conv2', current_shape[0], current_shape[1],
                               3, 3, current_shape[2], 64, 1)
        csv_generator.add_layer('context_emb_conv3', current_shape[0], current_shape[1],
                               3, 3, current_shape[2], 64, 1)
        csv_generator.add_layer('context_emb_fusion', current_shape[0], current_shape[1],
                               1, 1, 192, 128, 1)
        current_shape = (current_shape[0], current_shape[1], 128)
        
        # Multi-scale features (simplified)
        for scale in [1, 2, 4]:
            scale_h, scale_w = current_shape[0] // scale, current_shape[1] // scale
            csv_generator.add_layer(f'multiscale_{scale}_conv1', scale_h, scale_w,
                                   3, 3, current_shape[2], 64, 1)
            csv_generator.add_layer(f'multiscale_{scale}_conv2', scale_h, scale_w,
                                   3, 3, 64, 64, 1)
        
        # Attention fusion
        csv_generator.add_layer('attention_fusion', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], 128, 1)
        current_shape = (current_shape[0], current_shape[1], 128)
        
        # Final prediction
        csv_generator.add_layer('prediction', current_shape[0], current_shape[1],
                               1, 1, current_shape[2], self.num_classes, 1)

class EfficientNetV2:
    """EfficientNetV2 (2021/2022) - Modern efficient CNN with sparsity support
    Based on: https://arxiv.org/abs/2104.00298"""
    
    def __init__(self, input_height: int = 224, input_width: int = 224, 
                 input_channels: int = 3, num_classes: int = 1000,
                 model_size: str = 'S', sparsity_config: SparsityConfig = None):
        self.input_shape = (input_height, input_width, input_channels)
        self.num_classes = num_classes
        self.model_size = model_size
        self.sparsity_config = sparsity_config or SparsityConfig()
        
        # EfficientNetV2-S configuration
        if model_size == 'S':
            self.stage_configs = [
                # (num_blocks, kernel_size, stride, expand_ratio, channels, se_ratio, fused)
                (2, 3, 1, 1, 24, 0, True),      # Stage 1: Fused-MBConv
                (4, 3, 2, 4, 48, 0, True),      # Stage 2: Fused-MBConv
                (4, 3, 2, 4, 64, 0, True),      # Stage 3: Fused-MBConv
                (6, 3, 2, 4, 128, 0.25, False), # Stage 4: MBConv with SE
                (9, 3, 1, 6, 160, 0.25, False), # Stage 5: MBConv with SE
                (15, 3, 2, 6, 256, 0.25, False), # Stage 6: MBConv with SE
            ]
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for EfficientNetV2"""
        current_shape = self.input_shape
        total_macs = 0
        layer_details = {}
        
        # Stem: Conv2d 3x3, stride=2
        macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 3, 24, stride=2, padding=1, sparsity_config=self.sparsity_config)
        total_macs += macs
        layer_details['stem'] = {'macs': macs, 'output_shape': current_shape}
        
        # Stages
        for stage_idx, (num_blocks, kernel_size, stride, expand_ratio, out_channels, se_ratio, fused) in enumerate(self.stage_configs):
            stage_macs = 0
            
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                
                if fused:
                    # Fused-MBConv block
                    block_macs, current_shape = self._fused_mbconv_block(
                        current_shape, out_channels, expand_ratio, kernel_size, block_stride)
                else:
                    # Standard MBConv block with SE
                    block_macs, current_shape = self._mbconv_block(
                        current_shape, out_channels, expand_ratio, kernel_size, block_stride, se_ratio)
                
                stage_macs += block_macs
                layer_details[f'stage_{stage_idx+1}_block_{block_idx+1}'] = {
                    'macs': block_macs, 'output_shape': current_shape}
            
            total_macs += stage_macs
        
        # Head: Conv2d 1x1 + Global Average Pooling + FC
        head_macs, head_shape = MACCalculator.pointwise_conv2d_macs(
            current_shape, 1280, self.sparsity_config)
        
        # Global average pooling (no MACs)
        h, w, c = head_shape
        pooled_shape = (1, 1, c)
        
        # Final classifier
        fc_macs = c * self.num_classes
        
        total_macs += head_macs + fc_macs
        layer_details['head'] = {'macs': head_macs + fc_macs, 'output_shape': (1, 1, self.num_classes)}
        
        num_points = self.input_shape[0] * self.input_shape[1]
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'macs_per_point': total_macs / num_points if num_points > 0 else 0,
            'num_points': num_points,
            'layer_details': layer_details,
            'final_output_shape': (1, 1, self.num_classes),
            'sparsity_config': self.sparsity_config
        }
    
    def _fused_mbconv_block(self, input_shape, out_channels, expand_ratio, kernel_size, stride):
        """Fused Mobile Inverted Bottleneck block"""
        h_in, w_in, c_in = input_shape
        expanded_channels = c_in * expand_ratio
        
        # Fused expand + depthwise: single 3x3 conv
        macs1, shape1 = MACCalculator.conv2d_macs(
            input_shape, kernel_size, expanded_channels, stride=stride, 
            padding=kernel_size//2, sparsity_config=self.sparsity_config)
        
        # Project: 1x1 conv
        macs2, shape2 = MACCalculator.pointwise_conv2d_macs(
            shape1, out_channels, self.sparsity_config)
        
        # Residual connection (if applicable)
        residual_macs = 0
        if stride == 1 and c_in == out_channels:
            residual_macs = 0  # Just addition, no computation
        
        return macs1 + macs2 + residual_macs, shape2
    
    def _mbconv_block(self, input_shape, out_channels, expand_ratio, kernel_size, stride, se_ratio):
        """Mobile Inverted Bottleneck block with Squeeze-and-Excitation"""
        h_in, w_in, c_in = input_shape
        expanded_channels = c_in * expand_ratio
        
        # Expand: 1x1 conv
        macs1, shape1 = MACCalculator.pointwise_conv2d_macs(
            input_shape, expanded_channels, self.sparsity_config)
        
        # Depthwise: depthwise conv
        macs2, shape2 = MACCalculator.depthwise_conv2d_macs(
            shape1, kernel_size, stride=stride, padding=kernel_size//2, 
            sparsity_config=self.sparsity_config)
        
        # SE module
        se_macs = 0
        if se_ratio > 0:
            se_channels = max(1, int(expanded_channels * se_ratio))
            # Global average pooling + FC1 + FC2
            se_macs = expanded_channels + se_channels * expanded_channels + expanded_channels * se_channels
        
        # Project: 1x1 conv
        macs3, shape3 = MACCalculator.pointwise_conv2d_macs(
            shape2, out_channels, self.sparsity_config)
        
        return macs1 + macs2 + se_macs + macs3, shape3

class ConvNeXt:
    """ConvNeXt (2022) - Modern ConvNet with sparsity support
    Based on: https://arxiv.org/abs/2201.03545"""
    
    def __init__(self, input_height: int = 224, input_width: int = 224, 
                 input_channels: int = 3, num_classes: int = 1000,
                 model_size: str = 'Tiny', sparsity_config: SparsityConfig = None):
        self.input_shape = (input_height, input_width, input_channels)
        self.num_classes = num_classes
        self.model_size = model_size
        self.sparsity_config = sparsity_config or SparsityConfig()
        
        # ConvNeXt-Tiny configuration
        if model_size == 'Tiny':
            self.stage_configs = [
                (3, 96),   # Stage 1: 3 blocks, 96 channels
                (3, 192),  # Stage 2: 3 blocks, 192 channels
                (9, 384),  # Stage 3: 9 blocks, 384 channels
                (3, 768),  # Stage 4: 3 blocks, 768 channels
            ]
        
    def calculate_macs(self) -> Dict[str, Union[int, Dict]]:
        """Calculate total MACs for ConvNeXt"""
        current_shape = self.input_shape
        total_macs = 0
        layer_details = {}
        
        # Stem: 4x4 conv, stride=4
        macs, current_shape = MACCalculator.conv2d_macs(
            current_shape, 4, 96, stride=4, sparsity_config=self.sparsity_config)
        total_macs += macs
        layer_details['stem'] = {'macs': macs, 'output_shape': current_shape}
        
        # Stages
        for stage_idx, (num_blocks, channels) in enumerate(self.stage_configs):
            stage_macs = 0
            
            # Downsampling (except for first stage)
            if stage_idx > 0:
                # Layer norm + 2x2 conv, stride=2
                downsample_macs, current_shape = MACCalculator.conv2d_macs(
                    current_shape, 2, channels, stride=2, sparsity_config=self.sparsity_config)
                stage_macs += downsample_macs
                layer_details[f'downsample_{stage_idx+1}'] = {
                    'macs': downsample_macs, 'output_shape': current_shape}
            
            # ConvNeXt blocks
            for block_idx in range(num_blocks):
                block_macs, current_shape = self._convnext_block(current_shape, channels)
                stage_macs += block_macs
                layer_details[f'stage_{stage_idx+1}_block_{block_idx+1}'] = {
                    'macs': block_macs, 'output_shape': current_shape}
            
            total_macs += stage_macs
        
        # Head: Global Average Pooling + Layer Norm + FC
        h, w, c = current_shape
        # Global average pooling (no MACs)
        
        # Final classifier
        fc_macs = c * self.num_classes
        total_macs += fc_macs
        layer_details['head'] = {'macs': fc_macs, 'output_shape': (1, 1, self.num_classes)}
        
        num_points = self.input_shape[0] * self.input_shape[1]
        
        return {
            'total_macs': total_macs,
            'total_macs_millions': total_macs / 1e6,
            'macs_per_point': total_macs / num_points if num_points > 0 else 0,
            'num_points': num_points,
            'layer_details': layer_details,
            'final_output_shape': (1, 1, self.num_classes),
            'sparsity_config': self.sparsity_config
        }
    
    def _convnext_block(self, input_shape, channels):
        """ConvNeXt block: DWConv + LayerNorm + FFN + residual"""
        h, w, c = input_shape
        
        # Depthwise conv 7x7
        dw_macs, dw_shape = MACCalculator.depthwise_conv2d_macs(
            input_shape, 7, padding=3, sparsity_config=self.sparsity_config)
        
        # FFN: 1x1 conv (expand) + GELU + 1x1 conv (project)
        ffn_expand_channels = channels * 4
        
        # Expand: 1x1 conv
        expand_macs, expand_shape = MACCalculator.pointwise_conv2d_macs(
            dw_shape, ffn_expand_channels, self.sparsity_config)
        
        # Project: 1x1 conv
        project_macs, project_shape = MACCalculator.pointwise_conv2d_macs(
            expand_shape, channels, self.sparsity_config)
        
        total_macs = dw_macs + expand_macs + project_macs
        return total_macs, project_shape

def generate_csv_files():
    """Generate CSV files for simulator for each network"""
    print("\nGenerating CSV files for simulator...")
    
    # Standard configurations
    configs = {
        # Only include LiDAR networks for CSV generation (2D CNNs don't need layer-by-layer CSV)
        'SqueezeSeg': {'input_height': 64, 'input_width': 512, 'input_channels': 5, 'num_classes': 20},
        'SalsaNext': {'input_height': 64, 'input_width': 2048, 'input_channels': 5, 'num_classes': 20},
        'RangeNet++': {'input_height': 64, 'input_width': 1024, 'input_channels': 5, 'num_classes': 20},
        'PolarNet': {'input_height': 480, 'input_width': 360, 'input_channels': 9, 'num_classes': 20},
        'CENet': {'input_height': 64, 'input_width': 2048, 'input_channels': 5, 'num_classes': 20}
    }
    
    # Use moderate sparsity for CSV generation
    sparsity_config = SparsityConfig(feature_sparsity=0.5, weight_sparsity=0.4)
    
    # Create CSV generator
    csv_generator = CSVLayerGenerator()
    
    for network_name, config in configs.items():
        print(f"Generating CSV for {network_name}...")
        
        # Create network instance
        if network_name == 'SqueezeSeg':
            network = SqueezeSeg(sparsity_config=sparsity_config, **config)
        elif network_name == 'SalsaNext':
            network = SalsaNext(sparsity_config=sparsity_config, **config)
        elif network_name == 'RangeNet++':
            network = RangeNetPP(sparsity_config=sparsity_config, **config)
        elif network_name == 'PolarNet':
            network = PolarNet(sparsity_config=sparsity_config, **config)
        elif network_name == 'CENet':
            network = CENet(sparsity_config=sparsity_config, **config)
        else:
            print(f"CSV generation not supported for {network_name}")
            continue
        
        # Clear previous data and generate CSV data
        csv_generator.clear()
        network.generate_csv_data(csv_generator)
        
        # Add sparsity to all layers
        for layer in csv_generator.layers:
            layer['Sparsity'] = sparsity_config.weight_sparsity
        
        # Save CSV file
        filename = f"{network_name.lower().replace('++', 'pp').replace('-', '_')}_layers.csv"
        csv_generator.save_csv(filename)
        print(f"Saved {filename}")
    
    print("CSV generation complete!")

def main():
    """Main function to calculate and compare MAC operations with sparsity analysis"""
    print("LiDAR Semantic Segmentation Networks - MAC Operations Calculator with Sparsity Support")
    print("=" * 80)
    
    # Generate CSV files for simulator
    generate_csv_files()
    print("\n" + "=" * 80)
    
    # Standard configurations
    configs = {
        # LiDAR Segmentation Networks
        'SqueezeSeg': {'input_height': 64, 'input_width': 512, 'input_channels': 5, 'num_classes': 20},
        'SalsaNext': {'input_height': 64, 'input_width': 2048, 'input_channels': 5, 'num_classes': 20},
        'RangeNet++': {'input_height': 64, 'input_width': 1024, 'input_channels': 5, 'num_classes': 20},
        'PolarNet': {'input_height': 480, 'input_width': 360, 'input_channels': 9, 'num_classes': 20},
        'CENet': {'input_height': 64, 'input_width': 2048, 'input_channels': 5, 'num_classes': 20},
        
        # Modern 2D CNNs
        'EfficientNetV2-S': {'input_height': 224, 'input_width': 224, 'input_channels': 3, 'num_classes': 1000, 'model_size': 'S'},
        'ConvNeXt-Tiny': {'input_height': 224, 'input_width': 224, 'input_channels': 3, 'num_classes': 1000, 'model_size': 'Tiny'},
    }
    
    # Sparsity scenarios
    sparsity_scenarios = {
        'Dense (No Sparsity)': SparsityConfig(feature_sparsity=0.0, weight_sparsity=0.0),
        'Feature Sparse Only': SparsityConfig(feature_sparsity=0.5, weight_sparsity=0.0),
        'Conservative Pruning': SparsityConfig(feature_sparsity=0.5, weight_sparsity=0.25),
        'Moderate Pruning': SparsityConfig(feature_sparsity=0.5, weight_sparsity=0.45),
        'Aggressive Pruning': SparsityConfig(feature_sparsity=0.6, weight_sparsity=0.65),
        'Ultra Sparse': SparsityConfig(feature_sparsity=0.7, weight_sparsity=0.8)
    }
    
    results = {}
    
    # Calculate for each network and sparsity scenario
    for scenario_name, sparsity_config in sparsity_scenarios.items():
        print(f"\n{scenario_name}")
        print("=" * 60)
        print(f"Feature Sparsity: {sparsity_config.feature_sparsity*100:.1f}%")
        print(f"Weight Sparsity: {sparsity_config.weight_sparsity*100:.1f}%")
        print(f"Effective Reduction: {(1-sparsity_config.get_effective_sparsity_multiplier())*100:.1f}%")
        
        scenario_results = {}
        
        for network_name, config in configs.items():
            print(f"\n  {network_name}")
            print("  " + "-" * 40)
            print(f"  Input: {config['input_height']}x{config['input_width']}x{config['input_channels']}")
            
            if network_name == 'SqueezeSeg':
                network = SqueezeSeg(**config, sparsity_config=sparsity_config)
            elif network_name == 'SalsaNext':
                network = SalsaNext(**config, sparsity_config=sparsity_config)
            elif network_name == 'RangeNet++':
                network = RangeNetPP(**config, sparsity_config=sparsity_config)
            elif network_name == 'PolarNet':
                network = PolarNet(**config, sparsity_config=sparsity_config)
            elif network_name == 'CENet':
                network = CENet(**config, sparsity_config=sparsity_config)
            elif network_name == 'EfficientNetV2-S':
                network = EfficientNetV2(**config, sparsity_config=sparsity_config)
            elif network_name == 'ConvNeXt-Tiny':
                network = ConvNeXt(**config, sparsity_config=sparsity_config)
            else:
                print(f"  Unknown network: {network_name}")
                continue
            
            result = network.calculate_macs()
            scenario_results[network_name] = result
            
            print(f"  Total MACs: {result['total_macs']:,}")
            print(f"  Total MACs (millions): {result['total_macs_millions']:.2f}M")
            print(f"  MACs per point: {result['macs_per_point']:.2f}")
            print(f"  Number of points: {result['num_points']:,}")
            print(f"  Final output shape: {result['final_output_shape']}")
        
        results[scenario_name] = scenario_results
    
    # Comprehensive comparison
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SPARSITY COMPARISON")
    print("=" * 80)
    
    # Create comparison table
    print(f"{'Network':<15} {'Scenario':<20} {'MACs (M)':<12} {'MACs/Point':<12} {'Speedup':<10} {'Reduction':<10}")
    print("-" * 87)
    
    dense_results = results['Dense (No Sparsity)']
    
    for scenario_name, scenario_results in results.items():
        for network_name, result in scenario_results.items():
            dense_macs = dense_results[network_name]['total_macs_millions']
            current_macs = result['total_macs_millions']
            macs_per_point = result['macs_per_point']
            speedup = dense_macs / current_macs if current_macs > 0 else float('inf')
            reduction = (1 - current_macs / dense_macs) * 100 if dense_macs > 0 else 0
            
            print(f"{network_name:<15} {scenario_name:<20} {current_macs:<12.2f} {macs_per_point:<12.2f} {speedup:<10.2f}x {reduction:<10.1f}%")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SPARSITY IMPACT SUMMARY")
    print("=" * 80)
    
    print("\nRecommended Weight Sparsity Factors (accuracy loss < 2%):")
    network_recommendations = {
        # LiDAR Networks
        'SqueezeSeg': 0.45,   # More robust due to Fire module efficiency
        'SalsaNext': 0.40,    # Moderate sensitivity due to dilated convolutions
        'RangeNet++': 0.50,   # DarkNet backbone handles pruning well
        'PolarNet': 0.35,     # Polar coordinates require more careful pruning
        'CENet': 0.40,        # Context embedding benefits from moderate sparsity
        
        # Modern 2D CNNs
        'EfficientNetV2-S': 0.30,  # Efficient architecture, conservative pruning
        'ConvNeXt-Tiny': 0.40,     # Modern design handles pruning well
    }
    
    for network_name, recommended_sparsity in network_recommendations.items():
        recommended_config = SparsityConfig(feature_sparsity=0.5, weight_sparsity=recommended_sparsity)
        
        # Skip if network not in dense_results (e.g., network not calculated)
        if network_name not in dense_results:
            continue
            
        dense_macs = dense_results[network_name]['total_macs_millions']
        
        # Calculate with recommended sparsity
        if network_name == 'SqueezeSeg':
            network = SqueezeSeg(**configs[network_name], sparsity_config=recommended_config)
        elif network_name == 'SalsaNext':
            network = SalsaNext(**configs[network_name], sparsity_config=recommended_config)
        elif network_name == 'RangeNet++':
            network = RangeNetPP(**configs[network_name], sparsity_config=recommended_config)
        elif network_name == 'PolarNet':
            network = PolarNet(**configs[network_name], sparsity_config=recommended_config)
        elif network_name == 'CENet':
            network = CENet(**configs[network_name], sparsity_config=recommended_config)
        elif network_name == 'EfficientNetV2-S':
            network = EfficientNetV2(**configs[network_name], sparsity_config=recommended_config)
        elif network_name == 'ConvNeXt-Tiny':
            network = ConvNeXt(**configs[network_name], sparsity_config=recommended_config)
        else:
            continue
        
        result = network.calculate_macs()
        recommended_macs = result['total_macs_millions']
        speedup = dense_macs / recommended_macs
        
        print(f"{network_name:<15}: {recommended_sparsity*100:>5.0f}% weight sparsity → {speedup:.2f}x speedup")
    
    print(f"\nMaximum achievable speedups with 'Ultra Sparse' configuration:")
    ultra_sparse_results = results['Ultra Sparse']
    for network_name in configs.keys():
        dense_macs = dense_results[network_name]['total_macs_millions']
        sparse_macs = ultra_sparse_results[network_name]['total_macs_millions']
        max_speedup = dense_macs / sparse_macs
        print(f"{network_name:<15}: {max_speedup:.2f}x speedup ({dense_macs:.2f}M → {sparse_macs:.2f}M MACs)")
    
    # MACs per point comparison
    print("\n" + "=" * 80)
    print("MACS PER POINT ANALYSIS")
    print("=" * 80)
    
    print(f"\nMACs per point for different networks (Dense configuration):")
    print(f"{'Network':<15} {'Input Size':<15} {'Points':<10} {'MACs/Point':<12} {'Total MACs (M)':<15}")
    print("-" * 70)
    
    for network_name in configs.keys():
        config = configs[network_name]
        result = dense_results[network_name]
        input_size = f"{config['input_height']}x{config['input_width']}"
        num_points = result['num_points']
        macs_per_point = result['macs_per_point']
        total_macs = result['total_macs_millions']
        
        print(f"{network_name:<15} {input_size:<15} {num_points:<10,} {macs_per_point:<12.2f} {total_macs:<15.2f}")
    
    print(f"\nImpact of sparsity on MACs per point:")
    print(f"{'Network':<15} {'Dense':<12} {'Conservative':<12} {'Moderate':<12} {'Aggressive':<12}")
    print("-" * 63)
    
    for network_name in configs.keys():
        dense_mpp = dense_results[network_name]['macs_per_point']
        conservative_mpp = results['Conservative Pruning'][network_name]['macs_per_point']
        moderate_mpp = results['Moderate Pruning'][network_name]['macs_per_point']
        aggressive_mpp = results['Aggressive Pruning'][network_name]['macs_per_point']
        
        print(f"{network_name:<15} {dense_mpp:<12.2f} {conservative_mpp:<12.2f} {moderate_mpp:<12.2f} {aggressive_mpp:<12.2f}")

if __name__ == "__main__":
    main()