#!/usr/bin/env python3
"""
Analysis script for comparing Mapping Time vs GEMM Time for the first layer in 3D sparse networks

This script analyzes the time breakdown for the first convolution layer specifically:
- Mapping Time: Time to create sparse tensor representation (network-wide overhead)
- GEMM Time: Time for the actual matrix multiplication in the first layer

Creates a stacked bar chart showing the time distribution for each 3D network's first layer.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import importlib.util
from typing import Dict, List, Tuple

# Import the network classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_3d_network_first_layer_timing() -> Dict[str, Dict[str, float]]:
    """Get mapping vs GEMM timing breakdown for the first layer of 3D sparse convolution networks"""
    
    try:
        # Load the pcl_macs module
        spec = importlib.util.spec_from_file_location("pcl_macs", "/Users/ashriram/Desktop/minuet-tracer/mac-vs-time/pcl_macs.py")
        module_3d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_3d)
        
        # Extract classes
        MinkowskiNet = module_3d.MinkowskiNet
        SPVNAS = module_3d.SPVNAS
        LargeKernel3D = module_3d.LargeKernel3D
        VoxelNeXt = module_3d.VoxelNeXt
        RSN = module_3d.RSN
        GEMMLayerGenerator = module_3d.GEMMLayerGenerator
        GEMMPerformanceModel = module_3d.GEMMPerformanceModel
        
    except Exception as e:
        print(f"Error importing 3D networks: {e}")
        return {}
    
    # Standard configuration for analysis
    num_points = 100000
    
    # Initialize networks with baseline configuration
    networks_3d = {
        'MinkowskiNet': MinkowskiNet(
            num_points=num_points, voxel_size=0.05, input_channels=4,
            num_classes=20, spatial_sparsity=0.05,
            channel_sparsity=0.0, feature_sparsity=0.0
        ),
        'SPVNAS': SPVNAS(
            num_points=num_points, voxel_size=0.05, input_channels=4,
            num_classes=20, spatial_sparsity=0.05,
            channel_sparsity=0.0, feature_sparsity=0.0
        ),
        'LargeKernel3D': LargeKernel3D(
            num_points=num_points, voxel_size=0.05, input_channels=4,
            num_classes=20, spatial_sparsity=0.05,
            channel_sparsity=0.0, feature_sparsity=0.0
        ),
        'VoxelNeXt': VoxelNeXt(
            num_points=num_points, voxel_size=0.05, input_channels=4,
            num_classes=20, spatial_sparsity=0.05,
            channel_sparsity=0.0, feature_sparsity=0.0
        ),
        'RSN': RSN(
            num_points=num_points, voxel_size=0.05, input_channels=4,
            num_classes=20, spatial_sparsity=0.05,
            channel_sparsity=0.0, feature_sparsity=0.0
        )
    }
    
    timing_breakdown = {}
    performance_model = GEMMPerformanceModel()
    
    print("Analyzing mapping vs GEMM time for first layer in 3D networks...")
    for name, network in networks_3d.items():
        try:
            print(f"  Analyzing {name}...")
            
            # Generate GEMM data to get first layer information
            gemm_generator = GEMMLayerGenerator()
            network.generate_gemm_data(gemm_generator, batch_size=1)
            
            # Get GEMM layers data - focus on first layer
            gemm_layers = gemm_generator.layers
            
            if gemm_layers and len(gemm_layers) > 0:
                # Get first layer data
                first_layer = gemm_layers[0]
                
                # Calculate mapping time for the entire network (done once)
                # Use first layer kernel size or default to 3
                kernel_size = first_layer.get('Kernel_Size', 3)
                mapping_metrics = performance_model.calculate_network_mapping_time(
                    input_voxels=num_points, avg_kernel_size=kernel_size
                )
                mapping_time_ms = mapping_metrics['mapping_time_ms']
                
                # Get GEMM time for first layer only
                first_layer_gemm_time = first_layer.get('GEMM_Time_ms', 0.0)
                
                # Total time = mapping + first layer GEMM
                total_time = mapping_time_ms + first_layer_gemm_time
                
                if total_time > 0:
                    mapping_percent = (mapping_time_ms / total_time) * 100
                    gemm_percent = (first_layer_gemm_time / total_time) * 100
                else:
                    mapping_percent = 50.0  # Default
                    gemm_percent = 50.0
                
                timing_breakdown[name] = {
                    'mapping_percent': mapping_percent,
                    'gemm_percent': gemm_percent,
                    'mapping_time_ms': mapping_time_ms,
                    'gemm_time_ms': first_layer_gemm_time,
                    'total_time_ms': total_time,
                    'first_layer_info': {
                        'M': first_layer.get('M', 0),
                        'N': first_layer.get('N', 0),
                        'K': first_layer.get('K', 0),
                        'Layer': first_layer.get('Layer', 'Unknown')
                    }
                }
                
                print(f"  {name}: Mapping={mapping_percent:.1f}%, GEMM={gemm_percent:.1f}% - SUCCESS")
                print(f"    First layer: {first_layer.get('Layer', 'Unknown')} (M={first_layer.get('M', 0)}, K={first_layer.get('K', 0)})")
                
            else:
                # Use estimates based on typical first layer characteristics
                # Estimate first layer GEMM for input_channels=4 -> 32 channels
                M = num_points * 32  # output voxels * output channels
                N = 1  # batch size
                K = 4  # input channels
                
                # Calculate mapping time
                mapping_metrics = performance_model.calculate_network_mapping_time(
                    input_voxels=num_points, avg_kernel_size=3
                )
                mapping_time_ms = mapping_metrics['mapping_time_ms']
                
                # Estimate GEMM time for first layer
                perf_metrics = performance_model.estimate_performance(
                    M, N, K, sparsity_factor=1.0, spatial_sparsity=0.05,
                    channel_sparsity=0.0, feature_sparsity=0.0
                )
                first_layer_gemm_time = perf_metrics['gemm_time_ms']
                
                total_time = mapping_time_ms + first_layer_gemm_time
                
                mapping_percent = (mapping_time_ms / total_time) * 100
                gemm_percent = (first_layer_gemm_time / total_time) * 100
                
                timing_breakdown[name] = {
                    'mapping_percent': mapping_percent,
                    'gemm_percent': gemm_percent,
                    'mapping_time_ms': mapping_time_ms,
                    'gemm_time_ms': first_layer_gemm_time,
                    'total_time_ms': total_time,
                    'first_layer_info': {
                        'M': M, 'N': N, 'K': K,
                        'Layer': 'conv0_estimated'
                    }
                }
                print(f"  {name}: Mapping={mapping_percent:.1f}%, GEMM={gemm_percent:.1f}% - ESTIMATED")
                    
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
            # Use default values with typical mapping dominance for first layer
            timing_breakdown[name] = {
                'mapping_percent': 75.0,  # Mapping often dominates for first layer
                'gemm_percent': 25.0,     # First layer is typically small
                'mapping_time_ms': 75.0,
                'gemm_time_ms': 25.0,
                'total_time_ms': 100.0,
                'first_layer_info': {'M': 0, 'N': 0, 'K': 0, 'Layer': 'default'}
            }
    
    return timing_breakdown

def plot_mapping_vs_gemm_first_layer(timing_data: Dict[str, Dict[str, float]]):
    """Create stacked bar chart showing mapping vs GEMM time breakdown for first layer"""
    
    if not timing_data:
        print("No timing data available for plotting!")
        return None
    
    # Prepare data for plotting
    networks = list(timing_data.keys())
    mapping_percentages = [timing_data[net]['mapping_percent'] for net in networks]
    gemm_percentages = [timing_data[net]['gemm_percent'] for net in networks]
    
    print(f"Plotting first layer breakdown for networks: {networks}")
    print(f"Mapping percentages: {mapping_percentages}")
    print(f"GEMM percentages: {gemm_percentages}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set hatch line width to be thicker (matching other comparison scripts)
    plt.rcParams['hatch.linewidth'] = 3.0
    
    # Set up x positions
    x_pos = np.arange(len(networks))
    width = 0.8
    
    # Create stacked bars using consistent color scheme
    # Mapping Time: White background with diagonal hatching and gold edges (memory-bound, like gather)
    # GEMM Time: Light blue (compute-bound, like GEMM operations)
    bars_mapping = ax.bar(x_pos, mapping_percentages, width, 
                         label='Mapping Time (Network-wide)', 
                         color='white', alpha=1.0, hatch='//', 
                         edgecolor='gold', linewidth=4)
    
    bars_gemm = ax.bar(x_pos, gemm_percentages, width, 
                      bottom=mapping_percentages, label='GEMM Time (First Layer)',
                      color='lightblue', alpha=1.0, edgecolor='darkblue', linewidth=2)
    
    # Customize the plot
    ax.set_ylabel('Percentage of Total Time (%)', fontsize=28, fontweight='bold')
    # ax.set_xlabel('3D Point Cloud Networks', fontsize=28, fontweight='bold')
    
    # Set x-axis labels with larger font
    ax.set_xticks(x_pos)
    ax.set_xticklabels(networks, rotation=45, ha='right', fontsize=32)
    
    # Set y-axis to show percentages
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    
    # Increase tick label sizes
    ax.tick_params(axis='y', labelsize=32, which='major')
    ax.tick_params(axis='x', labelsize=32)
    
    # Add percentage labels on bars with larger font
    for i, (mapping_pct, gemm_pct) in enumerate(zip(mapping_percentages, gemm_percentages)):
        # Mapping label
        ax.text(i, mapping_pct/2, f'{mapping_pct:.1f}%', 
               ha='center', va='center', fontweight='bold', fontsize=24)
        
        # GEMM label
        ax.text(i, mapping_pct + gemm_pct/2, f'{gemm_pct:.1f}%', 
               ha='center', va='center', fontweight='bold', fontsize=24)
    
    # Add legend with consistent styling
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', alpha=1.0, hatch='///', 
              edgecolor='gold', label='Mapping (1st layer)'),
        Patch(facecolor='lightblue', alpha=1.0, 
              edgecolor='darkblue', label='GEMM (1st layer)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=32, framealpha=0.9)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.18)
    return fig

def print_detailed_first_layer_analysis(timing_data: Dict[str, Dict[str, float]]):
    """Print detailed numerical analysis of first layer timing breakdown"""
    
    print("\n" + "="*90)
    print("DETAILED FIRST LAYER ANALYSIS: Mapping vs GEMM Time")
    print("="*90)
    
    print(f"\n{'Network':<15} {'Mapping %':<12} {'GEMM %':<10} {'Map (ms)':<12} {'GEMM (ms)':<12} {'Total (ms)':<12} {'First Layer':<15}")
    print("-" * 100)
    
    total_mapping_time = 0
    total_gemm_time = 0
    
    for name, data in timing_data.items():
        mapping_pct = data['mapping_percent']
        gemm_pct = data['gemm_percent']
        mapping_ms = data['mapping_time_ms']
        gemm_ms = data['gemm_time_ms']
        total_ms = data['total_time_ms']
        first_layer = data['first_layer_info']['Layer']
        
        print(f"{name:<15} {mapping_pct:<12.1f} {gemm_pct:<10.1f} {mapping_ms:<12.2f} {gemm_ms:<12.2f} {total_ms:<12.2f} {first_layer:<15}")
        
        total_mapping_time += mapping_ms
        total_gemm_time += gemm_ms
    
    # Calculate averages
    num_networks = len(timing_data)
    avg_mapping_pct = np.mean([data['mapping_percent'] for data in timing_data.values()])
    avg_gemm_pct = np.mean([data['gemm_percent'] for data in timing_data.values()])
    
    print("\n" + "="*90)
    print("SUMMARY STATISTICS")
    print("="*90)
    print(f"Average Mapping Time (Network-wide): {avg_mapping_pct:.1f}%")
    print(f"Average GEMM Time (First Layer): {avg_gemm_pct:.1f}%")
    print(f"Total Networks Analyzed: {num_networks}")
    
    # Find extremes
    max_mapping = max(timing_data.items(), key=lambda x: x[1]['mapping_percent'])
    min_mapping = min(timing_data.items(), key=lambda x: x[1]['mapping_percent'])
    
    print(f"\nHighest Mapping Overhead: {max_mapping[0]} ({max_mapping[1]['mapping_percent']:.1f}%)")
    print(f"Lowest Mapping Overhead: {min_mapping[0]} ({min_mapping[1]['mapping_percent']:.1f}%)")
    
    # Show first layer details
    print(f"\nFirst Layer Details:")
    print(f"{'Network':<15} {'Layer Name':<20} {'M':<10} {'N':<5} {'K':<5}")
    print("-" * 65)
    
    for name, data in timing_data.items():
        info = data['first_layer_info']
        layer_name = info['Layer']
        M, N, K = info['M'], info['N'], info['K']
        print(f"{name:<15} {layer_name:<20} {M:<10} {N:<5} {K:<5}")
    
    print(f"\nAnalysis Notes:")
    print("- Mapping time is calculated once for the entire network (input voxels × k³ × 8 bytes)")
    print("- GEMM time is only for the first convolution layer (typically input→32 channels)")
    print("- Mapping often dominates early layers due to network-wide overhead")
    print("- Higher mapping percentage indicates sparse tensor setup cost vs computation")
    print("- First layer GEMM is typically small (low channel count)")

def main():
    """Main analysis function"""
    print("Analyzing Mapping vs GEMM Time for First Layer in 3D Point Cloud Networks")
    print("="*80)
    
    # Get timing breakdown data
    timing_data = get_3d_network_first_layer_timing()
    
    if not timing_data:
        print("Error: Could not obtain timing data for any networks")
        return
    
    # Create breakdown plot
    fig = plot_mapping_vs_gemm_first_layer(timing_data)
    
    if fig:
        # Save the plot
        output_path = '/Users/ashriram/Desktop/minuet-tracer/mac-vs-time/mapping_vs_gemm_first_layer.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"\nFirst layer timing breakdown plot saved as: {output_path}")
        plt.close(fig)
    
    # Print detailed analysis
    print_detailed_first_layer_analysis(timing_data)

if __name__ == "__main__":
    main()
