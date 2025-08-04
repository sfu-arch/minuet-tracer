#!/usr/bin/env python3
"""
Comparison script for MACs per point between 2D LiDAR networks and 3D sparse convolution networks

This script compares:
- 2D LiDAR networks: SqueezeSeg, SalsaNext, RangeNet++, PolarNet, CENet, EfficientNetV2, ConvNeXt
- 3D Sparse networks: MinkowskiNet, SPVNAS, LargeKernel3D, VoxelNeXt, RSN

The comparison shows the computational efficiency differences between 2D range image processing
and 3D sparse convolution approaches for LiDAR point cloud processing.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import importlib.util

# Import the network classes from both files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import 2D networks
from typing import Dict, List, Tuple

def get_2d_network_macs() -> Dict[str, float]:
    """Get MACs per point for 2D LiDAR networks"""
    
    # Import 2D network classes using importlib
    try:
        # Load the 2d_macs module
        spec = importlib.util.spec_from_file_location("2d_macs", "/Users/ashriram/Desktop/minuet-tracer/mac-vs-time/2d_macs.py")
        module_2d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_2d)
        
        # Extract classes
        SqueezeSeg = module_2d.SqueezeSeg
        SalsaNext = module_2d.SalsaNext
        RangeNetPP = module_2d.RangeNetPP  # Correct class name
        PolarNet = module_2d.PolarNet
        CENet = module_2d.CENet
        EfficientNetV2 = module_2d.EfficientNetV2
        ConvNeXt = module_2d.ConvNeXt
        SparsityConfig = module_2d.SparsityConfig
        
    except Exception as e:
        print(f"Error importing 2D networks: {e}")
        return {}
    
    # Standard configuration for fair comparison
    num_points = 100000  # Same as 3D networks
    
    # Initialize networks with dense configuration (no sparsity for baseline)
    sparsity_config = SparsityConfig(feature_sparsity=0.0, weight_sparsity=0.0)
    
    networks_2d = {
        'SqueezeSeg': SqueezeSeg(
            input_height=64, input_width=1024, input_channels=4, 
            num_classes=20, sparsity_config=sparsity_config
        ),
        'SalsaNext': SalsaNext(
            input_height=64, input_width=1024, input_channels=4,
            num_classes=20, sparsity_config=sparsity_config
        ),
        'RangeNet++': RangeNetPP(  # Use correct class name
            input_height=64, input_width=1024, input_channels=4,
            num_classes=20, sparsity_config=sparsity_config
        ),
        'PolarNet': PolarNet(
            input_height=64, input_width=1024, input_channels=4,
            num_classes=20, sparsity_config=sparsity_config
        ),
        'CENet': CENet(
            input_height=64, input_width=1024, input_channels=4,
            num_classes=20, sparsity_config=sparsity_config
        ),
        'EfficientNetV2': EfficientNetV2(
            input_height=64, input_width=1024, input_channels=4,
            num_classes=20, sparsity_config=sparsity_config
        ),
        'ConvNeXt': ConvNeXt(
            input_height=64, input_width=1024, input_channels=4,
            num_classes=20, sparsity_config=sparsity_config
        )
    }
    
    results_2d = {}
    
    print("Calculating 2D network MACs...")
    for name, network in networks_2d.items():
        try:
            print(f"  Calculating {name}...")
            result = network.calculate_macs()
            macs_per_point = result['macs_per_point']
            results_2d[name] = macs_per_point
            print(f"  {name}: {macs_per_point:.2f} MACs/point - SUCCESS")
        except Exception as e:
            print(f"  Error calculating {name}: {e}")
            results_2d[name] = 0.0
    
    print(f"Final 2D results: {results_2d}")
    return results_2d

def get_3d_network_macs() -> Dict[str, float]:
    """Get MACs per point for 3D sparse convolution networks"""
    
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
        
    except Exception as e:
        print(f"Error importing 3D networks: {e}")
        return {}
    
    # Standard configuration for fair comparison
    num_points = 100000
    
    # Initialize networks with dense configuration (minimal sparsity for baseline)
    networks_3d = {
        'MinkowskiNet': MinkowskiNet(
            num_points=num_points, voxel_size=0.05, input_channels=4,
            num_classes=20, spatial_sparsity=0.05,  # Only spatial sparsity (inherent to sparse conv)
            channel_sparsity=0.0, feature_sparsity=0.0  # No additional sparsity
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
    
    results_3d = {}
    
    print("Calculating 3D network MACs...")
    for name, network in networks_3d.items():
        try:
            print(f"  Calculating {name}...")
            result = network.calculate_macs()
            macs_per_point = result['macs_per_point']
            results_3d[name] = macs_per_point
            print(f"  {name}: {macs_per_point:.2f} MACs/point - SUCCESS")
        except Exception as e:
            print(f"  Error calculating {name}: {e}")
            results_3d[name] = 0.0
    
    print(f"Final 3D results: {results_3d}")
    return results_3d

def plot_macs_comparison(results_2d: Dict[str, float], results_3d: Dict[str, float]):
    """Create comparison plot of MACs per point"""
    
    print(f"Plotting with 2D results: {results_2d}")
    print(f"Plotting with 3D results: {results_3d}")
    
    # Prepare data for plotting - combine all networks
    all_networks = []
    all_macs = []
    all_colors = []
    
    # Add 2D networks (blue) - filter out zero values
    for name, macs in results_2d.items():
        if macs > 0:  # Only include networks with valid MAC counts
            all_networks.append(name)
            all_macs.append(macs)
            all_colors.append('blue')
            print(f"Added 2D network: {name} with {macs} MACs/point")
    
    # Add 3D networks (red/coral) - filter out zero values
    for name, macs in results_3d.items():
        if macs > 0:  # Only include networks with valid MAC counts
            all_networks.append(name)
            all_macs.append(macs)
            all_colors.append('red')
            print(f"Added 3D network: {name} with {macs} MACs/point")
    
    print(f"Total networks to plot: {len(all_networks)}")
    print(f"Network names: {all_networks}")
    print(f"MAC values: {all_macs}")
    
    if not all_networks:
        print("No valid networks to plot!")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set hatch line width to be thicker
    plt.rcParams['hatch.linewidth'] = 3.0
    
    # Create bar chart with different patterns
    x_pos = np.arange(len(all_networks))
    bars = []
    
    for i, (x, height, color) in enumerate(zip(x_pos, all_macs, all_colors)):
        if color == 'blue':  # 2D networks
            # Filled light blue
            bar = ax.bar(x, height, color='lightblue', alpha=1.0, 
                        edgecolor='darkblue', linewidth=2, width=0.8)
        else:  # 3D networks
            # White background with thick yellow slashes
            bar = ax.bar(x, height, color='white', alpha=1.0, 
                        edgecolor='gold', linewidth=4, hatch='//', width=0.8)
        bars.extend(bar)
    
    # Customize the plot - NO TITLE
    # ax.set_xlabel('Network Architecture', fontsize=18, fontweight='bold')
    ax.set_ylabel('MACs per Point (log scale)', fontsize=28, fontweight='bold')
    # ax.yaxis.set_label_coords(-0.08, 0.3)  # Move label downward
    
    # Set x-axis labels with larger font
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_networks, rotation=45, ha='right', fontsize=32)
    
    # Increase tick label sizes
    ax.tick_params(axis='y', labelsize=32, which='major')
    ax.tick_params(axis='y', labelsize=32, which='minor')
    ax.tick_params(axis='x', labelsize=32)
    
    # Remove value labels on bars - commented out for cleaner look
    # for i, (bar, macs) in enumerate(zip(bars, all_macs)):
    #     height = bar.get_height()
    #     # Adjust label position for log scale
    #     ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
    #             f'{macs:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add legend with larger font
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=1.0, 
              edgecolor='darkblue', label='2D Convolutions'),
        Patch(facecolor='white', alpha=1.0, hatch='///', 
              edgecolor='gold', label='3D Sparse Convolutions')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=32, framealpha=0.9)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    if all_macs:
        ax.set_ylim(min(all_macs) * 0.3, 1e8)  # Cap at 10^8
    
    # Add comparison arrow and text
    if results_2d and results_3d:
        avg_2d = np.mean([m for m in results_2d.values() if m > 0])
        avg_3d = np.mean([m for m in results_3d.values() if m > 0])
        
        if avg_3d > avg_2d:
            ratio = avg_3d / avg_2d
            
            # Find good positions for the arrow
            num_2d_valid = sum(1 for macs in results_2d.values() if macs > 0)
            num_3d_valid = sum(1 for macs in results_3d.values() if macs > 0)
            
            if num_2d_valid > 0 and num_3d_valid > 0:
                # Vertical arrow from bottom to top at separator line
                separator_x = num_2d_valid - 0.5
                arrow_bottom = min(all_macs) * 100
                arrow_top = max(all_macs) * 0.6
                
                ax.annotate('', xy=(separator_x, arrow_top), 
                           xytext=(separator_x, arrow_bottom),
                           arrowprops=dict(arrowstyle='->', lw=8, color='red', 
                                         shrinkA=0, shrinkB=0, mutation_scale=30))
                
                # Add text explaining the difference
                ax.text(separator_x + 0.5, arrow_top * 0.1,
                       f'3D Networks:\n{ratio:.0f}x more MACs/point',
                       ha='left', va='center', fontsize=32, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
    
    # Add vertical line to separate 2D and 3D networks
    num_2d_valid = sum(1 for macs in results_2d.values() if macs > 0)
    if num_2d_valid > 0:
        separator_x = num_2d_valid - 0.5
        ax.axvline(x=separator_x, color='gray', linestyle='--', alpha=0.7, linewidth=3)
    
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(left=0.10, right=0.93, top=0.90, bottom=0.18)
    return fig

def print_detailed_comparison(results_2d: Dict[str, float], results_3d: Dict[str, float]):
    """Print detailed numerical comparison"""
    
    print("\n" + "="*90)
    print("DETAILED COMPARISON: 2D LiDAR vs 3D Sparse Networks")
    print("="*90)
    
    # Combined results for overall ranking
    all_results = {}
    for name, macs in results_2d.items():
        all_results[f"{name} (2D)"] = macs
    for name, macs in results_3d.items():
        all_results[f"{name} (3D)"] = macs
    
    print("\nOverall Ranking (Most to Least Efficient):")
    print("-" * 60)
    print(f"{'Rank':<5} {'Network':<25} {'Type':<6} {'MACs/Point':<12}")
    print("-" * 60)
    
    # Sort all networks by MACs per point (ascending = most efficient first)
    sorted_all = sorted(all_results.items(), key=lambda x: x[1])
    
    for rank, (name, macs) in enumerate(sorted_all, 1):
        network_type = "2D" if "(2D)" in name else "3D"
        clean_name = name.replace(" (2D)", "").replace(" (3D)", "")
        print(f"{rank:<5} {clean_name:<25} {network_type:<6} {macs:<12.2f}")
    
    print("\n2D LiDAR Networks (Range Image Processing):")
    print("-" * 60)
    print(f"{'Network':<20} {'MACs/Point':<12} {'Rank in 2D':<12} {'Overall Rank':<15}")
    print("-" * 60)
    
    # Sort 2D networks by MACs per point
    sorted_2d = sorted(results_2d.items(), key=lambda x: x[1])
    
    for rank_2d, (name, macs) in enumerate(sorted_2d, 1):
        overall_rank = next(i for i, (full_name, _) in enumerate(sorted_all, 1) if f"{name} (2D)" == full_name)
        print(f"{name:<20} {macs:<12.2f} {rank_2d:<12} {overall_rank:<15}")
    
    print("\n3D Sparse Convolution Networks:")
    print("-" * 60)
    print(f"{'Network':<20} {'MACs/Point':<12} {'Rank in 3D':<12} {'Overall Rank':<15}")
    print("-" * 60)
    
    # Sort 3D networks by MACs per point
    sorted_3d = sorted(results_3d.items(), key=lambda x: x[1])
    
    for rank_3d, (name, macs) in enumerate(sorted_3d, 1):
        overall_rank = next(i for i, (full_name, _) in enumerate(sorted_all, 1) if f"{name} (3D)" == full_name)
        print(f"{name:<20} {macs:<12.2f} {rank_3d:<12} {overall_rank:<15}")
    
    # Overall comparison
    print("\n" + "="*90)
    print("STATISTICAL SUMMARY")
    print("="*90)
    
    if results_2d and results_3d:
        avg_2d = np.mean(list(results_2d.values()))
        avg_3d = np.mean(list(results_3d.values()))
        
        min_2d = min(results_2d.values())
        max_2d = max(results_2d.values())
        min_3d = min(results_3d.values())
        max_3d = max(results_3d.values())
        
        print(f"Average MACs/Point:")
        print(f"  2D LiDAR Networks: {avg_2d:.2f}")
        print(f"  3D Sparse Networks: {avg_3d:.2f}")
        print(f"  Ratio (3D/2D): {avg_3d/avg_2d:.2f}x")
        
        print(f"\nRange of MACs/Point:")
        print(f"  2D Networks: {min_2d:.2f} - {max_2d:.2f} (spread: {max_2d-min_2d:.2f})")
        print(f"  3D Networks: {min_3d:.2f} - {max_3d:.2f} (spread: {max_3d-min_3d:.2f})")
        
        print(f"\nMost Efficient Networks:")
        most_efficient_2d = min(results_2d.items(), key=lambda x: x[1])
        most_efficient_3d = min(results_3d.items(), key=lambda x: x[1])
        most_efficient_overall = min(all_results.items(), key=lambda x: x[1])
        
        print(f"  Best 2D: {most_efficient_2d[0]} ({most_efficient_2d[1]:.2f} MACs/point)")
        print(f"  Best 3D: {most_efficient_3d[0]} ({most_efficient_3d[1]:.2f} MACs/point)")
        print(f"  Overall: {most_efficient_overall[0]} ({most_efficient_overall[1]:.2f} MACs/point)")
        
        if most_efficient_3d[1] < most_efficient_2d[1]:
            efficiency_gain = most_efficient_2d[1]/most_efficient_3d[1]
            print(f"  → Best 3D is {efficiency_gain:.2f}x more efficient than best 2D!")
        else:
            efficiency_gain = most_efficient_3d[1]/most_efficient_2d[1]
            print(f"  → Best 2D is {efficiency_gain:.2f}x more efficient than best 3D!")
        
        # Count networks in different efficiency ranges
        print(f"\nEfficiency Distribution:")
        
        # Find efficiency thresholds
        all_macs = list(all_results.values())
        threshold_low = np.percentile(all_macs, 33)
        threshold_high = np.percentile(all_macs, 67)
        
        high_eff_2d = sum(1 for macs in results_2d.values() if macs <= threshold_low)
        med_eff_2d = sum(1 for macs in results_2d.values() if threshold_low < macs <= threshold_high)
        low_eff_2d = sum(1 for macs in results_2d.values() if macs > threshold_high)
        
        high_eff_3d = sum(1 for macs in results_3d.values() if macs <= threshold_low)
        med_eff_3d = sum(1 for macs in results_3d.values() if threshold_low < macs <= threshold_high)
        low_eff_3d = sum(1 for macs in results_3d.values() if macs > threshold_high)
        
        print(f"  High Efficiency (≤{threshold_low:.1f} MACs/pt): 2D={high_eff_2d}, 3D={high_eff_3d}")
        print(f"  Medium Efficiency ({threshold_low:.1f}-{threshold_high:.1f} MACs/pt): 2D={med_eff_2d}, 3D={med_eff_3d}")
        print(f"  Lower Efficiency (>{threshold_high:.1f} MACs/pt): 2D={low_eff_2d}, 3D={low_eff_3d}")

def main():
    """Main comparison function"""
    print("Comparing 2D LiDAR Networks vs 3D Sparse Convolution Networks")
    print("="*80)
    
    # Get MACs per point for both categories
    print("Loading 2D networks...")
    results_2d = get_2d_network_macs()
    print(f"2D results: {results_2d}")
    
    print("\nLoading 3D networks...")
    results_3d = get_3d_network_macs()
    print(f"3D results: {results_3d}")
    
    if not results_2d and not results_3d:
        print("Error: Could not load results from either network category")
        return
    
    # Create comparison plot
    if results_2d or results_3d:
        fig = plot_macs_comparison(results_2d, results_3d)
        
        # Save the plot
        output_path = '/Users/ashriram/Desktop/minuet-tracer/mac-vs-time/2d_vs_3d_macs_comparison.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=1.0, facecolor='white')
        print(f"\nComparison plot saved as: {output_path}")
        plt.close(fig)
    
    # Print detailed comparison
    print_detailed_comparison(results_2d, results_3d)
    
    print(f"\nAnalysis Notes:")
    print("- 2D networks process range images (H×W) with dense convolutions")
    print("- 3D networks process sparse voxel grids with sparse convolutions")
    print("- Both use 100K points, 4 input channels, 20 output classes")
    print("- Baseline comparison without additional sparsity optimizations")
    print("- Lower MACs/point indicates higher computational efficiency")

if __name__ == "__main__":
    main()
