#!/usr/bin/env python3
"""
Sparsity Factor Comparison script for 2D LiDAR networks vs 3D sparse convolution networks

This script compares overall sparsity factors calculated as:
Sparsity % = Spatial Sparsity × Activation Sparsity × Weight Sparsity

Networks analyzed:
- 2D LiDAR networks: RangeNet++, EfficientNetV2, SqueezeSeg
- 3D Sparse networks: PolarNet, MinkowskiNet variants, VoxelNeXt, LargeKernel3D variants, SPVNAS variants

The comparison shows the total sparsity potential differences between 2D dense processing
and 3D sparse convolution approaches for LiDAR point cloud processing.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from matplotlib.patches import Patch

def get_2d_network_sparsity() -> Dict[str, float]:
    """Get overall MAC density for 2D LiDAR networks (inverse of sparsity)"""
    
    # 2D Network sparsity data based on provided specifications
    networks_2d = {
        'RangeNet++': {
            'spatial_sparsity': 1.0,      # 100% spatial density (dense grid)
            'activation_sparsity': 1.0,    # No activation sparsity mentioned, assume dense
            'weight_sparsity': 1.0         # No weight sparsity mentioned, assume dense
        },
        'EfficientNetV2': {
            'spatial_sparsity': 1.0,      # 100% spatial density 
            'activation_sparsity': 0.4,    # 60% activation sparsity = 40% density
            'weight_sparsity': 0.5         # Add weight sparsity for EfficientNet (50% density)
        },
        'SqueezeSeg': {
            'spatial_sparsity': 1.0,      # 100% spatial density
            'activation_sparsity': 0.4,    # 60% activation sparsity = 40% density
            'weight_sparsity': 0.5         # Add weight sparsity for SqueezeSeg (50% density)
        }
    }
    
    results_2d = {}
    
    print("Calculating 2D network MAC density (inverse of sparsity)...")
    for name, sparsity_data in networks_2d.items():
        # Calculate overall MAC density as product of all density factors
        # (density = 1 - sparsity, so we use the complement for activation/weight)
        overall_mac_density = (sparsity_data['spatial_sparsity'] * 
                              sparsity_data['activation_sparsity'] * 
                              sparsity_data['weight_sparsity'])
        
        # Convert to percentage and store
        mac_density_percentage = overall_mac_density * 100
        results_2d[name] = mac_density_percentage
        
        # Calculate sparsity percentages for display
        spatial_density = sparsity_data['spatial_sparsity'] * 100
        activation_density = sparsity_data['activation_sparsity'] * 100
        weight_density = sparsity_data['weight_sparsity'] * 100
        
        print(f"  {name}: Spatial={spatial_density:.0f}%, "
              f"Activation={activation_density:.0f}%, "
              f"Weight={weight_density:.0f}% "
              f"→ MAC Density={mac_density_percentage:.1f}%")
    
    return results_2d

def get_3d_network_sparsity() -> Dict[str, float]:
    """Get overall sparsity factors for 3D sparse convolution networks"""
    
    # 3D Network sparsity data based on provided specifications
    networks_3d = {
        'PolarNet': {
            'spatial_sparsity': 0.025,     # 2-3%, use 2.5%
            'activation_sparsity': 0.5,    # 50% activation sparsity
            'weight_sparsity': 0.5         # 50% weight sparsity (other networks)
        },
        'MinkowskiNet (S3DIS)': {
            'spatial_sparsity': 0.04,      # 4%
            'activation_sparsity': 0.7,    # 70% activation sparsity (Minkowski exception)
            'weight_sparsity': 0.9         # 90% weight sparsity (Minkowski)
        },
        'MinkowskiNet (ScanNet)': {
            'spatial_sparsity': 0.01,      # 1%
            'activation_sparsity': 0.7,    # 70% activation sparsity (Minkowski exception)
            'weight_sparsity': 0.9         # 90% weight sparsity (Minkowski)
        },
        'VoxelNeXt (nuScenes)': {
            'spatial_sparsity': 0.10,      # 10%
            'activation_sparsity': 0.5,    # 50% activation sparsity
            'weight_sparsity': 0.5         # 50% weight sparsity (other networks)
        },
        'LargeKernel3D (ScanNet)': {
            'spatial_sparsity': 0.004,     # 0.4%
            'activation_sparsity': 0.5,    # 50% activation sparsity
            'weight_sparsity': 0.5         # 50% weight sparsity (other networks)
        },
        'LargeKernel3D (S3DIS)': {
            'spatial_sparsity': 0.01,      # 1%
            'activation_sparsity': 0.5,    # 50% activation sparsity
            'weight_sparsity': 0.5         # 50% weight sparsity (other networks)
        },
        'SPVNAS (KITTI)': {
            'spatial_sparsity': 0.30,      # 30%
            'activation_sparsity': 0.5,    # 50% activation sparsity
            'weight_sparsity': 0.45        # 45% weight sparsity (SPVNAS)
        },
        'SPVNAS (nuScenes)': {
            'spatial_sparsity': 0.10,      # 10%
            'activation_sparsity': 0.5,    # 50% activation sparsity
            'weight_sparsity': 0.45        # 45% weight sparsity (SPVNAS)
        }
    }
    
    results_3d = {}
    
    print("Calculating 3D network sparsity factors...")
    for name, sparsity_data in networks_3d.items():
        # Calculate overall sparsity as product of all sparsity factors
        overall_sparsity = (sparsity_data['spatial_sparsity'] * 
                           sparsity_data['activation_sparsity'] * 
                           sparsity_data['weight_sparsity'])
        
        # Convert to percentage and store
        sparsity_percentage = overall_sparsity * 100
        results_3d[name] = sparsity_percentage
        
        print(f"  {name}: Spatial={sparsity_data['spatial_sparsity']*100:.1f}%, "
              f"Activation={sparsity_data['activation_sparsity']*100:.0f}%, "
              f"Weight={sparsity_data['weight_sparsity']*100:.0f}% "
              f"→ Overall={sparsity_percentage:.3f}%")
    
    return results_3d

def plot_sparsity_comparison(results_2d: Dict[str, float], results_3d: Dict[str, float]):
    """Create comparison plot of sparsity factors (matching compare_2d_vs_3d_macs.py layout)"""
    
    print(f"Plotting with 2D results: {results_2d}")
    print(f"Plotting with 3D results: {results_3d}")
    
    # Prepare data for plotting - combine all networks
    all_networks = []
    all_sparsity = []
    all_colors = []
    
    # Add 2D networks (blue) - filter out zero values
    for name, sparsity in results_2d.items():
        if sparsity > 0:  # Only include networks with valid sparsity
            all_networks.append(name)
            all_sparsity.append(sparsity)
            all_colors.append('blue')
            print(f"Added 2D network: {name} with {sparsity:.3f}% sparsity")
    
    # Add 3D networks (red/coral) - filter out zero values
    for name, sparsity in results_3d.items():
        if sparsity > 0:  # Only include networks with valid sparsity
            all_networks.append(name)
            all_sparsity.append(sparsity)
            all_colors.append('red')
            print(f"Added 3D network: {name} with {sparsity:.3f}% sparsity")
    
    print(f"Total networks to plot: {len(all_networks)}")
    print(f"Network names: {all_networks}")
    print(f"Sparsity values: {all_sparsity}")
    
    if not all_networks:
        print("No valid networks to plot!")
        return None
    
    # Create the plot (matching compare_2d_vs_3d_macs.py layout)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set hatch line width to be thicker
    plt.rcParams['hatch.linewidth'] = 3.0
    
    # Create bar chart with different patterns (matching 2D vs 3D comparison styling)
    x_pos = np.arange(len(all_networks))
    bars = []
    
    for i, (x, height, color) in enumerate(zip(x_pos, all_sparsity, all_colors)):
        if color == 'blue':  # 2D networks
            # Filled light blue (matching 2D vs 3D comparison)
            bar = ax.bar(x, height, color='lightblue', alpha=1.0, 
                        edgecolor='darkblue', linewidth=2, width=0.8)
        else:  # 3D networks
            # White background with thick diagonal slashes and gold edges
            bar = ax.bar(x, height, color='white', alpha=1.0, 
                        edgecolor='gold', linewidth=4, hatch='//', width=0.8)
        bars.extend(bar)
    
    # Customize the plot (matching 2D vs 3D comparison styling)
    ax.set_ylabel('MAC Density(% log scale)', fontsize=28, fontweight='bold')
    
    # Set x-axis labels with larger font
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_networks, rotation=45, ha='right', fontsize=32)
    
    # Increase tick label sizes
    ax.tick_params(axis='y', labelsize=32, which='major')
    ax.tick_params(axis='y', labelsize=32, which='minor')
    ax.tick_params(axis='x', labelsize=32)
    
    # Add legend with same styling as 2D vs 3D comparison
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
    
    # Set y-axis to log scale (as requested)
    ax.set_yscale('log')
    if all_sparsity:
        # Set appropriate limits for sparsity data
        min_sparsity = min(all_sparsity)
        max_sparsity = max(all_sparsity)
        ax.set_ylim(min_sparsity * 0.5, max_sparsity * 2.0)
    
    # Add comparison arrow and text (if there's clear separation)
    if results_2d and results_3d:
        avg_2d = np.mean(list(results_2d.values()))
        avg_3d = np.mean(list(results_3d.values()))
        
        if avg_2d > avg_3d:
            ratio = avg_2d / avg_3d
            
            # Find good positions for the arrow
            num_2d_valid = len([s for s in results_2d.values() if s > 0])
            num_3d_valid = len([s for s in results_3d.values() if s > 0])
            
            if num_2d_valid > 0 and num_3d_valid > 0:
                # Vertical arrow from top to bottom at separator line
                separator_x = num_2d_valid - 0.5
                arrow_top = max(all_sparsity) * 0.8
                arrow_bottom = min(all_sparsity) * 3
                
                ax.annotate('', xy=(separator_x, arrow_bottom), 
                           xytext=(separator_x, arrow_top),
                           arrowprops=dict(arrowstyle='->', lw=8, color='red', 
                                         shrinkA=0, shrinkB=0, mutation_scale=30))
                
                # Add text explaining the difference
                ax.text(separator_x + 0.5, arrow_bottom * 2,
                       f'3D Networks:\n{ratio:.0f}x higher sparsity',
                       ha='left', va='center', fontsize=32, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
    
    # Add vertical line to separate 2D and 3D networks
    num_2d_valid = len([s for s in results_2d.values() if s > 0])
    if num_2d_valid > 0:
        separator_x = num_2d_valid - 0.5
        ax.axvline(x=separator_x, color='gray', linestyle='--', alpha=0.7, linewidth=3)
    
    # Adjust layout to match 2D vs 3D comparison
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(left=0.10, right=0.93, top=0.90, bottom=0.18)
    return fig

def print_detailed_sparsity_analysis(results_2d: Dict[str, float], results_3d: Dict[str, float]):
    """Print detailed numerical analysis of sparsity factors"""
    
    print("\n" + "="*90)
    print("DETAILED SPARSITY ANALYSIS: 2D vs 3D Networks")
    print("="*90)
    
    # Combined results for overall ranking
    all_results = {}
    for name, sparsity in results_2d.items():
        all_results[f"{name} (2D)"] = sparsity
    for name, sparsity in results_3d.items():
        all_results[f"{name} (3D)"] = sparsity
    
    print("\nOverall Ranking (Highest to Lowest Sparsity):")
    print("-" * 70)
    print(f"{'Rank':<5} {'Network':<35} {'Type':<6} {'Sparsity %':<12}")
    print("-" * 70)
    
    # Sort all networks by sparsity (descending = highest sparsity first)
    sorted_all = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, sparsity) in enumerate(sorted_all, 1):
        network_type = "2D" if "(2D)" in name else "3D"
        clean_name = name.replace(" (2D)", "").replace(" (3D)", "")
        print(f"{rank:<5} {clean_name:<35} {network_type:<6} {sparsity:<12.3f}")
    
    print("\n2D Range Image Networks:")
    print("-" * 70)
    print(f"{'Network':<25} {'Sparsity %':<12} {'Rank in 2D':<12} {'Overall Rank':<15}")
    print("-" * 70)
    
    # Sort 2D networks by sparsity
    sorted_2d = sorted(results_2d.items(), key=lambda x: x[1], reverse=True)
    
    for rank_2d, (name, sparsity) in enumerate(sorted_2d, 1):
        overall_rank = next(i for i, (full_name, _) in enumerate(sorted_all, 1) if f"{name} (2D)" == full_name)
        print(f"{name:<25} {sparsity:<12.3f} {rank_2d:<12} {overall_rank:<15}")
    
    print("\n3D Sparse Convolution Networks:")
    print("-" * 70)
    print(f"{'Network':<25} {'Sparsity %':<12} {'Rank in 3D':<12} {'Overall Rank':<15}")
    print("-" * 70)
    
    # Sort 3D networks by sparsity
    sorted_3d = sorted(results_3d.items(), key=lambda x: x[1], reverse=True)
    
    for rank_3d, (name, sparsity) in enumerate(sorted_3d, 1):
        overall_rank = next(i for i, (full_name, _) in enumerate(sorted_all, 1) if f"{name} (3D)" == full_name)
        print(f"{name:<25} {sparsity:<12.3f} {rank_3d:<12} {overall_rank:<15}")
    
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
        
        print(f"Average Sparsity Factor:")
        print(f"  2D Range Image Networks: {avg_2d:.3f}%")
        print(f"  3D Sparse Networks: {avg_3d:.3f}%")
        print(f"  Ratio (2D/3D): {avg_2d/avg_3d:.1f}x")
        
        print(f"\nSparsity Range:")
        print(f"  2D Networks: {min_2d:.3f}% - {max_2d:.3f}% (spread: {max_2d-min_2d:.3f}%)")
        print(f"  3D Networks: {min_3d:.3f}% - {max_3d:.3f}% (spread: {max_3d-min_3d:.3f}%)")
        
        print(f"\nHighest Sparsity Networks:")
        highest_2d = max(results_2d.items(), key=lambda x: x[1])
        highest_3d = max(results_3d.items(), key=lambda x: x[1])
        highest_overall = max(all_results.items(), key=lambda x: x[1])
        
        print(f"  Best 2D: {highest_2d[0]} ({highest_2d[1]:.3f}% sparsity)")
        print(f"  Best 3D: {highest_3d[0]} ({highest_3d[1]:.3f}% sparsity)")
        print(f"  Overall: {highest_overall[0]} ({highest_overall[1]:.3f}% sparsity)")
        
        print(f"\nSparsity Analysis Notes:")
        print("- Overall sparsity = spatial × activation × weight sparsity")
        print("- 2D networks have 100% spatial density but lower activation/weight sparsity")
        print("- 3D networks have very low spatial density but higher activation/weight sparsity")
        print("- Lower overall sparsity indicates more computation required")

def main():
    """Main sparsity comparison function"""
    print("Comparing Sparsity Factors: 2D Range Image vs 3D Sparse Convolution Networks")
    print("="*90)
    
    # Get sparsity factors for both categories
    print("Calculating 2D network sparsity factors...")
    results_2d = get_2d_network_sparsity()
    print(f"2D results: {results_2d}")
    
    print("\nCalculating 3D network sparsity factors...")
    results_3d = get_3d_network_sparsity()
    print(f"3D results: {results_3d}")
    
    if not results_2d and not results_3d:
        print("Error: Could not calculate sparsity for either network category")
        return
    
    # Create comparison plot
    if results_2d or results_3d:
        fig = plot_sparsity_comparison(results_2d, results_3d)
        
        # Save the plot
        output_path = '/Users/ashriram/Desktop/minuet-tracer/mac-vs-time/2d_vs_3d_sparsity.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=1.0, facecolor='white')
        print(f"\nSparsity comparison plot saved as: {output_path}")
        plt.close(fig)
    
    # Print detailed analysis
    print_detailed_sparsity_analysis(results_2d, results_3d)
    
    print(f"\nAnalysis Notes:")
    print("- 2D networks process dense range images with varying activation/weight sparsity")
    print("- 3D networks process highly sparse voxel grids with additional pruning")
    print("- Sparsity factor shows total computational reduction potential")
    print("- Lower sparsity percentage indicates denser computation")

if __name__ == "__main__":
    main()
