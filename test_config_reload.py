#!/usr/bin/env python3
"""
Test script to verify that configuration reloading works properly.
"""

import minuet_config
import json

def test_config_reload():
    """Test that configuration reloading updates global variables."""
    print("=== Testing Configuration Reload ===")
    
    # Load initial config
    print("Loading initial configuration...")
    minuet_config.get_config('config.json')
    minuet_config.print_current_config()
    
    # Store initial values
    initial_threads = minuet_config.NUM_THREADS
    initial_gemm_size = minuet_config.GEMM_SIZE
    
    # Create a temporary config with different values
    test_config = {
        "NUM_THREADS": 256,
        "GEMM_SIZE": 32768,
        "GEMM_ALIGNMENT": 8192,
        "debug": True,
        "output_dir": "./test_out/"
    }
    
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print("Created test_config.json with different values")
    
    # Reload with new config
    print("Reloading with test configuration...")
    minuet_config.reload_config('test_config.json')
    minuet_config.print_current_config()
    
    # Verify changes
    if minuet_config.NUM_THREADS == 256:
        print("✓ NUM_THREADS updated correctly")
    else:
        print(f"✗ NUM_THREADS not updated: expected 256, got {minuet_config.NUM_THREADS}")
    
    if minuet_config.GEMM_SIZE == 32768:
        print("✓ GEMM_SIZE updated correctly")
    else:
        print(f"✗ GEMM_SIZE not updated: expected 32768, got {minuet_config.GEMM_SIZE}")
    
    if minuet_config.debug == True:
        print("✓ debug flag updated correctly")
    else:
        print(f"✗ debug flag not updated: expected True, got {minuet_config.debug}")
    
    if minuet_config.output_dir == "./test_out/":
        print("✓ output_dir updated correctly")
    else:
        print(f"✗ output_dir not updated: expected './test_out/', got {minuet_config.output_dir}")
    
    # Reload back to original
    print("\nReloading back to original configuration...")
    minuet_config.reload_config('config.json')
    minuet_config.print_current_config()
    
    # Clean up
    import os
    if os.path.exists('test_config.json'):
        os.remove('test_config.json')
        print("Cleaned up test_config.json")
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_config_reload()
