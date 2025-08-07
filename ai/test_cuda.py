#!/usr/bin/env python3
"""
Test script to verify CUDA availability and GPU setup.
"""

import torch
import sys

def test_cuda():
    """Test CUDA availability and display GPU information."""
    print("=== CUDA and GPU Information ===")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Device name: {device_name}")
        
        # Get device properties
        device_properties = torch.cuda.get_device_properties(current_device)
        print(f"Device properties:")
        print(f"  - Total memory: {device_properties.total_memory / 1024**3:.1f} GB")
        print(f"  - Multi-processor count: {device_properties.multi_processor_count}")
        print(f"  - Compute capability: {device_properties.major}.{device_properties.minor}")
        
        # Test tensor operations on GPU
        print("\n=== Testing GPU Operations ===")
        try:
            # Create a test tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Perform matrix multiplication
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            z = torch.mm(x, y)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            
            print(f"GPU matrix multiplication (1000x1000): {elapsed_time:.2f} ms")
            print("✅ GPU operations working correctly!")
            
        except Exception as e:
            print(f"❌ Error during GPU operations: {e}")
            return False
            
    else:
        print("❌ CUDA is not available. Please check your GPU drivers and PyTorch installation.")
        return False
    
    return True

def test_memory():
    """Test GPU memory operations."""
    if not torch.cuda.is_available():
        return
    
    print("\n=== GPU Memory Test ===")
    
    # Get initial memory usage
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial GPU memory usage: {initial_memory:.2f} MB")
    
    # Allocate some memory
    tensor_size = 1000
    test_tensor = torch.randn(tensor_size, tensor_size).cuda()
    
    # Check memory after allocation
    allocated_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Memory after allocating {tensor_size}x{tensor_size} tensor: {allocated_memory:.2f} MB")
    
    # Free memory
    del test_tensor
    torch.cuda.empty_cache()
    
    # Check memory after freeing
    final_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Memory after freeing: {final_memory:.2f} MB")
    
    print("✅ GPU memory management working correctly!")

if __name__ == "__main__":
    print("Testing CUDA and GPU setup...")
    
    if test_cuda():
        test_memory()
        print("\n✅ All tests passed! GPU is ready for training.")
        sys.exit(0)
    else:
        print("\n❌ GPU tests failed. Please check your setup.")
        sys.exit(1) 