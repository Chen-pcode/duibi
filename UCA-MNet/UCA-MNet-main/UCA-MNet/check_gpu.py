#!/usr/bin/env python3
"""
GPU Check Script for LM_Net Ablation Study
Run this to verify GPU setup before running ablation studies
"""

import torch
import sys

def check_gpu_setup():
    """Comprehensive GPU setup check for Apple Silicon and NVIDIA"""
    print("🔍 GPU Setup Check")
    print("=" * 50)
    
    # Basic PyTorch and device check
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"MPS built: {torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False}")
    
    # Determine best device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n✅ Using NVIDIA CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print(f"\n✅ Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        print(f"\n⚠️  Using CPU only")
    
    print(f"Selected device: {device}")
    
    if device == "cuda":
        # NVIDIA GPU details
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        print()
        
        # GPU details
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print()
            
    elif device == "mps":
        # Apple Silicon details
        print(f"✅ Apple Silicon GPU acceleration enabled")
        print(f"✅ MPS backend available")
        print()
        
    # Memory test
    print("🧪 Memory Test:")
    try:
        # Test tensor creation on selected device
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"✅ {device.upper()} tensor operations: SUCCESS")
        
        if device == "cuda":
            print(f"✅ Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"✅ Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        elif device == "mps":
            print(f"✅ MPS tensor operations completed successfully")
        
        # Cleanup
        del x, y, z
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ {device.upper()} tensor operations: FAILED - {e}")
        
    # Quick model test
    print(f"\n🏗️  Model Test:")
    try:
        from LM_Net_dw import LM_Net
        
        # Test minimal model
        model = LM_Net(use_pfm=False, use_pdam=False, use_sem=False, use_frcm=False).to(device)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"✅ LM_Net on {device.upper()}: SUCCESS")
        print(f"✅ Input shape: {dummy_input.shape}")
        print(f"✅ Output shape: {output.shape}")
        
        if device == "cuda":
            print(f"✅ Model memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        elif device == "mps":
            print(f"✅ Model running on Apple Silicon GPU")
        
        # Cleanup
        del model, dummy_input, output
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ LM_Net on {device.upper()}: FAILED - {e}")
    else:
        print("❌ No GPU acceleration available")
        print("\n🔧 Recommendations for Apple Silicon Mac:")
        print("1. You have Apple M3 Pro - GPU acceleration is available via MPS!")
        print("2. Update PyTorch to latest version with MPS support:")
        print("   pip install --upgrade torch torchvision torchaudio")
        print("3. Verify MPS availability:")
        print("   python -c \"import torch; print(torch.backends.mps.is_available())\"")
        print("\n🔧 For NVIDIA systems:")
        print("1. Check if you have a CUDA-compatible GPU: nvidia-smi")
        print("2. Install CUDA drivers: https://developer.nvidia.com/cuda-downloads")
        print("3. Install PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
    print("\n" + "=" * 50)
    
    # System recommendations
    gpu_available = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    
    if device == "cuda":
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory < 4:
            print("⚠️  WARNING: Less than 4GB GPU memory. Consider:")
            print("   - Reducing batch size")
            print("   - Using gradient checkpointing")
            print("   - Testing simpler model variants first")
        elif total_memory >= 8:
            print("🚀 Great! 8GB+ GPU memory available")
            print("   - Can run full ablation study")
            print("   - Can use larger batch sizes")
            print("   - Multiple GPU training possible")
    elif device == "mps":
        print("🍎 Apple Silicon GPU acceleration enabled!")
        print("   - Faster than CPU for deep learning")
        print("   - Memory shared with system RAM")
        print("   - Optimal for research and development")
    else:
        print("💡 Running on CPU - consider upgrading for better performance")
    
    return gpu_available

def quick_benchmark():
    """Quick GPU vs CPU benchmark for both CUDA and MPS"""
    # Determine best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"  
    else:
        print("⚠️  No GPU acceleration available, skipping benchmark")
        return
        
    print(f"\n⚡ Quick Benchmark: {device.upper()} vs CPU")
    print("-" * 40)
    
    # Test sizes
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for size in sizes:
        print(f"\nMatrix size: {size[0]}x{size[1]}")
        
        # CPU test
        import time
        start = time.time()
        x_cpu = torch.randn(*size)
        y_cpu = torch.randn(*size)
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        # GPU test
        start = time.time()
        x_gpu = torch.randn(*size).to(device)
        y_gpu = torch.randn(*size).to(device)
        
        if device == "cuda":
            torch.cuda.synchronize()  # Ensure GPU ops complete
        
        z_gpu = torch.mm(x_gpu, y_gpu)
        
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()  # For MPS
            
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"  CPU time: {cpu_time*1000:.2f} ms")
        print(f"  {device.upper()} time: {gpu_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Cleanup
        del x_cpu, y_cpu, z_cpu, x_gpu, y_gpu, z_gpu

if __name__ == "__main__":
    gpu_available = check_gpu_setup()
    
    if gpu_available and "--benchmark" in sys.argv:
        quick_benchmark()
        
    # Determine device type for final message
    if torch.cuda.is_available():
        device_type = "CUDA GPU"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_type = "Apple Silicon MPS"
    else:
        device_type = "CPU only"
        
    print(f"\n🎯 Ready for ablation study: {'YES' if gpu_available else 'NO'} ({device_type})")
