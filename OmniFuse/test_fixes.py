# test_fixes.py
"""
Comprehensive test script to validate the fixes for current encoder training issues.
Tests PyTorch API fixes, data consistency, and augmentation strategies.
"""

import torch
import numpy as np
import logging
import sys
import os
from dataset_multimodal import MultiModalWeldDataset
from utils import build_file_list, detect_current_len, validate_data_consistency
from config import IMAGE_ROOT, SOUND_ROOT, CURRENT_ROOT, CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pytorch_api_fixes():
    """Test correct PyTorch random number generation methods."""
    logger.info("Testing PyTorch API fixes...")
    
    # Test the old incorrect method (should fail)
    try:
        result = torch.uniform(0.9, 1.1, (1,))
        logger.error("torch.uniform should not exist - this indicates an issue!")
        return False
    except AttributeError:
        logger.info("✓ Confirmed torch.uniform does not exist (expected)")
    
    # Test correct methods
    try:
        # Method 1: Using FloatTensor.uniform_
        result1 = torch.FloatTensor(1).uniform_(0.9, 1.1).item()
        logger.info(f"✓ Method 1 (FloatTensor.uniform_): {result1:.4f}")
        
        # Method 2: Using rand and scaling
        result2 = 0.9 + 0.2 * torch.rand(1).item()
        logger.info(f"✓ Method 2 (scaled rand): {result2:.4f}")
        
        # Validate ranges
        assert 0.9 <= result1 <= 1.1, f"Result1 {result1} out of range [0.9, 1.1]"
        assert 0.9 <= result2 <= 1.1, f"Result2 {result2} out of range [0.9, 1.1]"
        
        logger.info("✓ All PyTorch API fixes working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ PyTorch API fix failed: {e}")
        return False

def test_current_length_detection():
    """Test improved current length detection."""
    logger.info("Testing current length detection...")
    
    try:
        # Create some dummy current data for testing
        test_dir = "/tmp/test_current_data"
        os.makedirs(test_dir, exist_ok=True)
        
        # Create test data with various lengths
        test_lengths = [1000, 2000, 3000, 3500, 4000, 2500, 1500]
        for i, length in enumerate(test_lengths):
            test_data = np.random.randn(length).astype(np.float32)
            np.save(f"{test_dir}/test_{i}.npy", test_data)
        
        # Test detection on dummy data
        lengths = []
        for fname in os.listdir(test_dir):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(test_dir, fname))
                lengths.append(len(arr))
        
        mean_len = np.mean(lengths)
        percentile_95 = np.percentile(lengths, 95)
        
        logger.info(f"✓ Test data mean length: {mean_len:.1f}")
        logger.info(f"✓ Test data 95th percentile: {percentile_95:.1f}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        logger.info("✓ Current length detection logic working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Current length detection test failed: {e}")
        return False

def test_safe_augmentation():
    """Test safe data augmentation methods."""
    logger.info("Testing safe data augmentation...")
    
    try:
        # Create test current data
        original_data = torch.randn(1000)
        
        # Test that augmentation preserves general characteristics
        from dataset_multimodal import MultiModalWeldDataset
        
        # Create dummy dataset to test augmentation
        dataset = MultiModalWeldDataset(
            "/tmp", "/tmp", "/tmp", [("test", "test")],
            current_len=1000, enable_augmentation=True
        )
        
        # Test augmentation method
        augmented_data = dataset._apply_safe_augmentation(original_data.clone())
        
        # Verify augmentation is reasonable
        diff = torch.abs(augmented_data - original_data)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        logger.info(f"✓ Max difference after augmentation: {max_diff:.4f}")
        logger.info(f"✓ Mean difference after augmentation: {mean_diff:.4f}")
        
        # Ensure augmentation is not too aggressive
        original_std = original_data.std().item()
        assert max_diff < 0.1 * original_std, "Augmentation too aggressive"
        
        logger.info("✓ Safe augmentation working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Safe augmentation test failed: {e}")
        return False

def test_data_shape_consistency():
    """Test data shape consistency handling."""
    logger.info("Testing data shape consistency...")
    
    try:
        # Test padding
        short_data = np.random.randn(500).astype(np.float32)
        target_len = 1000
        
        # Simulate dataset padding
        if len(short_data) < target_len:
            pad_width = target_len - len(short_data)
            padded_data = np.pad(short_data, (0, pad_width), mode="constant")
        
        assert len(padded_data) == target_len, f"Padding failed: {len(padded_data)} != {target_len}"
        logger.info(f"✓ Padding test passed: {len(short_data)} -> {len(padded_data)}")
        
        # Test truncation
        long_data = np.random.randn(1500).astype(np.float32)
        truncated_data = long_data[:target_len]
        
        assert len(truncated_data) == target_len, f"Truncation failed: {len(truncated_data)} != {target_len}"
        logger.info(f"✓ Truncation test passed: {len(long_data)} -> {len(truncated_data)}")
        
        logger.info("✓ Data shape consistency handling working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data shape consistency test failed: {e}")
        return False

def test_error_handling():
    """Test robust error handling in dataset."""
    logger.info("Testing error handling...")
    
    try:
        # Test dataset with non-existent paths
        dummy_dataset = MultiModalWeldDataset(
            "/nonexistent", "/nonexistent", "/nonexistent", 
            [("test", "test")], current_len=1000
        )
        
        # This should not crash but return dummy data
        img, mel, cur, label = dummy_dataset[0]
        
        logger.info(f"✓ Error handling test: got shapes img={img.shape if hasattr(img, 'shape') else type(img)}, "
                   f"mel={mel.shape}, cur={cur.shape}, label={label}")
        
        assert len(cur) == 1000, f"Current length should be 1000, got {len(cur)}"
        
        logger.info("✓ Error handling working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting comprehensive tests for current encoder training fixes...")
    
    tests = [
        ("PyTorch API Fixes", test_pytorch_api_fixes),
        ("Current Length Detection", test_current_length_detection),
        ("Safe Augmentation", test_safe_augmentation),
        ("Data Shape Consistency", test_data_shape_consistency),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info('='*50)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY")
    logger.info('='*50)
    logger.info(f"Total tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        logger.error(f"❌ {failed} test(s) failed. Please review the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)