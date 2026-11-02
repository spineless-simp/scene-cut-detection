"""Tests for the frame difference module."""

import numpy as np
from scene_cut_detector.frame_diff import grayscale, calculate_frame_difference


def test_grayscale_conversion():
    """Test grayscale conversion of RGB frame."""
    # Create a test RGB frame (HWC format, values 0-255)
    h, w = 100, 100
    rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_frame[:, :, 0] = 255  # Red
    rgb_frame[:, :, 1] = 128  # Green
    rgb_frame[:, :, 2] = 64   # Blue
    
    # Convert to grayscale
    gray = grayscale(rgb_frame)
    
    # Check shape
    assert gray.shape == (h, w)
    assert gray.dtype == np.float32
    
    # Check values are in valid range
    assert gray.min() >= 0
    assert gray.max() <= 255
    
    # Verify it's actually grayscale (all channels same)
    for i in range(h):
        for j in range(w):
            r, g, b = rgb_frame[i, j]
            expected = 0.299 * r + 0.587 * g + 0.114 * b
            assert np.isclose(gray[i, j], expected, atol=1.0)


def test_grayscale_white():
    """Test grayscale conversion of white frame."""
    h, w = 50, 50
    rgb_frame = np.full((h, w, 3), 255, dtype=np.uint8)
    gray = grayscale(rgb_frame)
    
    assert np.allclose(gray, 255.0)


def test_grayscale_black():
    """Test grayscale conversion of black frame."""
    h, w = 50, 50
    rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)
    gray = grayscale(rgb_frame)
    
    assert np.allclose(gray, 0.0)


def test_frame_difference_identical():
    """Test frame difference of identical frames is zero."""
    h, w = 100, 100
    frame1 = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    
    diff = calculate_frame_difference(frame1, frame2)
    
    # Difference should be essentially zero
    assert np.allclose(diff, 0.0, atol=1e-6)


def test_frame_difference_different():
    """Test frame difference of different frames is positive."""
    h, w = 100, 100
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2 = np.full((h, w, 3), 255, dtype=np.uint8)
    
    diff = calculate_frame_difference(frame1, frame2)
    
    # Difference should be normalized to [0, 1] range
    assert diff > 0.8


def test_frame_difference_black_white():
    """Test frame difference between black and white."""
    h, w = 100, 100
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2 = np.full((h, w, 3), 255, dtype=np.uint8)
    
    diff = calculate_frame_difference(frame1, frame2)
    
    # Maximum possible difference should be normalized to 1.0
    assert 0.9 < diff <= 1.0


def test_frame_difference_partial():
    """Test frame difference with partial change."""
    h, w = 100, 100
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[h//2:, w//2:, :] = 255  # Bottom-right quarter
    
    diff = calculate_frame_difference(frame1, frame2)
    
    # Difference should be in middle range
    assert 0.1 < diff < 0.9
