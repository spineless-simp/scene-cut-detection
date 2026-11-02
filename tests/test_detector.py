"""Tests for the scene cut detector module."""

import pytest
import numpy as np
from scene_cut_detector.detector import SceneCutDetector


def test_detector_threshold():
    """Test that detector triggers on high difference."""
    detector = SceneCutDetector(threshold=0.5, min_separation=0)
    
    # Create identical frames (no cut)
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    
    is_cut = detector.process_frame(frame2, frame_index=0)
    assert not is_cut
    
    # Create different frames (cut)
    frame3 = np.full((100, 100, 3), 255, dtype=np.uint8)
    is_cut = detector.process_frame(frame3, frame_index=1)
    assert is_cut


def test_detector_with_min_separation():
    """Test minimum separation between cuts."""
    detector = SceneCutDetector(threshold=0.5, min_separation=2)
    
    # First frame - should NOT be a cut (no previous frame to compare to)
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    is_cut = detector.process_frame(frame1, frame_index=0)
    assert not is_cut  # First frame establishes baseline
    
    # Second frame with high difference - should be cut
    frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
    is_cut = detector.process_frame(frame2, frame_index=1)
    assert is_cut
    
    # Third frame with high difference - should NOT be cut (min_separation)
    is_cut = detector.process_frame(frame1, frame_index=2)
    assert not is_cut
    
    # Fourth frame - should also not be cut
    is_cut = detector.process_frame(frame2, frame_index=3)
    assert not is_cut
    
    # Fifth frame - should be a cut (min_separation passed)
    is_cut = detector.process_frame(frame1, frame_index=4)
    assert is_cut


def test_detector_below_threshold():
    """Test that small changes don't trigger cuts."""
    detector = SceneCutDetector(threshold=0.9, min_separation=0)
    
    # Create frames with small difference
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[0, 0, :] = 1  # Single pixel change
    
    is_cut = detector.process_frame(frame2, frame_index=0)
    assert not is_cut


def test_detector_sequential_frames():
    """Test detector with a sequence of frames."""
    detector = SceneCutDetector(threshold=0.5, min_separation=0)
    
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8),      # Frame 0
        np.zeros((100, 100, 3), dtype=np.uint8),      # Frame 1 (same)
        np.full((100, 100, 3), 255, dtype=np.uint8),  # Frame 2 (cut)
        np.full((100, 100, 3), 255, dtype=np.uint8),  # Frame 3 (same)
        np.zeros((100, 100, 3), dtype=np.uint8),      # Frame 4 (cut)
    ]
    
    cut_frames = []
    for i, frame in enumerate(frames):
        if detector.process_frame(frame, frame_index=i):
            cut_frames.append(i)
    
    # Should detect cuts at frames 2 and 4 (changes from previous frame)
    # Frame 0 is not a cut (no previous frame to compare to)
    assert 2 in cut_frames
    assert 4 in cut_frames


def test_detector_boundary_values():
    """Test detector with boundary threshold values."""
    # Test with very high threshold (should rarely trigger)
    high_threshold = SceneCutDetector(threshold=1.0, min_separation=0)
    
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
    
    # First frame (frame_index=0) should NOT be a cut (no previous frame)
    is_cut = high_threshold.process_frame(frame1, frame_index=0)
    assert not is_cut
    
    # Full black-to-white difference at threshold=1.0 should trigger
    is_cut = high_threshold.process_frame(frame2, frame_index=1)
    assert is_cut
    
    # Test with very low threshold (should always trigger on comparison)
    low_threshold = SceneCutDetector(threshold=0.01, min_separation=0)
    
    # Frame 0 with low threshold - not a cut (first frame)
    is_cut = low_threshold.process_frame(frame1, frame_index=0)
    assert not is_cut
    
    # Frame 1 (different from frame 0) should trigger
    is_cut = low_threshold.process_frame(frame2, frame_index=1)
    assert is_cut


def test_detector_reset():
    """Test detector reset functionality."""
    detector = SceneCutDetector(threshold=0.5, min_separation=0)
    
    # Process some frames
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
    
    detector.process_frame(frame1, frame_index=0)
    detector.process_frame(frame2, frame_index=1)
    
    # Reset
    detector.reset()
    
    # After reset, should process frames fresh
    # First frame (frame_index=0) should NOT be a cut (no previous frame)
    is_cut = detector.process_frame(frame1, frame_index=0)
    assert not is_cut
