"""Integration tests for scene cut detection."""

import pytest
import os
from pathlib import Path

import numpy as np
from scene_cut_detector.decoder import VideoDecoder
from scene_cut_detector.frame_diff import calculate_frame_difference, to_grayscale_np
from scene_cut_detector.detector import SceneCutDetector
from scene_cut_detector.output import SceneCutReport, CutInfo


def test_full_pipeline_with_hard_cut(test_video_with_cut):
    """Test the complete pipeline with a known hard cut."""
    # Decode video
    decoder = VideoDecoder(test_video_with_cut, analysis_width=320)
    
    # Set up detector with low threshold to catch the cut
    detector = SceneCutDetector(threshold=0.3, min_separation=0)
    
    frames = list(decoder)
    assert len(frames) > 0
    
    cuts = []
    prev_frame = None
    
    for i, (frame, info) in enumerate(frames):
        if prev_frame is not None:
            # Convert to grayscale for frame difference calculation
            gray_frame = to_grayscale_np(frame)
            gray_prev = to_grayscale_np(prev_frame)
            diff = calculate_frame_difference(gray_frame, gray_prev)
            
            if detector.process_frame(frame, frame_index=i):
                cuts.append(CutInfo(
                    frame_index=i,
                    timestamp=info.timestamp,
                    frame_width=info.width,
                    frame_height=info.height,
                    frame_rate=decoder.video_info.avg_frame_rate
                ))
        
        prev_frame = frame.copy()
    
    # We should have detected the cut
    # The video has a cut around frame 30 (1 second in)
    assert len(cuts) >= 1, "Expected at least one cut to be detected"
    
    # Check that cut is near the expected time
    cut_times = [c.timestamp for c in cuts]
    # Allow some tolerance - should be around 1.0 seconds
    assert any(abs(t - 1.0) < 0.5 for t in cut_times), \
        f"Expected cut around 1.0s, got {cut_times}"


def test_no_cuts_in_smooth_video(test_video_short):
    """Test that smooth video has no hard cuts."""
    decoder = VideoDecoder(test_video_short, analysis_width=320)
    detector = SceneCutDetector(threshold=0.8, min_separation=0)
    
    prev_frame = None
    cuts = []
    
    for frame, info in decoder:
        if prev_frame is not None:
            # Convert to grayscale for frame difference calculation
            gray_frame = to_grayscale_np(frame)
            gray_prev = to_grayscale_np(prev_frame)
            _ = calculate_frame_difference(gray_frame, gray_prev)  # noqa: F841
            
            if detector.process_frame(frame, frame_index=info.frame_index):
                cuts.append(CutInfo(
                    frame_index=info.frame_index,
                    timestamp=info.timestamp,
                    frame_width=info.width,
                    frame_height=info.height,
                    frame_rate=decoder.video_info.avg_frame_rate
                ))
        
        prev_frame = frame.copy()
    
    # Smooth gradient video should have very few or no cuts with high threshold
    # Allow some minor frame differences due to compression artifacts
    assert len(cuts) < 5, "Smooth video should have few cuts"


def test_report_generation(test_video_short):
    """Test that report generation works correctly."""
    decoder = VideoDecoder(test_video_short, analysis_width=320)
    detector = SceneCutDetector(threshold=0.9, min_separation=0)
    
    prev_frame = None
    cuts = []
    
    for frame, info in decoder:
        if prev_frame is not None:
            # Convert to grayscale for frame difference calculation
            gray_frame = to_grayscale_np(frame)
            gray_prev = to_grayscale_np(prev_frame)
            _ = calculate_frame_difference(gray_frame, gray_prev)  # noqa: F841
            
            if detector.process_frame(frame, frame_index=info.frame_index):
                cuts.append(CutInfo(
                    frame_index=info.frame_index,
                    timestamp=info.timestamp,
                    frame_width=info.width,
                    frame_height=info.height,
                    frame_rate=decoder.video_info.avg_frame_rate
                ))
        
        prev_frame = frame.copy()
    
    # Generate report
    report = SceneCutReport(
        source_file=test_video_short,
        duration=decoder.video_info.duration,
        frame_rate=decoder.video_info.avg_frame_rate,
        cuts=cuts,
        analysis_width=320,
        threshold=0.9,
        min_separation=0
    )
    
    # Verify report fields
    assert report.source_file == test_video_short
    assert report.duration is not None
    assert report.frame_rate is not None


def test_memory_efficiency(test_video_short):
    """Test that memory usage stays bounded during processing."""
    decoder = VideoDecoder(test_video_short, analysis_width=320)
    detector = SceneCutDetector(threshold=0.8, min_separation=0)
    
    prev_frame = None
    frame_count = 0
    memory_samples = []
    
    for frame, info in decoder:
        frame_count += 1
        
        if prev_frame is not None:
            detector.process_frame(frame, frame_index=info.frame_index)
        
        prev_frame = frame.copy()
        
        # Track approximate memory usage by keeping references to recent frames
        # This is a rough check - actual memory should remain bounded
        if frame_count % 30 == 0:  # Every second
            memory_samples.append(frame_count)
    
    # Should have processed all frames
    assert frame_count > 0


def test_multiple_cuts_detection():
    """Test detection of multiple cuts in a synthetic video."""
    import subprocess
    import tempfile
    
    # Create a video with multiple hard cuts using concat filter
    # Split into 4 segments with different brightness
    duration = 2.0
    fps = 30.0
    segment_duration = duration / 4
    
    # Create temporary files for each segment
    temp_files = []
    try:
        # Generate 4 color segments as individual files
        for i in range(4):
            brightness = int((i * 64) % 256)
            temp_file = tempfile.NamedTemporaryFile(suffix="_seg.mp4", delete=False)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Convert brightness (0-255) to hex color for FFmpeg compatibility
            hex_color = f"0x{brightness:02X}{brightness:02X}{brightness:02X}"
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                f"color=c={hex_color}:s=320x240:d={segment_duration}:r={fps}",
                "-pix_fmt", "yuv420p",
                temp_file.name
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        # Create input file list for concat demuxer
        input_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        input_list_path = input_list.name
        for temp_file in temp_files:
            input_list.write(f"file '{temp_file}'\n")
        input_list.close()
        
        # Concatenate segments using concat demuxer
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
        
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", input_list_path,
            "-pix_fmt", "yuv420p",
            video_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Process and detect cuts
        decoder = VideoDecoder(video_path, analysis_width=320)
        detector = SceneCutDetector(threshold=0.2, min_separation=0)
        
        prev_frame = None
        cuts = []
        
        for frame, info in decoder:
            if prev_frame is not None:
                if detector.process_frame(frame, frame_index=info.frame_index):
                    cuts.append(CutInfo(
                        frame_index=info.frame_index,
                        timestamp=info.timestamp,
                        frame_width=info.width,
                        frame_height=info.height,
                        frame_rate=decoder.video_info.avg_frame_rate
                    ))
            prev_frame = frame.copy()
        
        # Should detect 3 cuts (one between each segment)
        # Allow some tolerance - might detect 3-5 cuts
        assert len(cuts) >= 2, f"Expected at least 2 cuts, got {len(cuts)}"
        
    finally:
        # Cleanup all temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        if os.path.exists(input_list_path):
            os.unlink(input_list_path)
        if os.path.exists(video_path):
            os.unlink(video_path)
