"""Tests for the decoder module."""

import pytest
from scene_cut_detector.decoder import VideoDecoder, VideoInfo, FrameInfo


def test_decoder_basic(test_video_short):
    """Test basic video decoding functionality."""
    decoder = VideoDecoder(test_video_short, analysis_width=320)
    
    # Check video info
    assert decoder.video_info is not None
    assert decoder.video_info.width > 0
    assert decoder.video_info.height > 0
    assert decoder.video_info.duration is not None
    
    # Check we can decode at least one frame
    frames = list(decoder)
    assert len(frames) > 0
    
    frame, info = frames[0]
    assert frame is not None
    assert info.width > 0
    assert info.height > 0
    assert info.timestamp >= 0
    assert info.frame_index == 0


def test_decoder_frame_rate(test_video_short):
    """Test that frame rate is correctly detected."""
    decoder = VideoDecoder(test_video_short, analysis_width=320)
    
    fps = decoder.video_info.avg_frame_rate
    assert fps is not None
    assert fps > 0
    
    # Check total frames
    frames = list(decoder)
    expected_frames = int(decoder.video_info.duration * fps)
    
    # Allow some tolerance for frame counting
    assert len(frames) >= expected_frames * 0.9


def test_decoder_timestamps(test_video_short):
    """Test that timestamps are correctly recorded."""
    decoder = VideoDecoder(test_video_short, analysis_width=320)
    
    frames = list(decoder)
    timestamps = [info.timestamp for _, info in frames]
    
    # Timestamps should be monotonically increasing
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i-1]
    
    # First timestamp should be 0
    assert timestamps[0] == 0.0
    
    # Last timestamp should be close to duration
    if decoder.video_info.duration is not None:
        assert timestamps[-1] <= decoder.video_info.duration


def test_decoder_downscaling(test_video_short):
    """Test that downscaling works correctly."""
    analysis_width = 160
    decoder = VideoDecoder(test_video_short, analysis_width=analysis_width)
    
    frame, info = next(iter(decoder))
    
    # Check that frame is downscaled to analysis_width
    assert info.width == analysis_width
    # Height should maintain aspect ratio
    expected_height = int(240 * (analysis_width / 320))
    assert info.height == expected_height


def test_decoder_invalid_file():
    """Test handling of invalid file."""
    with pytest.raises(FileNotFoundError):
        VideoDecoder("/nonexistent/file.mp4", analysis_width=320)
