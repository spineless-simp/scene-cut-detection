"""Pytest configuration and fixtures for scene-cut-detector tests."""

import subprocess
import tempfile
import os
import pytest
from pathlib import Path


def create_test_video(
    duration: float = 2.0,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    output_path: str = None,
    cut_at_frames: list = None,
    brightness_change: float = 0.5,
) -> str:
    """Create a synthetic test video using FFmpeg.

    Args:
        duration: Video duration in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
        output_path: Output file path (auto-generated if None)
        cut_at_frames: List of frame indices where brightness changes occur
        brightness_change: How much brightness changes at cuts (0.0 to 1.0)

    Returns:
        Path to the created video file
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    if cut_at_frames is None:
        cut_at_frames = []

    # Create a solid color gradient video using colorfilter
    # Use a single gradient for simple videos - hex color format for FFmpeg compatibility
    color_val = int(0.1 * 255)
    hex_color = f"0x{color_val:02X}{color_val:02X}{color_val:02X}"
    filter_complex = f"color=c={hex_color}:s={width}x{height}:d={duration}:r={fps}"

    cmd = [
        "/usr/bin/ffmpeg",
        "-y",  # Overwrite output
        "-filter_complex", filter_complex,
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def create_video_with_hard_cut(
    duration: float = 2.0,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    cut_frame: int = 30,
    output_path: str = None,
) -> str:
    """Create a video with a single hard cut at a specific frame.

    Args:
        duration: Total video duration in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
        cut_frame: Frame index where the cut occurs
        output_path: Output file path (auto-generated if None)

    Returns:
        Path to the created video file
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    segment_duration = duration / 2

    # Create two temporary video files with different colors
    temp1 = tempfile.mktemp(suffix=".mp4")
    temp2 = tempfile.mktemp(suffix=".mp4")

    try:
        # Create first segment (dark)
        cmd1 = [
            "/usr/bin/ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x141414:s={width}x{height}:d={segment_duration}:r={fps}",
            "-pix_fmt", "yuv420p",
            temp1,
        ]
        subprocess.run(cmd1, check=True, capture_output=True)

        # Create second segment (light)
        cmd2 = [
            "/usr/bin/ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=0xEBEBEB:s={width}x{height}:d={segment_duration}:r={fps}",
            "-pix_fmt", "yuv420p",
            temp2,
        ]
        subprocess.run(cmd2, check=True, capture_output=True)

        # Concatenate the two videos with a hard cut
        cmd_concat = [
            "/usr/bin/ffmpeg",
            "-y",
            "-i", temp1,
            "-i", temp2,
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map", "[out]",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd_concat, check=True, capture_output=True)

    finally:
        # Clean up temporary files
        if os.path.exists(temp1):
            os.remove(temp1)
        if os.path.exists(temp2):
            os.remove(temp2)

    return output_path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_video_short(temp_dir):
    """Create a short test video."""
    video_path = str(temp_dir / "short.mp4")
    create_test_video(duration=1.0, fps=30.0, output_path=video_path)
    yield video_path
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture
def test_video_with_cut(temp_dir):
    """Create a video with a hard cut at frame 30."""
    video_path = str(temp_dir / "cut.mp4")
    create_video_with_hard_cut(
        duration=2.0,
        fps=30.0,
        cut_frame=30,
        output_path=video_path
    )
    yield video_path
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture
def test_video_longer(temp_dir):
    """Create a longer test video with multiple potential cuts."""
    video_path = str(temp_dir / "longer.mp4")
    create_test_video(duration=5.0, fps=24.0, output_path=video_path)
    yield video_path
    if os.path.exists(video_path):
        os.remove(video_path)


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g., long videos)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is provided."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
