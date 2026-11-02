"""Frame processing module for grayscale conversion and difference calculation."""

import numpy as np

from .decoder import FrameInfo


def grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame to grayscale numpy array.
    
    Args:
        frame: RGB frame as numpy array (HWC format, values 0-255)
        
    Returns:
        2D numpy array of float32 grayscale values (0-255)
    """
    # Standard luminance conversion: Y = 0.299*R + 0.587*G + 0.114*B
    gray = np.dot(frame[:, :, :3], [0.299, 0.587, 0.114]).astype(np.float32)
    return gray


def to_grayscale(frame_data: bytes, width: int, height: int) -> np.ndarray:
    """Convert raw frame data to grayscale numpy array.
    
    Args:
        frame_data: Raw bytes of frame data (gray or RGB24 format)
        width: Frame width in pixels
        height: Frame height in pixels
        
    Returns:
        2D numpy array of uint8 grayscale values (0-255)
    """
    # Determine bytes per pixel
    if len(frame_data) == width * height:
        # Already grayscale
        grayscale = np.frombuffer(frame_data, dtype=np.uint8)
    elif len(frame_data) == width * height * 3:
        # RGB24 format - convert to grayscale using weighted average
        rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
        # Standard luminance conversion: Y = 0.299*R + 0.587*G + 0.114*B
        grayscale = np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        raise ValueError(
            f"Frame data size {len(frame_data)} doesn't match expected "
            f"{width * height} (gray) or {width * height * 3} (RGB)"
        )
    
    return grayscale


def to_grayscale_np(frame: np.ndarray) -> np.ndarray:
    """Convert a numpy RGB frame to grayscale.
    
    Args:
        frame: RGB frame as numpy array (HWC format, values 0-255)
        
    Returns:
        2D numpy array of float32 grayscale values (0-255)
    """
    # Standard luminance conversion: Y = 0.299*R + 0.587*G + 0.114*B
    gray = np.dot(frame[:, :, :3], [0.299, 0.587, 0.114]).astype(np.float32)
    return gray


def frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate normalized frame difference as a metric of change.
    
    The difference is calculated as the mean absolute difference between
    corresponding pixels, normalized to [0, 1] range.
    
    Args:
        frame1: First grayscale frame as numpy array
        frame2: Second grayscale frame as numpy array
        
    Returns:
        Normalized difference score in range [0, 1]
    """
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes must match: {frame1.shape} vs {frame2.shape}")
    
    # Calculate mean absolute difference
    diff = np.abs(frame2.astype(np.int16) - frame1.astype(np.int16))
    mean_diff = np.mean(diff)
    
    # Normalize to [0, 1] range (255 is max possible difference for uint8)
    return float(mean_diff) / 255.0


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Alias for frame_difference - calculate normalized frame difference.
    
    Args:
        frame1: First grayscale frame as numpy array
        frame2: Second grayscale frame as numpy array
        
    Returns:
        Normalized difference score in range [0, 1]
    """
    return frame_difference(frame1, frame2)


def mean_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate mean frame difference (average absolute difference per pixel).
    
    Args:
        frame1: First grayscale frame as numpy array
        frame2: Second grayscale frame as numpy array
        
    Returns:
        Mean absolute difference score in range [0, 1]
    """
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes must match: {frame1.shape} vs {frame2.shape}")
    
    # Calculate mean absolute difference
    diff = np.abs(frame2.astype(np.int16) - frame1.astype(np.int16))
    mean_diff = np.mean(diff)
    
    # Normalize to [0, 1] range
    return float(mean_diff) / 255.0


def calculate_frame_info(width: int, height: int, gray: bool = True) -> FrameInfo:
    """Create a FrameInfo object for analysis frames.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        gray: Whether frames are grayscale
        
    Returns:
        FrameInfo with analysis parameters
    """
    return FrameInfo(
        width=width,
        height=height,
        timestamp=0.0,  # Will be updated by decoder
        frame_index=0,  # Will be updated by decoder
    )
