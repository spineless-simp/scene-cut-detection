"""Scene cut detection module for identifying hard cuts in video streams."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .decoder import VideoInfo
from .frame_diff import frame_difference, mean_frame_difference, to_grayscale_np


@dataclass
class SceneCut:
    """Represents a detected scene cut.
    
    Attributes:
        frame_index: Zero-based index of the frame where the cut occurs
        timestamp: Timestamp in seconds when the cut occurs
        difference_score: The difference score that triggered detection
    """
    frame_index: int
    timestamp: float
    difference_score: float


class SceneCutDetector:
    """Detects hard scene cuts in video streams.
    
    Uses frame differencing to identify significant changes between
    consecutive frames that indicate scene boundaries.
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        min_separation: float = 0.5,
        use_mean_diff: bool = False,
    ):
        """Initialize the scene cut detector.
        
        Args:
            threshold: Difference score threshold for cut detection (0-1).
                      Higher values require more dramatic changes.
            min_separation: Minimum seconds between detected cuts.
                            Prevents duplicate detections from similar frames.
            use_mean_diff: If True, use mean frame difference instead of max.
        """
        self.threshold = threshold
        self.min_separation = min_separation
        self.use_mean_diff = use_mean_diff
        self._prev_frame: np.ndarray | None = None
        self._last_cut_frame: int = -1
        self._last_cut_timestamp: float = -1.0
        self._video_info: VideoInfo | None = None
        
    def set_video_info(self, info: VideoInfo) -> None:
        """Set video information for timestamp calculations.
        
        Args:
            info: VideoInfo from the decoder
        """
        self._video_info = info
        
    def reset(self) -> None:
        """Reset detector state for a new video."""
        self._prev_frame = None
        self._last_cut_frame = -1
        self._last_cut_timestamp = -1.0
        
    def process_frame(self, frame: np.ndarray, frame_index: int) -> bool:
        """Process a single frame and check for scene cuts.
        
        Args:
            frame: Frame as numpy array (HxWx3 uint8)
            frame_index: Zero-based frame index
            
        Returns:
            True if a scene cut was detected, False otherwise
        """
        # Convert to grayscale
        frame_gray = to_grayscale_np(frame)
        
        is_cut = False
        
        if self._prev_frame is None:
            # First frame - return False (no previous frame to compare to)
            self._prev_frame = frame_gray
            return False
        else:
            # Calculate difference
            if self.use_mean_diff:
                diff = mean_frame_difference(self._prev_frame, frame_gray)
            else:
                diff = frame_difference(self._prev_frame, frame_gray)
            
            # Check if this exceeds threshold
            if diff >= self.threshold:
                # Check minimum separation (in frame indices)
                last_cut_frame = self._last_cut_frame if self._last_cut_frame >= 0 else -1
                
                if last_cut_frame < 0 or (frame_index - last_cut_frame) > int(self.min_separation):
                    is_cut = True
                    self._last_cut_frame = frame_index
                    # Calculate timestamp for output
                    if self._video_info and self._video_info.avg_frame_rate:
                        self._last_cut_timestamp = frame_index / self._video_info.avg_frame_rate
                    else:
                        self._last_cut_timestamp = frame_index * 0.033333
        
        self._prev_frame = frame_gray
        return is_cut
    
    def process_stream(
        self, 
        frame_iterator: Iterator[tuple[int, float, bytes, int, int]],
    ) -> Iterator[tuple[int, float, bytes, int, int] | SceneCut]:
        """Process a stream of frames and yield cuts as they're detected.
        
        This generator yields frames as they're processed, and yields
        SceneCut objects when cuts are detected.
        
        Args:
            frame_iterator: Iterator yielding (frame_index, timestamp, frame_data, width, height)
            
        Yields:
            Either frame data or detected SceneCut objects
        """
        for frame_index, timestamp, frame_data, width, height in frame_iterator:
            cuts = self.process_frame(frame_index, timestamp, frame_data, width, height)
            
            # Yield the frame data first
            yield frame_index, timestamp, frame_data, width, height
            
            # Then yield any detected cuts
            for cut in cuts:
                yield cut
