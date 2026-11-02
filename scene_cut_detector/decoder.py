"""FFmpeg-based video decoder with streaming support and timestamp preservation."""

import subprocess
import sys
import threading
import re
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class FrameInfo:
    """Information about a decoded frame."""
    width: int
    height: int
    timestamp: float  # Presentation timestamp in seconds
    frame_index: int  # Zero-based frame index


@dataclass
class VideoInfo:
    """Information about the source video."""
    width: int
    height: int
    duration: Optional[float]  # Duration in seconds, None if unknown
    avg_frame_rate: Optional[float]  # Average FPS, None if variable rate
    pixel_format: str
    stream_index: int


class VideoDecoder:
    """Stream video frames from FFmpeg with timestamp preservation.
    
    Uses FFmpeg to decode video and output raw frames along with
    presentation timestamps via a metadata stream.
    """
    
    def __init__(
        self,
        input_path: str,
        analysis_width: int,
        analysis_height: Optional[int] = None,
        pix_fmt: str = "rgb24",
    ):
        """Initialize the video decoder.
        
        Args:
            input_path: Path to the input video file
            analysis_width: Width to downscale frames to for analysis
            analysis_height: Optional height to downscale frames to (defaults to None, maintains aspect ratio)
            pix_fmt: Pixel format for output ("gray" or "rgb24")
        """
        import os
        
        self.input_path = input_path
        self.analysis_width = analysis_width
        self.analysis_height = analysis_height
        self.pix_fmt = pix_fmt
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._info: Optional[VideoInfo] = None
        self._original_width: Optional[int] = None
        self._original_height: Optional[int] = None
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        import time
        import json
        
        # First, get video info without starting the full decoder
        self._original_width, self._original_height = self._get_original_dimensions(input_path)
        
        # Get duration and frame rate using ffprobe
        duration, avg_frame_rate = self._get_video_metadata(input_path)
        
        # Calculate the actual output dimensions
        if self.analysis_height is None and self._original_width and self._original_height:
            # Maintain aspect ratio
            aspect_ratio = self._original_height / self._original_width
            self.analysis_height = int(self.analysis_width * aspect_ratio)
        elif self.analysis_height is None:
            # Fallback default
            self.analysis_height = 240
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", f"scale={analysis_width}:{self.analysis_height}",
            "-pix_fmt", pix_fmt,
            "-f", "rawvideo",
            "-",
        ]
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,  # 1MB buffer for stdout
        )
        
        # Start thread to read stderr asynchronously
        self._stderr_output: list[bytes] = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        
        # Wait briefly for stderr to be populated
        time.sleep(0.1)
        
        # Parse video info (fill in any missing metadata from stderr)
        parsed_info = self._parse_video_info()
        
        # Use ffprobe values if stderr parsing didn't find them
        self._info = VideoInfo(
            width=self.analysis_width,
            height=self.analysis_height,
            duration=duration if duration else parsed_info.duration,
            avg_frame_rate=avg_frame_rate if avg_frame_rate else parsed_info.avg_frame_rate,
            pixel_format=pix_fmt,
            stream_index=0,
        )
        
    def _read_stderr(self) -> None:
        """Read stderr in a separate thread to avoid blocking."""
        if self._process and self._process.stderr:
            try:
                while True:
                    line = self._process.stderr.read(1024)
                    if not line:
                        break
                    with self._stderr_lock:
                        self._stderr_output.append(line)
            except Exception:
                pass
                
    def _get_original_dimensions(self, input_path: str) -> tuple[Optional[int], Optional[int]]:
        """Get original video dimensions using ffprobe."""
        import json
        
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            input_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data.get("streams", [{}])[0]
                width = stream.get("width")
                height = stream.get("height")
                return int(width), int(height)
        except Exception:
            pass
            
        # Fallback: use ffmpeg to get dimensions
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-v", "quiet",
            "-an",
            "-vframes", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"
        ]
        
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = proc.communicate(timeout=5)
            
            # Parse dimensions from stderr
            dim_match = re.search(r"Stream.*Video:.*(\\d+)x(\\d+)", stderr.decode("utf-8", errors="replace"))
            if dim_match:
                return int(dim_match.group(1)), int(dim_match.group(2))
        except Exception:
            pass
            
        return None, None
    
    def _get_video_metadata(self, input_path: str) -> tuple[Optional[float], Optional[float]]:
        """Get video duration and frame rate using ffprobe."""
        import json
        
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration,r_frame_rate",
            "-of", "json",
            input_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data.get("streams", [{}])[0]
                
                # Get duration
                duration_str = stream.get("duration")
                duration = float(duration_str) if duration_str else None
                
                # Get frame rate (r_frame_rate is usually more accurate for variable frame rate)
                frame_rate_str = stream.get("r_frame_rate")
                avg_frame_rate = None
                if frame_rate_str and "/" in frame_rate_str:
                    try:
                        num, denom = frame_rate_str.split("/")
                        avg_frame_rate = float(num) / float(denom)
                    except (ValueError, ZeroDivisionError):
                        pass
                
                return duration, avg_frame_rate
        except Exception:
            pass
            
        return None, None
    
    def _parse_video_info(self) -> VideoInfo:
        """Parse video information from FFmpeg stderr."""
        import re
        
        if self._process is None:
            raise RuntimeError("Process not started")
            
        # Get stderr text from collected output
        with self._stderr_lock:
            stderr_text = b"".join(self._stderr_output).decode("utf-8", errors="replace")
        
        # Parse video dimensions
        width = self.analysis_width
        height = self.analysis_height
        
        duration = None
        avg_frame_rate = None
        pixel_format = self.pix_fmt
        stream_index = 0
        
        # Try to extract duration - multiple patterns to try
        duration_patterns = [
            r"Duration:\\s*(\\d+):(\\d+):(\\d+)[.,](\\d+)",
            r"duration:\\s*(\\d+(?:\\.\\d+)?)",
        ]
        
        for pattern in duration_patterns:
            duration_match = re.search(pattern, stderr_text, re.IGNORECASE)
            if duration_match:
                if len(duration_match.groups()) == 4:
                    # Hours:Minutes:Seconds.millis format
                    hours, minutes, seconds, millis = duration_match.groups()
                    duration = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0
                else:
                    # Direct seconds format
                    duration = float(duration_match.group(1))
                break
                
        # Try to extract frame rate - multiple patterns
        fps_patterns = [
            r"(\\d+(?:\\.\\d+)?)\\s*(?:fps|fps,)",
            r"Stream.*Video:.*(\\d+(?:\\.\\d+)?)\\s*tbr",
            r"Stream.*Video:.*(\\d+(?:\\.\\d+)?)\\s*tb",
        ]
        
        for pattern in fps_patterns:
            fps_match = re.search(pattern, stderr_text, re.IGNORECASE)
            if fps_match:
                try:
                    avg_frame_rate = float(fps_match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
                    
        return VideoInfo(
            width=width,
            height=height,
            duration=duration,
            avg_frame_rate=avg_frame_rate,
            pixel_format=pixel_format,
            stream_index=stream_index,
        )
        
    @property
    def video_info(self) -> Optional[VideoInfo]:
        """Return video information."""
        return self._info
    
    def stop(self) -> None:
        """Stop the FFmpeg process and clean up resources."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None
            
    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameInfo]]:
        """Return an iterator over frames.
        
        Yields:
            Tuple of (frame: np.ndarray, info: FrameInfo)
        """
        if self._process is None:
            raise RuntimeError("Decoder not started")
            
        if self._info is None:
            raise RuntimeError("Video info not available")
            
        frame_size = self.analysis_width * self.analysis_height
        if self.pix_fmt != "gray":
            frame_size *= 3  # RGB24: 3 bytes per pixel
            
        info = self._info
        frame_index = 0
        timestamp = 0.0
        
        # Calculate timestamp increment based on frame rate
        if info.avg_frame_rate and info.avg_frame_rate > 0:
            timestamp_increment = 1.0 / info.avg_frame_rate
        else:
            timestamp_increment = 0.033333  # Assume ~30 fps if unknown
            
        try:
            while True:
                # Read one frame
                frame_data = self._process.stdout.read(frame_size)
                
                if len(frame_data) < frame_size:
                    # End of stream or error
                    break
                    
                # Convert bytes to numpy array
                if self.pix_fmt == "gray":
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.analysis_height, self.analysis_width))
                else:
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.analysis_height, self.analysis_width, 3))
                
                frame_info = FrameInfo(
                    width=self.analysis_width,
                    height=self.analysis_height,
                    timestamp=timestamp,
                    frame_index=frame_index,
                )
                
                yield frame, frame_info
                
                frame_index += 1
                timestamp += timestamp_increment
                
        finally:
            # Ensure process is cleaned up
            self.stop()
    
    def __next__(self) -> tuple[np.ndarray, FrameInfo]:
        """Return the next frame from the iterator.
        
        Returns:
            Tuple of (frame: np.ndarray, info: FrameInfo)
        """
        # Create iterator on first call
        if not hasattr(self, '_iterator'):
            self._iterator = self.__iter__()
        
        return next(self._iterator)
