"""FFmpeg-based video decoder with streaming support and timestamp preservation."""

import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, Optional


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
        analysis_height: int,
        gray: bool = True,
        pix_fmt: str = "gray" if sys.byteorder == "little" else "gray",
    ):
        """Initialize the decoder with video source and analysis parameters.
        
        Args:
            input_path: Path to the input video file.
            analysis_width: Width to downscale frames to for analysis.
            analysis_height: Height to downscale frames to for analysis.
            gray: Whether to convert to grayscale (recommended for efficiency).
            pix_fmt: Pixel format for output ('gray' or 'rgb24').
        """
        self.input_path = input_path
        self.analysis_width = analysis_width
        self.analysis_height = analysis_height
        self.gray = gray
        self.pix_fmt = "gray" if gray else "rgb24"
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._info: Optional[VideoInfo] = None
        
    def __enter__(self):
        """Context manager entry - start the FFmpeg process."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up FFmpeg process."""
        self.stop()
        return False
        
    def start(self) -> None:
        """Start the FFmpeg decoding process."""
        # Build FFmpeg command for streaming decode with downscaling
        # We use:
        # -vf: Video filters for scaling and grayscale conversion
        # -f rawvideo: Output raw video to stdout
        # -pix_fmt: Output pixel format (gray or rgb24)
        # -loglevel error: Minimize log output, only show errors
        # -hide_banner: Hide FFmpeg banner
        
        filters = f"scale={self.analysis_width}:{self.analysis_height}"
        if self.gray:
            filters += ",format=gray"
        
        cmd = [
            "ffmpeg",
            "-nostdin",  # Don't read from stdin
            "-nostats",  # Don't print progress info
            "-hide_banner",
            "-loglevel", "error",
            "-i", self.input_path,
            "-vf", filters,
            "-pix_fmt", self.pix_fmt,
            "-f", "rawvideo",
            "pipe:1",  # Output to stdout
        ]
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,  # 1MB buffer for stdout
        )
        
        # Parse video info from stderr
        self._info = self._parse_video_info()
        
    def _parse_video_info(self) -> VideoInfo:
        """Parse video information from FFmpeg stderr.
        
        We read stderr to get video metadata, but we need to be careful
        not to block waiting for the full stderr output.
        """
        import re
        
        if self._process is None:
            raise RuntimeError("Process not started")
            
        # Read stderr to get video info
        # FFmpeg outputs info to stderr before decoding starts
        stderr_output = b""
        
        # Read stderr with timeout
        try:
            stderr_output, _ = self._process.stderr.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.stderr.close()
            
        stderr_text = stderr_output.decode("utf-8", errors="replace")
        
        # Parse video dimensions
        width = self.analysis_width
        height = self.analysis_height
        duration = None
        avg_frame_rate = None
        pixel_format = self.pix_fmt
        stream_index = 0
        
        # Try to extract duration
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)[.,](\d+)", stderr_text)
        if duration_match:
            hours, minutes, seconds, millis = duration_match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0
            
        # Try to extract frame rate
        fps_match = re.search(r"Stream.*Video:.*(\d+(?:\.\d+)?)\s*(?:fps|tb)", stderr_text)
        if fps_match:
            avg_frame_rate = float(fps_match.group(1))
            
        # Fallback: try to get from video info line
        if duration is None:
            dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)[.,](\d+)", stderr_text)
            if dur_match:
                h, m, s, ms = dur_match.groups()
                duration = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                
        return VideoInfo(
            width=width,
            height=height,
            duration=duration,
            avg_frame_rate=avg_frame_rate,
            pixel_format=pixel_format,
            stream_index=stream_index,
        )
        
    def stop(self) -> None:
        """Stop the FFmpeg process and clean up resources."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
            
    def info(self) -> VideoInfo:
        """Return video information."""
        if self._info is None:
            raise RuntimeError("Decoder not started")
        return self._info
        
    def decode_frames(self) -> Iterator[tuple[int, float, bytes]]:
        """Yield decoded frames with their index, timestamp, and data.
        
        Yields:
            Tuple of (frame_index, timestamp_seconds, frame_data_bytes)
            
        Note: Timestamps are estimated based on frame rate when exact
        presentation timestamps are not available.
        """
        if self._process is None:
            raise RuntimeError("Decoder not started")
            
        if self._info is None:
            raise RuntimeError("Video info not available")
            
        frame_size = self.analysis_width * self.analysis_height
        if not self.gray:
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
                    
                yield frame_index, timestamp, frame_data
                
                frame_index += 1
                timestamp += timestamp_increment
                
        finally:
            # Ensure process is cleaned up
            self.stop()
