"""Output serialization module for CSV and JSON scene cut reports."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any

from .detector import SceneCut


@dataclass
class CutInfo:
    """Information about a detected cut."""
    frame_index: int
    timestamp: float
    frame_width: int
    frame_height: int
    frame_rate: float


class SceneCutReport:
    """Generates scene cut reports in various formats."""
    
    def __init__(self, source_file: str, duration: float, frame_rate: float, cuts: list[CutInfo],
                 analysis_width: int | None = None, analysis_height: int | None = None,
                 threshold: float | None = None, min_separation: float | None = None):
        """Initialize the report generator.
        
        Args:
            source_file: Path to the source video file
            duration: Video duration in seconds
            frame_rate: Video frame rate (frames per second)
            cuts: List of CutInfo objects representing detected cuts
            analysis_width: Width used for analysis (if downscaling was applied)
            analysis_height: Height used for analysis (if downscaling was applied)
            threshold: Threshold value used for cut detection
            min_separation: Minimum separation between cuts (in seconds or frames)
        """
        self.source_file = source_file
        self.duration = duration
        self.frame_rate = frame_rate
        self.cuts = cuts
        self.analysis_width = analysis_width
        self.analysis_height = analysis_height
        self.threshold = threshold
        self.min_separation = min_separation
        self.detection_time = datetime.now().isoformat()
        
    def to_csv(self) -> str:
        """Generate CSV report.
        
        Returns:
            CSV formatted string with headers: source_file,duration,frame_rate,cut_count,frame_index,timestamp
        """
        output = StringIO()
        writer = csv.writer(output)
        
        # Metadata header
        writer.writerow(["source_file", "duration", "frame_rate", "cut_count"])
        writer.writerow([self.source_file, f"{self.duration:.6f}", f"{self.frame_rate:.6f}", len(self.cuts)])
        
        # Empty row for separator
        writer.writerow([])
        
        # Cuts header
        writer.writerow(["frame_index", "timestamp"])
        
        # Data rows
        for cut in self.cuts:
            writer.writerow([
                cut.frame_index,
                f"{cut.timestamp:.6f}",
            ])
        
        return output.getvalue()
    
    def to_json(self, pretty: bool = False) -> str:
        """Generate JSON report.
        
        Args:
            pretty: If True, format with indentation
            
        Returns:
            JSON formatted string
        """
        report = {
            "source_file": self.source_file,
            "duration": round(self.duration, 6),
            "frame_rate": round(self.frame_rate, 6),
            "detection_time": self.detection_time,
            "total_cuts": len(self.cuts),
            "cuts": [
                {
                    "frame_index": cut.frame_index,
                    "timestamp": round(cut.timestamp, 6),
                    "frame_width": cut.frame_width,
                    "frame_height": cut.frame_height,
                    "frame_rate": cut.frame_rate,
                }
                for cut in self.cuts
            ],
        }
        
        if pretty:
            return json.dumps(report, indent=2)
        return json.dumps(report)
    
    def to_csv_file(self, output_path: str) -> None:
        """Save report to a CSV file.
        
        Args:
            output_path: Path for output CSV file
        """
        with open(output_path, "w", newline="") as f:
            f.write(self.to_csv())
    
    def to_json_file(self, output_path: str, pretty: bool = True) -> None:
        """Save report to a JSON file.
        
        Args:
            output_path: Path for output JSON file
            pretty: If True, format with indentation
        """
        with open(output_path, "w") as f:
            f.write(self.to_json(pretty=pretty))
    
    def to_dict(self) -> dict[str, Any]:
        """Get report as a Python dictionary.
        
        Returns:
            Dictionary representation of the report
        """
        return {
            "source_file": self.source_file,
            "duration": self.duration,
            "frame_rate": self.frame_rate,
            "detection_time": self.detection_time,
            "total_cuts": len(self.cuts),
            "cuts": [
                {
                    "frame_index": cut.frame_index,
                    "timestamp": cut.timestamp,
                    "frame_width": cut.frame_width,
                    "frame_height": cut.frame_height,
                    "frame_rate": cut.frame_rate,
                }
                for cut in self.cuts
            ],
        }


def save_csv(report: SceneCutReport, output_path: str) -> None:
    """Save report to a CSV file.
    
    Args:
        report: SceneCutReport to save
        output_path: Path for output CSV file
    """
    with open(output_path, "w", newline="") as f:
        f.write(report.to_csv())


def save_json(report: SceneCutReport, output_path: str, pretty: bool = True) -> None:
    """Save report to a JSON file.
    
    Args:
        report: SceneCutReport to save
        output_path: Path for output JSON file
        pretty: If True, format with indentation
    """
    with open(output_path, "w") as f:
        f.write(report.to_json(pretty=pretty))


def format_timestamp(seconds: float) -> str:
    """Format timestamp as HH:MM:SS.mmm.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
