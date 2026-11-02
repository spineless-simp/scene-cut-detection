"""Tests for the output serialization module."""

import json
import tempfile
import os
from pathlib import Path

import pytest
from scene_cut_detector.output import SceneCutReport, CutInfo


def test_cut_info_creation():
    """Test CutInfo dataclass creation."""
    cut = CutInfo(
        frame_index=42,
        timestamp=1.5,
        frame_width=1920,
        frame_height=1080,
        frame_rate=30.0
    )
    
    assert cut.frame_index == 42
    assert cut.timestamp == 1.5
    assert cut.frame_width == 1920
    assert cut.frame_height == 1080
    assert cut.frame_rate == 30.0


def test_scene_cut_report_creation():
    """Test SceneCutReport creation."""
    cuts = [
        CutInfo(frame_index=0, timestamp=0.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
        CutInfo(frame_index=30, timestamp=1.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
    ]
    
    report = SceneCutReport(
        source_file="test.mp4",
        duration=2.0,
        frame_rate=30.0,
        cuts=cuts,
        analysis_width=640,
        threshold=0.5,
        min_separation=0.1
    )
    
    assert report.source_file == "test.mp4"
    assert report.duration == 2.0
    assert len(report.cuts) == 2


def test_to_csv_basic():
    """Test CSV output generation."""
    cuts = [
        CutInfo(frame_index=0, timestamp=0.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
        CutInfo(frame_index=30, timestamp=1.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
    ]
    
    report = SceneCutReport(
        source_file="test.mp4",
        duration=2.0,
        frame_rate=30.0,
        cuts=cuts,
        analysis_width=640,
        threshold=0.5,
        min_separation=0.1
    )
    
    csv_output = report.to_csv()
    
    # Check CSV contains expected fields
    assert "timestamp" in csv_output
    assert "frame_index" in csv_output
    assert "test.mp4" in csv_output
    
    # Check values
    assert "0.0" in csv_output
    assert "1.0" in csv_output


def test_to_csv_empty():
    """Test CSV output with no cuts."""
    report = SceneCutReport(
        source_file="test.mp4",
        duration=2.0,
        frame_rate=30.0,
        cuts=[],
        analysis_width=640,
        threshold=0.5,
        min_separation=0.1
    )
    
    csv_output = report.to_csv()
    
    # Should still have header
    assert "timestamp" in csv_output
    assert "frame_index" in csv_output


def test_to_json_basic():
    """Test JSON output generation."""
    cuts = [
        CutInfo(frame_index=0, timestamp=0.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
        CutInfo(frame_index=30, timestamp=1.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
    ]
    
    report = SceneCutReport(
        source_file="test.mp4",
        duration=2.0,
        frame_rate=30.0,
        cuts=cuts,
        analysis_width=640,
        threshold=0.5,
        min_separation=0.1
    )
    
    json_output = report.to_json()
    data = json.loads(json_output)
    
    assert data["source_file"] == "test.mp4"
    assert data["duration"] == 2.0
    assert len(data["cuts"]) == 2
    assert data["cuts"][0]["timestamp"] == 0.0
    assert data["cuts"][1]["timestamp"] == 1.0


def test_to_json_empty():
    """Test JSON output with no cuts."""
    report = SceneCutReport(
        source_file="test.mp4",
        duration=2.0,
        frame_rate=30.0,
        cuts=[],
        analysis_width=640,
        threshold=0.5,
        min_separation=0.1
    )
    
    json_output = report.to_json()
    data = json.loads(json_output)
    
    assert len(data["cuts"]) == 0


def test_to_csv_file(temp_dir):
    """Test CSV output to file."""
    cuts = [
        CutInfo(frame_index=0, timestamp=0.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
    ]
    
    report = SceneCutReport(
        source_file="test.mp4",
        duration=2.0,
        frame_rate=30.0,
        cuts=cuts,
        analysis_width=640,
        threshold=0.5,
        min_separation=0.1
    )
    
    output_path = temp_dir / "cuts.csv"
    report.to_csv_file(str(output_path))
    
    assert output_path.exists()
    
    # Verify file contents
    content = output_path.read_text()
    assert "timestamp" in content
    assert "0.0" in content


def test_to_json_file(temp_dir):
    """Test JSON output to file."""
    cuts = [
        CutInfo(frame_index=0, timestamp=0.0, frame_width=1920, frame_height=1080, frame_rate=30.0),
    ]
    
    report = SceneCutReport(
        source_file="test.mp4",
        duration=2.0,
        frame_rate=30.0,
        cuts=cuts,
        analysis_width=640,
        threshold=0.5,
        min_separation=0.1
    )
    
    output_path = temp_dir / "cuts.json"
    report.to_json_file(str(output_path))
    
    assert output_path.exists()
    
    # Verify file contents
    content = output_path.read_text()
    data = json.loads(content)
    assert data["source_file"] == "test.mp4"
    assert len(data["cuts"]) == 1
