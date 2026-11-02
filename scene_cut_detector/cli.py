"""Command-line interface for scene cut detection."""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .decoder import VideoDecoder
from .detector import SceneCut, SceneCutDetector
from .frame_diff import to_grayscale
from .output import SceneCutReport, format_timestamp


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="scene-cut-detector",
        description="Detect hard scene cuts in video files using streaming processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                    # Default detection
  %(prog)s video.mp4 -t 0.5 -o cuts.csv # Custom threshold, CSV output
  %(prog)s video.mp4 -j                 # JSON output with pretty printing
  %(prog)s video.mp4 --min-sep 1.0      # Minimum 1 second between cuts
        """,
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Input video file path",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (CSV or JSON based on extension)",
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.3,
        help="Difference threshold for cut detection (0-1, default: 0.3)",
    )
    
    parser.add_argument(
        "--min-sep",
        type=float,
        default=0.5,
        dest="min_separation",
        help="Minimum seconds between cuts (default: 0.5)",
    )
    
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=160,
        help="Analysis frame width (default: 160)",
    )
    
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=90,
        dest="height",
        help="Analysis frame height (default: 90)",
    )
    
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output in JSON format (overrides output extension detection)",
    )
    
    parser.add_argument(
        "-p", "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    
    parser.add_argument(
        "--mean-diff",
        action="store_true",
        help="Use mean difference instead of max difference",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    return parser


def process_video(
    input_path: Path,
    output_path: Path | None,
    threshold: float,
    min_separation: float,
    width: int,
    height: int,
    use_mean_diff: bool,
    verbose: bool,
    json_output: bool = False,
    pretty: bool = False,
) -> list[SceneCut]:
    """Process a video file and detect scene cuts.
    
    Args:
        input_path: Path to input video
        output_path: Path for output file (None for console only)
        threshold: Difference threshold for detection
        min_separation: Minimum seconds between cuts
        width: Analysis frame width
        height: Analysis frame height
        use_mean_diff: Use mean difference instead of max
        verbose: Enable verbose progress output
        json_output: Force JSON output format
        pretty: Pretty-print JSON output
        
    Returns:
        List of detected SceneCut objects
    """
    # Initialize decoder
    decoder = VideoDecoder(
        input_path=str(input_path),
        analysis_width=width,
        analysis_height=height,
        gray=True,
    )
    
    # Initialize detector
    detector = SceneCutDetector(
        threshold=threshold,
        min_separation=min_separation,
        use_mean_diff=use_mean_diff,
    )
    
    cuts: list[SceneCut] = []
    frame_count = 0
    start_time = time.time()
    
    try:
        decoder.start()
        info = decoder.info()
        detector.set_video_info(info)
        
        if verbose:
            print(f"Video info: {info.width}x{info.height}", file=sys.stderr)
            if info.duration:
                print(f"Duration: {format_timestamp(info.duration)}", file=sys.stderr)
            if info.avg_frame_rate:
                print(f"FPS: {info.avg_frame_rate:.2f}", file=sys.stderr)
            print(f"Processing at {width}x{height} for analysis", file=sys.stderr)
            print(file=sys.stderr)
        
        # Process frames
        for frame_index, timestamp, frame_data in decoder.decode_frames():
            # Detect cuts
            new_cuts = detector.process_frame(
                frame_index, timestamp, frame_data, width, height
            )
            cuts.extend(new_cuts)
            
            frame_count += 1
            
            # Progress reporting
            if verbose and frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Calculate progress
                if info.duration and info.avg_frame_rate:
                    total_frames = info.duration * info.avg_frame_rate
                    progress = (frame_index + 1) / total_frames * 100
                    remaining = (total_frames - frame_index - 1) / fps if fps > 0 else 0
                    print(
                        f"\rProcessed {frame_index + 1} frames ({progress:.1f}%): "
                        f"{fps:.1f} FPS, {remaining:.1f}s remaining",
                        end="",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"\rProcessed {frame_count} frames at {fps:.1f} FPS",
                        end="",
                        file=sys.stderr,
                    )
        
        if verbose:
            print(file=sys.stderr)  # New line after progress
            print(file=sys.stderr)
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        if verbose:
            print(f"Processed {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS)", file=sys.stderr)
            print(f"Detected {len(cuts)} scene cuts", file=sys.stderr)
        
        # Output results
        if cuts or output_path:
            report = SceneCutReport(str(input_path), cuts)
            
            if output_path:
                # Determine output format
                use_json = json_output or output_path.suffix.lower() in (".json", ".js")
                
                if use_json:
                    with open(output_path, "w") as f:
                        f.write(report.to_json(pretty=pretty))
                    if verbose:
                        print(f"Results saved to {output_path} (JSON)", file=sys.stderr)
                else:
                    with open(output_path, "w", newline="") as f:
                        f.write(report.to_csv())
                    if verbose:
                        print(f"Results saved to {output_path} (CSV)", file=sys.stderr)
            
            # Print to stdout
            if not output_path:
                print(report.to_csv(), end="")
            elif verbose:
                print(report.to_csv(), end="")
        
    finally:
        decoder.stop()
    
    return cuts


def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    # Validate input file
    if not parsed.input.exists():
        print(f"Error: Input file not found: {parsed.input}", file=sys.stderr)
        return 1
    
    if not parsed.input.is_file():
        print(f"Error: Not a file: {parsed.input}", file=sys.stderr)
        return 1
    
    try:
        process_video(
            input_path=parsed.input,
            output_path=parsed.output,
            threshold=parsed.threshold,
            min_separation=parsed.min_separation,
            width=parsed.width,
            height=parsed.height,
            use_mean_diff=parsed.mean_diff,
            verbose=parsed.verbose,
            json_output=parsed.json,
            pretty=parsed.pretty,
        )
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if parsed.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
