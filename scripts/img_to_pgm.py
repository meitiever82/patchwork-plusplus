#!/usr/bin/env python3
"""
PNG to PGM Grid Map Converter
Converts PNG format grid maps to PGM format with proper coordinate system handling.
Compatible with ROS map_server format.
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple, Optional


class GridMapParams:
    def __init__(self, resolution: float = 0.1, origin_x: float = 0.0, origin_y: float = 0.0,
                 use_custom_origin: bool = False, ground_value: int = 0, 
                 unknown_value: int = 100, obstacle_value: int = 200):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.use_custom_origin = use_custom_origin
        self.ground_value = ground_value
        self.unknown_value = unknown_value
        self.obstacle_value = obstacle_value
        self.map_width = 0.0
        self.map_height = 0.0
        self.center_x = 0.0
        self.center_y = 0.0


def png_to_grid_values(png_image: np.ndarray, params: GridMapParams) -> np.ndarray:
    """
    Convert PNG image values to grid map values.
    Assumes: 0 (black) = ground, 127 (gray) = unknown, 255 (white) = obstacle
    """
    grid_map = np.zeros_like(png_image, dtype=np.uint8)
    
    # Convert visualization values back to grid values
    grid_map[png_image == 0] = params.ground_value      # black -> ground (0)
    grid_map[png_image == 127] = params.unknown_value   # gray -> unknown (100)  
    grid_map[png_image == 255] = params.obstacle_value  # white -> obstacle (200)
    
    # Handle other values by mapping to closest category
    mask_other = (png_image != 0) & (png_image != 127) & (png_image != 255)
    if np.any(mask_other):
        print(f"Warning: Found {np.sum(mask_other)} pixels with non-standard values, mapping to closest category")
        # Map based on brightness
        grid_map[mask_other & (png_image < 64)] = params.ground_value
        grid_map[mask_other & (png_image >= 64) & (png_image < 192)] = params.unknown_value
        grid_map[mask_other & (png_image >= 192)] = params.obstacle_value
    
    return grid_map


def save_pgm_file(grid_map: np.ndarray, pgm_path: str, params: GridMapParams) -> None:
    """Save grid map as PGM file with proper coordinate system."""
    
    with open(pgm_path, 'w') as f:
        # PGM header (ASCII format P2)
        f.write("P2\n")
        f.write("# Grid map converted from PNG\n")
        f.write(f"# Resolution: {params.resolution} m/pixel\n")
        
        if params.use_custom_origin:
            f.write(f"# Custom origin alignment: grid (0,0) = world ({params.origin_x}, {params.origin_y})\n")
            f.write(f"# Map center: ({params.center_x}, {params.center_y})\n")
            
            # Calculate world coverage
            world_x_min = params.origin_x
            world_y_min = params.origin_y
            world_x_max = params.origin_x + (grid_map.shape[1] - 1) * params.resolution
            world_y_max = params.origin_y + (grid_map.shape[0] - 1) * params.resolution
            f.write(f"# World coverage: [{world_x_min}, {world_x_max}] x [{world_y_min}, {world_y_max}]\n")
        else:
            f.write("# Auto-calculated parameters (data-centered)\n")
            f.write(f"# Map center: ({params.center_x}, {params.center_y})\n")
            grid_origin_x = params.center_x - params.map_width / 2.0
            grid_origin_y = params.center_y - params.map_height / 2.0
            f.write(f"# Grid origin: ({grid_origin_x}, {grid_origin_y})\n")
        
        f.write(f"# Map size: {params.map_width}m x {params.map_height}m\n")
        f.write("# Coordinate system: origin at bottom-left, x->right, y->up\n")
        f.write(f"# Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")
        f.write(f"{grid_map.shape[1]} {grid_map.shape[0]}\n")
        f.write("255\n")  # max value
        
        # Write data with coordinate system conversion
        # PGM format starts from top-left, but our coordinate system has origin at bottom-left
        # So we need to flip y-axis for output
        for row in range(grid_map.shape[0] - 1, -1, -1):  # From bottom to top
            row_values = []
            for col in range(grid_map.shape[1]):
                row_values.append(str(grid_map[row, col]))
            f.write(" ".join(row_values) + "\n")


def save_yaml_file(yaml_path: str, pgm_filename: str, params: GridMapParams) -> None:
    """Save map metadata as YAML file."""
    
    with open(yaml_path, 'w') as f:
        f.write(f"image: {pgm_filename}\n")
        f.write(f"resolution: {params.resolution}\n")
        
        # Set origin based on custom origin setting
        if params.use_custom_origin:
            f.write(f"origin: [{params.origin_x}, {params.origin_y}, 0.0]\n")
            f.write(f"# Custom origin alignment: grid map (0,0) = world ({params.origin_x}, {params.origin_y})\n")
        else:
            origin_x = params.center_x - params.map_width / 2.0
            origin_y = params.center_y - params.map_height / 2.0
            f.write(f"origin: [{origin_x}, {origin_y}, 0.0]\n")
            f.write(f"# Auto-calculated origin: grid map (0,0) = world ({origin_x}, {origin_y})\n")
        
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
        f.write("# Grid map coordinate system:\n")
        f.write("# - Origin at bottom-left corner\n")
        f.write("# - X-axis points right\n")
        f.write("# - Y-axis points up\n")
        f.write(f"# - Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")


def convert_png_to_pgm(png_path: str, output_path: str, params: GridMapParams) -> None:
    """Main conversion function."""
    
    # Load PNG image
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG file not found: {png_path}")
    
    # Read as grayscale
    png_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if png_image is None:
        raise ValueError(f"Cannot read PNG image: {png_path}")
    
    print(f"Loaded PNG image: {png_image.shape[1]}x{png_image.shape[0]} pixels")
    
    # Calculate map dimensions
    params.map_width = png_image.shape[1] * params.resolution
    params.map_height = png_image.shape[0] * params.resolution
    
    # Set center coordinates (if not using custom origin)
    if not params.use_custom_origin:
        params.center_x = params.map_width / 2.0
        params.center_y = params.map_height / 2.0
    
    # Convert PNG values to grid map values
    grid_map = png_to_grid_values(png_image, params)
    
    # Flip image vertically since PNG might be stored with top-left origin
    # but we want bottom-left origin coordinate system
    grid_map = cv2.flip(grid_map, 0)
    
    # Generate output filenames
    base_name = os.path.splitext(output_path)[0]
    pgm_path = base_name + "_gridmap.pgm"
    yaml_path = base_name + "_gridmap.yaml"
    pgm_filename = os.path.basename(pgm_path)
    
    # Save PGM file
    save_pgm_file(grid_map, pgm_path, params)
    print(f"Grid map saved as PGM: {pgm_path}")
    
    # Save YAML metadata
    save_yaml_file(yaml_path, pgm_filename, params)
    print(f"Map metadata saved as YAML: {yaml_path}")
    
    # Print statistics
    total_pixels = grid_map.size
    ground_pixels = np.sum(grid_map == params.ground_value)
    unknown_pixels = np.sum(grid_map == params.unknown_value)
    obstacle_pixels = np.sum(grid_map == params.obstacle_value)
    
    print(f"\nConversion statistics:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Ground pixels: {ground_pixels} ({100*ground_pixels/total_pixels:.1f}%)")
    print(f"  Unknown pixels: {unknown_pixels} ({100*unknown_pixels/total_pixels:.1f}%)")
    print(f"  Obstacle pixels: {obstacle_pixels} ({100*obstacle_pixels/total_pixels:.1f}%)")
    print(f"  Map size: {params.map_width:.2f}m x {params.map_height:.2f}m")


def main():
    # python png_to_pgm.py input_map.png output_map
    # 指定分辨率和自定义原点
    # python png_to_pgm.py input_map.png output_map -r 0.1 --origin-x -10 --origin-y -5 --use-custom-origin
    # 自定义像素值映射
    # python png_to_pgm.py input_map.png output_map --ground-value 0 --unknown-value 128 --obstacle-value 255
    parser = argparse.ArgumentParser(description="Convert PNG grid map to PGM format")
    parser.add_argument("input_png", help="Input PNG file path")
    parser.add_argument("output_path", help="Output file path (without extension)")
    parser.add_argument("--resolution", "-r", type=float, default=0.1, 
                       help="Map resolution in meters per pixel (default: 0.1)")
    parser.add_argument("--origin-x", type=float, default=0.0,
                       help="Custom origin X coordinate (default: 0.0)")
    parser.add_argument("--origin-y", type=float, default=0.0,
                       help="Custom origin Y coordinate (default: 0.0)")
    parser.add_argument("--use-custom-origin", action="store_true",
                       help="Use custom origin instead of auto-calculated center")
    parser.add_argument("--ground-value", type=int, default=0,
                       help="Grid value for ground/free space (default: 0)")
    parser.add_argument("--unknown-value", type=int, default=100,
                       help="Grid value for unknown space (default: 100)")
    parser.add_argument("--obstacle-value", type=int, default=200,
                       help="Grid value for obstacles (default: 200)")
    
    args = parser.parse_args()
    
    # Create parameters object
    params = GridMapParams(
        resolution=args.resolution,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        use_custom_origin=args.use_custom_origin,
        ground_value=args.ground_value,
        unknown_value=args.unknown_value,
        obstacle_value=args.obstacle_value
    )
    
    try:
        convert_png_to_pgm(args.input_png, args.output_path, params)
        print("\nConversion completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
