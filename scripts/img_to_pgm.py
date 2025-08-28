#!/usr/bin/env python3
"""
PNG to PGM Grid Map Converter (Fixed Version)
Converts PNG format grid maps to PGM format with proper world coordinate handling.
Compatible with ROS map_server format and vSLAM coordinate systems.
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple, Optional


class GridMapParams:
    def __init__(self, resolution: float = 0.1, ground_value: int = 0, 
                 unknown_value: int = 200, obstacle_value: int = 100):
        self.resolution = resolution
        self.ground_value = ground_value
        self.unknown_value = unknown_value
        self.obstacle_value = obstacle_value
        
        # World coordinate bounds (to be set from CloudCompare data)
        self.world_x_min = 0.0
        self.world_x_max = 0.0
        self.world_y_min = 0.0
        self.world_y_max = 0.0
        
        # Derived parameters
        self.map_width = 0.0
        self.map_height = 0.0


def png_to_grid_values(png_image: np.ndarray, params: GridMapParams) -> np.ndarray:
    """
    Convert PNG image values to grid map values.
    Assumes: 0 (black) = ground, 127 (gray) = unknown, 255 (white) = obstacle
    """
    grid_map = np.zeros_like(png_image, dtype=np.uint8)
    
    # Convert visualization values back to grid values
    grid_map[png_image == 0] = params.ground_value      # black -> ground (0)
    grid_map[png_image == 127] = params.unknown_value   # gray -> unknown (200)  
    grid_map[png_image == 255] = params.obstacle_value  # white -> obstacle (100)
    
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
    
    # Calculate map center and grid origin (matching C++ program format)
    map_center_x = (params.world_x_min + params.world_x_max) / 2.0
    map_center_y = (params.world_y_min + params.world_y_max) / 2.0
    grid_origin_x = params.world_x_min
    grid_origin_y = params.world_y_min
    
    with open(pgm_path, 'w') as f:
        # PGM header (ASCII format P2)
        f.write("P2\n")
        f.write("# Grid map converted from PNG\n")
        f.write(f"# Resolution: {params.resolution} m/pixel\n")
        f.write(f"# Map center: ({map_center_x}, {map_center_y})\n")
        f.write(f"# Grid origin: ({grid_origin_x}, {grid_origin_y})\n")
        f.write(f"# Map size: {params.map_width:.3f}m x {params.map_height:.3f}m\n")
        f.write("# Coordinate system: same as input PNG (top-left origin)\n")
        f.write(f"# Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")
        f.write(f"{grid_map.shape[1]} {grid_map.shape[0]}\n")
        f.write("255\n")  # max value
        
        # Write data in original order (no coordinate system conversion)
        # Keep the same orientation as the input PNG
        for row in range(grid_map.shape[0]):  # From top to bottom
            row_values = []
            for col in range(grid_map.shape[1]):
                row_values.append(str(grid_map[row, col]))
            f.write(" ".join(row_values) + "\n")


def save_yaml_file(yaml_path: str, pgm_filename: str, params: GridMapParams) -> None:
    """Save map metadata as YAML file with correct world coordinates."""
    
    with open(yaml_path, 'w') as f:
        f.write(f"image: {pgm_filename}\n")
        f.write(f"resolution: {params.resolution}\n")
        
        # Grid map origin is at world_x_min, world_y_min
        # This is where grid coordinate (0,0) maps to in world coordinates
        f.write(f"origin: [{params.world_x_min}, {params.world_y_min}, 0.0]\n")
        
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
        
        # Add coordinate transform information for vSLAM system
        f.write("# Coordinate Transform Information\n")
        f.write("pcl_to_grid_transform:\n")
        f.write(f"  resolution: {params.resolution}\n")
        f.write(f"  grid_origin_world_x: {params.world_x_min}\n")
        f.write(f"  grid_origin_world_y: {params.world_y_min}\n")
        f.write(f"  world_bounds: [{params.world_x_min}, {params.world_x_max}, {params.world_y_min}, {params.world_y_max}]\n")
        f.write("  coordinate_system: top_left_origin\n")
        
        # Grid map coordinate system explanation
        f.write("# Grid map coordinate system:\n")
        f.write("# - Origin at top-left corner (same as PNG)\n")
        f.write("# - X-axis points right\n")
        f.write("# - Y-axis points down\n")
        f.write(f"# - Grid (0,0) = World ({params.world_x_min}, {params.world_y_max:.3f})\n")
        f.write(f"# - Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")


def convert_png_to_pgm(png_path: str, output_path: str, params: GridMapParams,
                      world_x_min: float, world_x_max: float,
                      world_y_min: float, world_y_max: float) -> None:
    """Main conversion function with world coordinate bounds."""
    
    # Load PNG image
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG file not found: {png_path}")
    
    # Read as grayscale
    png_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if png_image is None:
        raise ValueError(f"Cannot read PNG image: {png_path}")
    
    print(f"Loaded PNG image: {png_image.shape[1]}x{png_image.shape[0]} pixels")
    
    # Set world coordinate bounds
    params.world_x_min = world_x_min
    params.world_x_max = world_x_max
    params.world_y_min = world_y_min
    params.world_y_max = world_y_max
    
    # Calculate map dimensions from world coordinates
    params.map_width = world_x_max - world_x_min
    params.map_height = world_y_max - world_y_min
    
    print(f"World coordinate bounds: [{world_x_min}, {world_x_max}] x [{world_y_min}, {world_y_max}]")
    print(f"Map size: {params.map_width:.3f}m x {params.map_height:.3f}m")
    
    # Verify image size consistency
    expected_width = int(params.map_width / params.resolution)
    expected_height = int(params.map_height / params.resolution)
    
    print(f"Expected grid size: {expected_width}x{expected_height} pixels")
    print(f"Actual image size: {png_image.shape[1]}x{png_image.shape[0]} pixels")
    
    if png_image.shape[1] != expected_width or png_image.shape[0] != expected_height:
        print("Warning: Image size doesn't exactly match world bounds and resolution!")
        print("This may cause coordinate alignment issues.")
        
        # Option to adjust resolution to match
        actual_res_x = params.map_width / png_image.shape[1]
        actual_res_y = params.map_height / png_image.shape[0]
        print(f"Implied resolution: X={actual_res_x:.6f}, Y={actual_res_y:.6f}")
        
        if abs(actual_res_x - actual_res_y) < 0.001:  # Square pixels
            suggested_resolution = (actual_res_x + actual_res_y) / 2
            print(f"Suggested resolution: {suggested_resolution:.6f}")
    
    # Convert PNG values to grid map values
    grid_map = png_to_grid_values(png_image, params)
    
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


def main():
    parser = argparse.ArgumentParser(description="Convert PNG grid map to PGM format with world coordinates")
    parser.add_argument("input_png", help="Input PNG file path")
    parser.add_argument("output_path", help="Output file path (without extension)")
    
    # World coordinate bounds (required)
    parser.add_argument("--world-x-min", type=float, required=True,
                       help="World coordinate X minimum (from CloudCompare)")
    parser.add_argument("--world-x-max", type=float, required=True,
                       help="World coordinate X maximum (from CloudCompare)")
    parser.add_argument("--world-y-min", type=float, required=True,
                       help="World coordinate Y minimum (from CloudCompare)")
    parser.add_argument("--world-y-max", type=float, required=True,
                       help="World coordinate Y maximum (from CloudCompare)")
    
    # Grid map parameters
    parser.add_argument("--resolution", "-r", type=float, required=True,
                       help="Map resolution in meters per pixel (from CloudCompare step)")
    parser.add_argument("--ground-value", type=int, default=0,
                       help="Grid value for ground/free space (default: 0)")
    parser.add_argument("--unknown-value", type=int, default=200,
                       help="Grid value for unknown space (default: 200)")
    parser.add_argument("--obstacle-value", type=int, default=100,
                       help="Grid value for obstacles (default: 100)")
    
    parser.add_argument("--grid-origin-x", type=float, required=True,
                   help="World coordinate where grid (0,0) maps to - X")
    parser.add_argument("--grid-origin-y", type=float, required=True, 
                   help="World coordinate where grid (0,0) maps to - Y")

    args = parser.parse_args()
    
    # Create parameters object
    params = GridMapParams(
        resolution=args.resolution,
        ground_value=args.ground_value,
        unknown_value=args.unknown_value,
        obstacle_value=args.obstacle_value
    )
    
    try:
        convert_png_to_pgm(
            args.input_png, 
            args.output_path, 
            params,
            args.world_x_min,
            args.world_x_max, 
            args.world_y_min,
            args.world_y_max
        )
        print("\nConversion completed successfully!")
        print("\nUsage example for your CloudCompare data:")
        print(f"python {os.path.basename(__file__)} gridmap.png output \\")
        print(f"    --resolution 0.1 \\")
        print(f"    --world-x-min -10.7 --world-x-max 13.7 \\")
        print(f"    --world-y-min -5.19 --world-y-max 25.5236")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


def save_pgm_file(grid_map: np.ndarray, pgm_path: str, params: GridMapParams) -> None:
    """Save grid map as PGM file with proper coordinate system."""
    
    # Calculate map center and grid origin (matching C++ program format)
    map_center_x = (params.world_x_min + params.world_x_max) / 2.0
    map_center_y = (params.world_y_min + params.world_y_max) / 2.0
    grid_origin_x = params.world_x_min
    grid_origin_y = params.world_y_min
    
    with open(pgm_path, 'w') as f:
        # PGM header (ASCII format P2) - matching C++ format
        f.write("P2\n")
        f.write("# Grid map converted from PNG\n")
        f.write(f"# Resolution: {params.resolution} m/pixel\n")
        f.write(f"# Map center: ({map_center_x}, {map_center_y})\n")
        f.write(f"# Grid origin: ({grid_origin_x}, {grid_origin_y})\n")
        f.write(f"# Map size: {params.map_width:.3f}m x {params.map_height:.3f}m\n")
        f.write("# Coordinate system: origin at bottom-left, x->right, y->up\n")
        f.write(f"# Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")
        f.write(f"{grid_map.shape[1]} {grid_map.shape[0]}\n")
        f.write("255\n")  # max value
        
        # Write data with coordinate system conversion (matching C++ program)
        # PGM format starts from top-left, but our coordinate system has origin at bottom-left
        # So we need to flip y-axis for output
        for row in range(grid_map.shape[0] - 1, -1, -1):  # From bottom to top
            row_values = []
            for col in range(grid_map.shape[1]):
                row_values.append(str(grid_map[row, col]))
            f.write(" ".join(row_values) + "\n")


def save_yaml_file(yaml_path: str, pgm_filename: str, params: GridMapParams) -> None:
    """Save map metadata as YAML file with correct world coordinates."""
    
    with open(yaml_path, 'w') as f:
        f.write(f"image: {pgm_filename}\n")
        f.write(f"resolution: {params.resolution}\n")
        
        # Grid map origin is at world_x_min, world_y_min
        # This is where grid coordinate (0,0) maps to in world coordinates
        f.write(f"origin: [{params.world_x_min}, {params.world_y_min}, 0.0]\n")
        
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
        
        # Add coordinate transform information for vSLAM system
        f.write("# Coordinate Transform Information\n")
        f.write("pcl_to_grid_transform:\n")
        f.write(f"  resolution: {params.resolution}\n")
        f.write(f"  grid_origin_world_x: {params.world_x_min}\n")
        f.write(f"  grid_origin_world_y: {params.world_y_min}\n")
        f.write(f"  world_bounds: [{params.world_x_min}, {params.world_x_max}, {params.world_y_min}, {params.world_y_max}]\n")
        f.write("  coordinate_system: bottom_left_origin\n")
        
        # Grid map coordinate system explanation
        f.write("# Grid map coordinate system:\n")
        f.write("# - Origin at bottom-left corner\n")
        f.write("# - X-axis points right\n")
        f.write("# - Y-axis points up\n")
        f.write(f"# - Grid (0,0) = World ({params.world_x_min}, {params.world_y_min})\n")
        f.write(f"# - Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")


def convert_png_to_pgm(png_path: str, output_path: str, params: GridMapParams,
                      world_x_min: float, world_x_max: float,
                      world_y_min: float, world_y_max: float) -> None:
    """Main conversion function with world coordinate bounds."""
    
    # Load PNG image
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG file not found: {png_path}")
    
    # Read as grayscale
    png_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if png_image is None:
        raise ValueError(f"Cannot read PNG image: {png_path}")
    
    print(f"Loaded PNG image: {png_image.shape[1]}x{png_image.shape[0]} pixels")
    
    # Set world coordinate bounds
    params.world_x_min = world_x_min
    params.world_x_max = world_x_max
    params.world_y_min = world_y_min
    params.world_y_max = world_y_max
    
    # Calculate map dimensions from world coordinates
    params.map_width = world_x_max - world_x_min
    params.map_height = world_y_max - world_y_min
    
    print(f"World coordinate bounds: [{world_x_min}, {world_x_max}] x [{world_y_min}, {world_y_max}]")
    print(f"Map size: {params.map_width:.3f}m x {params.map_height:.3f}m")
    
    # Verify image size consistency
    expected_width = int(params.map_width / params.resolution)
    expected_height = int(params.map_height / params.resolution)
    
    print(f"Expected grid size: {expected_width}x{expected_height} pixels")
    print(f"Actual image size: {png_image.shape[1]}x{png_image.shape[0]} pixels")
    
    if png_image.shape[1] != expected_width or png_image.shape[0] != expected_height:
        print("Warning: Image size doesn't exactly match world bounds and resolution!")
        print("This may cause coordinate alignment issues.")
        
        # Option to adjust resolution to match
        actual_res_x = params.map_width / png_image.shape[1]
        actual_res_y = params.map_height / png_image.shape[0]
        print(f"Implied resolution: X={actual_res_x:.6f}, Y={actual_res_y:.6f}")
        
        if abs(actual_res_x - actual_res_y) < 0.001:  # Square pixels
            suggested_resolution = (actual_res_x + actual_res_y) / 2
            print(f"Suggested resolution: {suggested_resolution:.6f}")
    
    # Convert PNG values to grid map values
    grid_map = png_to_grid_values(png_image, params)
    
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
    
    # Print coordinate transformation information for vSLAM system
    print(f"\nCoordinate Transform Information:")
    print(f"  Grid origin world coordinates: ({world_x_min}, {world_y_min})")
    print(f"  Map center: ({(world_x_min + world_x_max)/2.0}, {(world_y_min + world_y_max)/2.0})")
    print(f"  Resolution: {params.resolution} m/pixel")