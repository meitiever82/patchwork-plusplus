#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <patchwork/patchworkpp.h>

// for list folder
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

using namespace open3d;

struct GridMapParams {
  double resolution       = 0.1;    // grid resolution in meters
  double map_width        = 100.0;  // map width in meters
  double map_height       = 100.0;  // map height in meters
  double center_x         = 0.0;    // map center x coordinate
  double center_y         = 0.0;    // map center y coordinate
  uint8_t ground_value    = 0;      // value for ground cells
  uint8_t unknown_value   = 100;    // value for unknown cells
  uint8_t obstacle_value  = 200;    // value for obstacle cells
  double height_threshold = 3.5;    // height threshold for filtering high points - 提高以保留更多墙壁

  // 形态学操作参数 - 专注于地面连续性
  int erosion_size  = 1;  // 腐蚀核大小 - 减小以保留障碍物细节
  int dilation_size = 1;  // 膨胀核大小 - 轻微安全边界
  int opening_size  = 2;  // 开运算核大小 - 只移除极小噪声
  int closing_size  = 4;  // 闭运算核大小 - 增大以填充地面间隙

  // 坐标系对齐参数
  bool use_custom_origin = false;  // 是否使用自定义原点
  double origin_x        = 0.0;    // 3D点云原点的X坐标
  double origin_y        = 0.0;    // 3D点云原点的Y坐标
  double origin_z        = 0.0;    // 3D点云原点的Z坐标（可选，用于验证）
};

void read_bin(const std::string bin_path, Eigen::MatrixXf& cloud) {
  FILE* file = fopen(bin_path.c_str(), "rb");
  if (!file) {
    std::cerr << "error: failed to load " << bin_path << std::endl;
    return;
  }

  std::vector<float> buffer(1000000);
  size_t num_points =
      fread(reinterpret_cast<char*>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
  fclose(file);

  cloud.resize(num_points, 4);
  for (int i = 0; i < num_points; i++) {
    cloud.row(i) << buffer[i * 4], buffer[i * 4 + 1], buffer[i * 4 + 2], buffer[i * 4 + 3];
  }
}

void read_pcd(const std::string pcd_path, Eigen::MatrixXf& cloud) {
  auto pcd = io::CreatePointCloudFromFile(pcd_path);
  if (pcd->IsEmpty()) {
    std::cerr << "error: failed to load " << pcd_path << std::endl;
    return;
  }

  cloud.resize(pcd->points_.size(), 3);
  for (size_t i = 0; i < pcd->points_.size(); i++) {
    cloud.row(i) << pcd->points_[i](0), pcd->points_[i](1), pcd->points_[i](2);
  }
}

void eigen2geo(const Eigen::MatrixX3f& points, std::shared_ptr<geometry::PointCloud> geo) {
  geo->points_.clear();
  geo->points_.reserve(points.rows());
  for (int i = 0; i < points.rows(); i++) {
    geo->points_.push_back(Eigen::Vector3d(points(i, 0), points(i, 1), points(i, 2)));
  }
}

void addNormals(const Eigen::MatrixX3f& normals, std::shared_ptr<geometry::PointCloud> geo) {
  geo->normals_.clear();
  geo->normals_.reserve(normals.rows());
  for (int i = 0; i < normals.rows(); i++) {
    geo->normals_.push_back(Eigen::Vector3d(normals(i, 0), normals(i, 1), normals(i, 2)));
  }
}

// 辅助函数：分析点云并建议原点位置
void analyzePointCloudForOrigin(const Eigen::MatrixXf& cloud) {
  if (cloud.rows() == 0) return;

  std::cout << "\n=== Point Cloud Origin Analysis ===" << std::endl;

  // 计算统计信息
  Eigen::VectorXf x_coords = cloud.col(0);
  Eigen::VectorXf y_coords = cloud.col(1);
  Eigen::VectorXf z_coords = cloud.col(2);

  float x_min = x_coords.minCoeff();
  float x_max = x_coords.maxCoeff();
  float y_min = y_coords.minCoeff();
  float y_max = y_coords.maxCoeff();
  float z_min = z_coords.minCoeff();
  float z_max = z_coords.maxCoeff();

  float x_mean = x_coords.mean();
  float y_mean = y_coords.mean();
  float z_mean = z_coords.mean();

  std::cout << "Point cloud statistics:" << std::endl;
  std::cout << "  X range: [" << x_min << ", " << x_max << "], mean: " << x_mean << std::endl;
  std::cout << "  Y range: [" << y_min << ", " << y_max << "], mean: " << y_mean << std::endl;
  std::cout << "  Z range: [" << z_min << ", " << z_max << "], mean: " << z_mean << std::endl;

  // 分析可能的原点位置
  std::cout << "\nPossible origin scenarios:" << std::endl;

  // 1. 数据围绕原点分布
  if (std::abs(x_mean) < 1.0 && std::abs(y_mean) < 1.0) {
    std::cout << "  ✓ Scenario 1: Data centered around origin (0, 0)" << std::endl;
    std::cout << "    -> Use: origin_x=0 origin_y=0" << std::endl;
  }

  // 2. 传感器在原点，数据在正象限
  if (x_min >= -1.0 && y_min >= -1.0 && x_min < 10.0 && y_min < 10.0) {
    std::cout << "  ✓ Scenario 2: Sensor at origin, data in positive quadrant" << std::endl;
    std::cout << "    -> Use: origin_x=0 origin_y=0" << std::endl;
  }

  // 3. 数据明显偏移
  if (std::abs(x_mean) > 10.0 || std::abs(y_mean) > 10.0) {
    std::cout << "  ✓ Scenario 3: Data is offset from coordinate origin" << std::endl;
    std::cout << "    -> Consider: origin_x=" << x_mean << " origin_y=" << y_mean
              << " (data center)" << std::endl;
    std::cout << "    -> Or use specific reference point if known" << std::endl;
  }

  // 4. 检查是否有接近原点的点
  int near_origin_count = 0;
  for (int i = 0; i < cloud.rows(); ++i) {
    if (std::abs(cloud(i, 0)) < 2.0 && std::abs(cloud(i, 1)) < 2.0) {
      near_origin_count++;
    }
  }

  float near_origin_ratio = static_cast<float>(near_origin_count) / cloud.rows();
  std::cout << "\nPoints near (0,0): " << near_origin_count << " (" << near_origin_ratio * 100
            << "%)" << std::endl;

  if (near_origin_ratio > 0.1) {
    std::cout << "  -> Significant data near origin, likely sensor-centered coordinate system"
              << std::endl;
  } else {
    std::cout << "  -> Little data near origin, coordinate system may be transformed" << std::endl;
  }

  std::cout << "\n💡 Recommended usage:" << std::endl;
  std::cout << "  Auto-centered: ./demo file.pcd output 0.2" << std::endl;
  std::cout << "  Origin-aligned: ./demo file.pcd output 0.2 2 3 <origin_x> <origin_y>"
            << std::endl;
  std::cout << "=======================================\n" << std::endl;
}

GridMapParams calculateOptimalParams(const Eigen::MatrixX3f& ground_points,
                                     const Eigen::MatrixX3f& nonground_points,
                                     double resolution      = 0.2,
                                     bool use_custom_origin = false,
                                     double custom_origin_x = 0.0,
                                     double custom_origin_y = 0.0) {
  GridMapParams params;
  params.resolution        = resolution;
  params.use_custom_origin = use_custom_origin;
  params.origin_x          = custom_origin_x;
  params.origin_y          = custom_origin_y;

  if (use_custom_origin) {
    std::cout << "Using custom origin alignment:" << std::endl;
    std::cout << "  3D origin: (" << custom_origin_x << ", " << custom_origin_y << ", 0)"
              << std::endl;
    std::cout << "  Grid map (0,0) will align to this 3D point" << std::endl;

    // 使用自定义原点时，需要根据数据分布确定地图大小
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    // 计算所有点的边界框
    for (int i = 0; i < ground_points.rows(); ++i) {
      min_x = std::min(min_x, ground_points(i, 0));
      max_x = std::max(max_x, ground_points(i, 0));
      min_y = std::min(min_y, ground_points(i, 1));
      max_y = std::max(max_y, ground_points(i, 1));
    }

    for (int i = 0; i < nonground_points.rows(); ++i) {
      if (nonground_points(i, 2) <= params.height_threshold) {
        min_x = std::min(min_x, nonground_points(i, 0));
        max_x = std::max(max_x, nonground_points(i, 0));
        min_y = std::min(min_y, nonground_points(i, 1));
        max_y = std::max(max_y, nonground_points(i, 1));
      }
    }

    // 计算相对于自定义原点的范围
    float range_x_min = min_x - custom_origin_x;
    float range_x_max = max_x - custom_origin_x;
    float range_y_min = min_y - custom_origin_y;
    float range_y_max = max_y - custom_origin_y;

    // 确保地图能包含所有数据，并留一些边界
    float margin    = 10.0;  // 10米边界
    float map_x_min = std::min(range_x_min - margin, -margin);
    float map_x_max = std::max(range_x_max + margin, margin);
    float map_y_min = std::min(range_y_min - margin, -margin);
    float map_y_max = std::max(range_y_max + margin, margin);

    params.map_width  = map_x_max - map_x_min;
    params.map_height = map_y_max - map_y_min;

    // 计算地图中心（世界坐标）
    params.center_x = custom_origin_x + (map_x_min + map_x_max) / 2.0;
    params.center_y = custom_origin_y + (map_y_min + map_y_max) / 2.0;

    std::cout << "Custom origin grid map parameters:" << std::endl;
    std::cout << "  3D data range: [" << min_x << ", " << max_x << "] x [" << min_y << ", " << max_y
              << "]" << std::endl;
    std::cout << "  Grid map range relative to origin: [" << map_x_min << ", " << map_x_max
              << "] x [" << map_y_min << ", " << map_y_max << "]" << std::endl;

  } else {
    // 原有的自动计算逻辑
    std::cout << "Using automatic parameter calculation (data-centered):" << std::endl;

    // Calculate bounding box of all points
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    // Check ground points
    for (int i = 0; i < ground_points.rows(); ++i) {
      min_x = std::min(min_x, ground_points(i, 0));
      max_x = std::max(max_x, ground_points(i, 0));
      min_y = std::min(min_y, ground_points(i, 1));
      max_y = std::max(max_y, ground_points(i, 1));
    }

    // Check nonground points (only those <= height_threshold for bounding box calculation)
    for (int i = 0; i < nonground_points.rows(); ++i) {
      if (nonground_points(i, 2) <= params.height_threshold) {
        min_x = std::min(min_x, nonground_points(i, 0));
        max_x = std::max(max_x, nonground_points(i, 0));
        min_y = std::min(min_y, nonground_points(i, 1));
        max_y = std::max(max_y, nonground_points(i, 1));
      }
    }

    // Add margin (20% on each side)
    float margin_x = (max_x - min_x) * 0.2;
    float margin_y = (max_y - min_y) * 0.2;
    min_x -= margin_x;
    max_x += margin_x;
    min_y -= margin_y;
    max_y += margin_y;

    // Calculate map parameters
    params.map_width  = max_x - min_x;
    params.map_height = max_y - min_y;
    params.center_x   = (min_x + max_x) / 2.0;
    params.center_y   = (min_y + max_y) / 2.0;

    std::cout << "  Bounding box: [" << min_x << ", " << max_x << "] x [" << min_y << ", " << max_y
              << "]" << std::endl;
  }

  std::cout << "  Map size: " << params.map_width << "m x " << params.map_height << "m"
            << std::endl;
  std::cout << "  Map center: (" << params.center_x << ", " << params.center_y << ")" << std::endl;

  if (use_custom_origin) {
    // 计算grid map原点在世界坐标系中的位置
    double grid_origin_world_x = params.center_x - params.map_width / 2.0;
    double grid_origin_world_y = params.center_y - params.map_height / 2.0;
    std::cout << "  Grid map origin in world coords: (" << grid_origin_world_x << ", "
              << grid_origin_world_y << ")" << std::endl;

    // 验证3D原点是否在grid map范围内
    int origin_grid_x    = static_cast<int>((custom_origin_x - grid_origin_world_x) / resolution);
    int origin_grid_y    = static_cast<int>((custom_origin_y - grid_origin_world_y) / resolution);
    int map_width_cells  = static_cast<int>(params.map_width / params.resolution);
    int map_height_cells = static_cast<int>(params.map_height / params.resolution);

    if (origin_grid_x >= 0 && origin_grid_x < map_width_cells && origin_grid_y >= 0 &&
        origin_grid_y < map_height_cells) {
      std::cout << "  ✅ 3D origin maps to grid cell (" << origin_grid_x << ", " << origin_grid_y
                << ")" << std::endl;
    } else {
      std::cout << "  ⚠️  3D origin is outside grid map bounds!" << std::endl;
    }
  } else {
    std::cout << "  Grid map origin: (" << (params.center_x - params.map_width / 2.0) << ", "
              << (params.center_y - params.map_height / 2.0) << ")" << std::endl;
  }

  std::cout << "  Resolution: " << params.resolution << "m/cell" << std::endl;
  std::cout << "  Grid size: " << static_cast<int>(params.map_width / params.resolution) << " x "
            << static_cast<int>(params.map_height / params.resolution) << " cells" << std::endl;
  std::cout << "  Morphology: opening=" << params.opening_size
            << ", closing=" << params.closing_size << ", dilation=" << params.dilation_size
            << std::endl;

  return params;
}

// 优化外边界轮廓
cv::Mat smoothOuterBoundaries(const cv::Mat& obstacles_mask) {
  cv::Mat smoothed_obstacles = obstacles_mask.clone();
  
  std::cout << "  Smoothing outer boundaries..." << std::endl;
  
  // 1. 寻找所有连通域
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(obstacles_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  
  // 2. 对每个连通域进行轮廓优化
  cv::Mat result = cv::Mat::zeros(obstacles_mask.size(), CV_8UC1);
  
  for (size_t i = 0; i < contours.size(); i++) {
    if (contours[i].size() < 5) continue; // 跳过太小的轮廓
    
    // 2.1 多边形逼近平滑轮廓
    std::vector<cv::Point> approx_contour;
    double epsilon = 0.02 * cv::arcLength(contours[i], true); // 自适应精度
    cv::approxPolyDP(contours[i], approx_contour, epsilon, true);
    
    // 2.2 对于大型建筑物，使用更精细的平滑
    if (cv::contourArea(contours[i]) > 500) { // 50平方米以上的大建筑
      // 使用更小的epsilon获得更精细的轮廓
      epsilon = 0.01 * cv::arcLength(contours[i], true);
      cv::approxPolyDP(contours[i], approx_contour, epsilon, true);
    }
    
    // 2.3 填充优化后的轮廓
    std::vector<std::vector<cv::Point>> smooth_contours;
    smooth_contours.push_back(approx_contour);
    cv::fillPoly(result, smooth_contours, cv::Scalar(255));
  }
  
  // 3. 轻微的形态学平滑去除小锯齿
  cv::Mat smooth_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(result, result, cv::MORPH_CLOSE, smooth_kernel);
  cv::morphologyEx(result, result, cv::MORPH_OPEN, smooth_kernel);
  
  int before_pixels = cv::countNonZero(obstacles_mask);
  int after_pixels = cv::countNonZero(result);
  std::cout << "  Boundary smoothing: " << before_pixels << " -> " << after_pixels 
            << " pixels (" << ((after_pixels - before_pixels) * 100.0 / before_pixels) 
            << "% change)" << std::endl;
  
  return result;
}

// 连接断开的边界线
cv::Mat connectBoundaries(const cv::Mat& obstacles_mask) {
  cv::Mat connected_obstacles = obstacles_mask.clone();
  
  std::cout << "  Connecting broken boundaries..." << std::endl;
  
  // 1. 检测边缘
  cv::Mat edges;
  cv::Canny(obstacles_mask * 255, edges, 30, 90);
  
  // 2. 使用形态学闭运算连接近邻边界
  cv::Mat line_kernel_h = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1)); // 水平线核
  cv::Mat line_kernel_v = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5)); // 垂直线核
  
  cv::Mat connected_h, connected_v;
  cv::morphologyEx(connected_obstacles, connected_h, cv::MORPH_CLOSE, line_kernel_h);
  cv::morphologyEx(connected_obstacles, connected_v, cv::MORPH_CLOSE, line_kernel_v);
  
  // 3. 合并水平和垂直连接结果
  cv::bitwise_or(connected_h, connected_v, connected_obstacles);
  
  // 4. 使用对角线核连接对角边界
  cv::Mat diag_kernel = (cv::Mat_<uint8_t>(3, 3) << 
                         1, 0, 0,
                         0, 1, 0, 
                         0, 0, 1);
  cv::Mat diag_kernel2 = (cv::Mat_<uint8_t>(3, 3) << 
                          0, 0, 1,
                          0, 1, 0,
                          1, 0, 0);
  
  cv::Mat connected_d1, connected_d2;
  cv::morphologyEx(connected_obstacles, connected_d1, cv::MORPH_CLOSE, diag_kernel);
  cv::morphologyEx(connected_obstacles, connected_d2, cv::MORPH_CLOSE, diag_kernel2);
  
  // 5. 最终合并
  cv::bitwise_or(connected_obstacles, connected_d1, connected_obstacles);
  cv::bitwise_or(connected_obstacles, connected_d2, connected_obstacles);
  
  // 6. 轻微平滑以去除连接产生的锯齿
  cv::Mat smooth_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
  cv::morphologyEx(connected_obstacles, connected_obstacles, cv::MORPH_CLOSE, smooth_kernel);
  
  return connected_obstacles;
}

// 移除小连通域
cv::Mat removeSmallRegions(const cv::Mat& binary_mask, int min_area_threshold) {
  cv::Mat labels, stats, centroids;
  cv::Mat cleaned_mask = binary_mask.clone();
  
  int num_components = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids);
  
  int removed_count = 0;
  for (int i = 1; i < num_components; i++) { // 跳过背景(0)
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < min_area_threshold) {
      cleaned_mask.setTo(0, labels == i);
      removed_count++;
    }
  }
  
  std::cout << "  Removed " << removed_count << " small regions (< " 
            << min_area_threshold << " pixels)" << std::endl;
  
  return cleaned_mask;
}

// 使用形态学操作后处理网格地图
cv::Mat postProcessGridMap(const cv::Mat& grid_map, const GridMapParams& params) {
  cv::Mat processed_map = grid_map.clone();

  std::cout << "Post-processing with morphological operations..." << std::endl;

  // 分别提取障碍物和地面掩码
  cv::Mat obstacle_mask = (processed_map == params.obstacle_value);
  cv::Mat ground_mask   = (processed_map == params.ground_value);

  // 创建形态学操作核
  cv::Mat opening_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(params.opening_size, params.opening_size));
  cv::Mat closing_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(params.closing_size, params.closing_size));
  cv::Mat dilation_kernel = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(params.dilation_size, params.dilation_size));

  // 1. 对障碍物进行开运算：移除小的孤立障碍物
  cv::Mat cleaned_obstacles;
  cv::morphologyEx(obstacle_mask, cleaned_obstacles, cv::MORPH_OPEN, opening_kernel);

  // 2. 对地面进行开运算：移除小的孤立地面区域
  cv::Mat cleaned_ground;
  cv::morphologyEx(ground_mask, cleaned_ground, cv::MORPH_OPEN, opening_kernel);

  // 3. 对障碍物进行闭运算：填充障碍物内部的小洞
  cv::morphologyEx(cleaned_obstacles, cleaned_obstacles, cv::MORPH_CLOSE, closing_kernel);

  // 4. 对地面进行多次闭运算：填充地面内部的小洞，提高连续性
  cv::morphologyEx(cleaned_ground, cleaned_ground, cv::MORPH_CLOSE, closing_kernel);
  
  // 4.1. 额外的地面连续性处理 - 再次闭运算以连接近邻地面区域
  cv::Mat larger_closing_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
  cv::morphologyEx(cleaned_ground, cleaned_ground, cv::MORPH_CLOSE, larger_closing_kernel);

  // 4.5. 不移除障碍物区域，保留所有检测到的障碍物
  // int min_obstacle_area = static_cast<int>(4 / (params.resolution * params.resolution)); 
  // cleaned_obstacles = removeSmallRegions(cleaned_obstacles, min_obstacle_area);

  // 4.6. 边界连续性优化 - 连接断开的建筑物边界
  cleaned_obstacles = connectBoundaries(cleaned_obstacles);

  // 4.7. 外边界平滑优化 - 优化建筑物外轮廓
  cleaned_obstacles = smoothOuterBoundaries(cleaned_obstacles);

  // 5. 可选：对障碍物进行轻微膨胀以提供安全边界
  cv::Mat safety_obstacles;
  cv::dilate(cleaned_obstacles, safety_obstacles, dilation_kernel);

  // 6. 重建网格地图
  processed_map.setTo(params.unknown_value);  // 先设为未知

  // 优先级：障碍物 > 地面 > 未知
  // 先设置地面，再设置障碍物（障碍物具有更高优先级）
  processed_map.setTo(params.ground_value, cleaned_ground);
  processed_map.setTo(params.obstacle_value, safety_obstacles);

  // 统计处理效果
  int original_obstacles = cv::countNonZero(obstacle_mask);
  int final_obstacles    = cv::countNonZero(safety_obstacles);
  int original_ground    = cv::countNonZero(ground_mask);
  int final_ground       = cv::countNonZero(cleaned_ground);

  std::cout << "Morphological processing results:" << std::endl;
  std::cout << "  Obstacles: " << original_obstacles << " -> " << final_obstacles << " (removed "
            << original_obstacles - final_obstacles << " pixels)" << std::endl;
  std::cout << "  Ground: " << original_ground << " -> " << final_ground << " (removed "
            << original_ground - final_ground << " pixels)" << std::endl;

  return processed_map;
}

cv::Mat generateGridMap(const Eigen::MatrixX3f& ground_points,
                        const Eigen::MatrixX3f& nonground_points,
                        const GridMapParams& params) {
  int map_width_cells  = static_cast<int>(params.map_width / params.resolution);
  int map_height_cells = static_cast<int>(params.map_height / params.resolution);

  // Initialize grid map with unknown values (100)
  cv::Mat grid_map =
      cv::Mat::ones(map_height_cells, map_width_cells, CV_8UC1) * params.unknown_value;

  // Lambda function to convert world coordinates to grid coordinates
  // 坐标关系说明：
  // - 世界坐标系：3D点云的坐标系
  // - 网格坐标系：左下角(0,0)，右上角(map_width_cells-1, map_height_cells-1)
  // - X轴：世界坐标向右为正，对应网格坐标列数增加
  // - Y轴：世界坐标向上为正，对应网格坐标行数增加（数学坐标系）
  auto worldToGrid = [&](double world_x, double world_y, int& grid_x, int& grid_y) -> bool {
    if (params.use_custom_origin) {
      // 使用自定义原点：grid map (0,0) 对应 世界坐标 (origin_x, origin_y)
      grid_x = static_cast<int>((world_x - params.origin_x) / params.resolution);
      grid_y = static_cast<int>((world_y - params.origin_y) / params.resolution);
    } else {
      // 原有逻辑：以地图中心为基准
      grid_x = static_cast<int>((world_x - params.center_x + params.map_width / 2.0) /
                                params.resolution);
      grid_y = static_cast<int>((world_y - params.center_y + params.map_height / 2.0) /
                                params.resolution);
    }

    // 检查是否在有效范围内
    return grid_x >= 0 && grid_x < map_width_cells && grid_y >= 0 && grid_y < map_height_cells;
  };

  // 处理地面点：值为0
  std::cout << "Processing ground points..." << std::endl;
  int ground_cells_count = 0;
  for (int i = 0; i < ground_points.rows(); ++i) {
    int grid_x, grid_y;
    if (worldToGrid(ground_points(i, 0), ground_points(i, 1), grid_x, grid_y)) {
      // 直接存储，不在这里翻转
      grid_map.at<uint8_t>(grid_y, grid_x) = params.ground_value;  // 0
      ground_cells_count++;
    }
  }

  // 处理非地面点：忽略高度>2米的点，其余标记为障碍物（值为200）
  std::cout << "Processing nonground points (filtering height > " << params.height_threshold
            << "m)..." << std::endl;
  int obstacle_cells_count = 0;
  int filtered_high_points = 0;

  for (int i = 0; i < nonground_points.rows(); ++i) {
    // 过滤掉高度大于2米的点
    if (nonground_points(i, 2) > params.height_threshold) {
      filtered_high_points++;
      continue;  // 忽略这些点
    }

    int grid_x, grid_y;
    if (worldToGrid(nonground_points(i, 0), nonground_points(i, 1), grid_x, grid_y)) {
      // 直接存储，不在这里翻转
      // 只有当该格子还是未知状态时才标记为障碍物
      // 这样可以避免覆盖已经标记的地面点
      if (grid_map.at<uint8_t>(grid_y, grid_x) == params.unknown_value) {
        grid_map.at<uint8_t>(grid_y, grid_x) = params.obstacle_value;  // 200
        obstacle_cells_count++;
      }
    }
  }

  std::cout << "Filtered " << filtered_high_points << " high points (z > "
            << params.height_threshold << "m)" << std::endl;
  std::cout << "Marked " << ground_cells_count
            << " ground cells (value: " << static_cast<int>(params.ground_value) << ")"
            << std::endl;
  std::cout << "Marked " << obstacle_cells_count
            << " obstacle cells (value: " << static_cast<int>(params.obstacle_value) << ")"
            << std::endl;

  // 后处理：移除孤立的小区域
  cv::Mat processed_grid_map = postProcessGridMap(grid_map, params);

  return processed_grid_map;
}

void saveGridMap(const cv::Mat& grid_map,
                 const std::string& output_path,
                 const GridMapParams& params) {
  // Save as PGM format (Portable Graymap)
  std::string pgm_path = output_path + "_gridmap.pgm";

  std::ofstream pgm_file(pgm_path);
  if (!pgm_file.is_open()) {
    std::cerr << "Error: Cannot create PGM file: " << pgm_path << std::endl;
    return;
  }

  // PGM header (ASCII format P2)
  pgm_file << "P2" << std::endl;
  pgm_file << "# Grid map generated from point cloud" << std::endl;
  pgm_file << "# Resolution: " << params.resolution << " m/pixel" << std::endl;

  if (params.use_custom_origin) {
    pgm_file << "# Custom origin alignment: grid (0,0) = world (" << params.origin_x << ", "
             << params.origin_y << ")" << std::endl;
    pgm_file << "# Map center: (" << params.center_x << ", " << params.center_y << ")" << std::endl;
    // 显示网格地图覆盖的实际世界坐标范围
    double world_x_min = params.origin_x;
    double world_y_min = params.origin_y;
    double world_x_max =
        params.origin_x +
        (static_cast<int>(params.map_width / params.resolution) - 1) * params.resolution;
    double world_y_max =
        params.origin_y +
        (static_cast<int>(params.map_height / params.resolution) - 1) * params.resolution;
    pgm_file << "# World coverage: [" << world_x_min << ", " << world_x_max << "] x ["
             << world_y_min << ", " << world_y_max << "]" << std::endl;
  } else {
    pgm_file << "# Auto-calculated parameters (data-centered)" << std::endl;
    pgm_file << "# Map center: (" << params.center_x << ", " << params.center_y << ")" << std::endl;
    pgm_file << "# Grid origin: (" << (params.center_x - params.map_width / 2.0) << ", "
             << (params.center_y - params.map_height / 2.0) << ")" << std::endl;
  }

  pgm_file << "# Map size: " << params.map_width << "m x " << params.map_height << "m" << std::endl;
  pgm_file << "# Coordinate system: origin at bottom-left, x->right, y->up" << std::endl;
  pgm_file << "# Values: 0=ground, 100=unknown, 200=obstacle" << std::endl;
  pgm_file << grid_map.cols << " " << grid_map.rows << std::endl;
  pgm_file << "255" << std::endl;  // max value

  // Write data with coordinate system conversion
  // PGM格式从左上角开始，但我们的坐标系是左下角原点
  // 所以需要翻转y轴输出
  for (int row = grid_map.rows - 1; row >= 0; --row) {  // 从上到下遍历Mat，对应从下到上的网格坐标
    for (int col = 0; col < grid_map.cols; ++col) {
      uint8_t value = grid_map.at<uint8_t>(row, col);
      pgm_file << static_cast<int>(value);
      if (col < grid_map.cols - 1) pgm_file << " ";
    }
    pgm_file << std::endl;
  }

  pgm_file.close();
  std::cout << "Grid map saved as PGM: " << pgm_path << std::endl;

  // 同时保存元数据文件
  std::string yaml_path = output_path + "_gridmap.yaml";
  std::ofstream yaml_file(yaml_path);
  if (yaml_file.is_open()) {
    yaml_file << "image: " << output_path + "_gridmap.pgm" << std::endl;
    yaml_file << "resolution: " << params.resolution << std::endl;

    // 根据是否使用自定义原点来设置origin
    if (params.use_custom_origin) {
      yaml_file << "origin: [" << params.origin_x << ", " << params.origin_y << ", 0.0]"
                << std::endl;
      yaml_file << "# Custom origin alignment: grid map (0,0) = world (" << params.origin_x << ", "
                << params.origin_y << ")" << std::endl;
    } else {
      yaml_file << "origin: [" << (params.center_x - params.map_width / 2.0) << ", "
                << (params.center_y - params.map_height / 2.0) << ", 0.0]" << std::endl;
      yaml_file << "# Auto-calculated origin: grid map (0,0) = world ("
                << (params.center_x - params.map_width / 2.0) << ", "
                << (params.center_y - params.map_height / 2.0) << ")" << std::endl;
    }

    yaml_file << "negate: 0" << std::endl;
    yaml_file << "occupied_thresh: 0.65" << std::endl;
    yaml_file << "free_thresh: 0.196" << std::endl;
    yaml_file << "# Grid map coordinate system:" << std::endl;
    yaml_file << "# - Origin at bottom-left corner" << std::endl;
    yaml_file << "# - X-axis points right" << std::endl;
    yaml_file << "# - Y-axis points up" << std::endl;
    yaml_file << "# - Values: 0=ground, 100=unknown, 200=obstacle" << std::endl;
    yaml_file.close();
    std::cout << "Map metadata saved as YAML: " << yaml_path << std::endl;
  }

  // 可选：也保存PNG格式用于可视化
  std::string png_path = output_path + "_gridmap_vis.png";
  cv::Mat vis_map;

  // 为可视化创建映射：0->0(黑), 100->127(灰), 200->255(白)
  grid_map.copyTo(vis_map);
  for (int i = 0; i < vis_map.rows; ++i) {
    for (int j = 0; j < vis_map.cols; ++j) {
      uint8_t value = vis_map.at<uint8_t>(i, j);
      if (value == params.ground_value) {  // 0 -> 0 (black)
        vis_map.at<uint8_t>(i, j) = 0;
      } else if (value == params.unknown_value) {  // 100 -> 127 (gray)
        vis_map.at<uint8_t>(i, j) = 127;
      } else if (value == params.obstacle_value) {  // 200 -> 255 (white)
        vis_map.at<uint8_t>(i, j) = 255;
      }
    }
  }

  // 翻转y轴用于保存（让保存的图像也是左下角原点）
  cv::Mat display_png;
  cv::flip(vis_map, display_png, 0);  // 垂直翻转
  cv::imwrite(png_path, display_png);
  std::cout << "Visualization PNG saved (bottom-left origin): " << png_path << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "Execute " << __FILE__ << std::endl;

  // Get the dataset
  std::string input_cloud_filepath;
  std::string output_path = "output";
  double resolution       = 0.1;  // default 10cm resolution
  int opening_size        = 2;    // default opening kernel size
  int closing_size        = 3;    // default closing kernel size
  bool use_custom_origin  = false;
  double custom_origin_x  = 0.0;
  double custom_origin_y  = 0.0;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0]
              << " <point_cloud_file> [output_path] [resolution] [opening_size] [closing_size] "
                 "[origin_x] [origin_y]"
              << std::endl;
    std::cout << "Supported formats: .bin, .pcd" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  resolution: grid resolution in meters (default: 0.2m)" << std::endl;
    std::cout << "  opening_size: kernel size for removing small regions (default: 2)" << std::endl;
    std::cout << "  closing_size: kernel size for filling holes (default: 3)" << std::endl;
    std::cout << "  origin_x, origin_y: 3D point cloud origin coordinates (optional)" << std::endl;
    std::cout << "\nCoordinate alignment:" << std::endl;
    std::cout << "  Without origin_x,y: grid map auto-centered on data" << std::endl;
    std::cout << "  With origin_x,y: grid map (0,0) aligned to 3D origin (origin_x, origin_y)"
              << std::endl;
    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  <o>_gridmap.pgm: main grid map in PGM format" << std::endl;
    std::cout << "  <o>_gridmap.yaml: metadata file for ROS compatibility" << std::endl;
    std::cout << "  <o>_gridmap_vis.png: visualization image" << std::endl;
    std::cout << "\nCoordinate system: origin at bottom-left, X->right, Y->up" << std::endl;
    std::cout << "Grid values: 0=ground, 100=unknown, 200=obstacle" << std::endl;
    return -1;
  } else {
    input_cloud_filepath = argv[1];
    if (argc > 2) {
      output_path = argv[2];
    }
    if (argc > 3) {
      resolution = std::stod(argv[3]);
      std::cout << "Using custom resolution: " << resolution << "m" << std::endl;
    }
    if (argc > 4) {
      opening_size = std::stoi(argv[4]);
      std::cout << "Using custom opening_size: " << opening_size << std::endl;
    }
    if (argc > 5) {
      closing_size = std::stoi(argv[5]);
      std::cout << "Using custom closing_size: " << closing_size << std::endl;
    }
    if (argc > 6 && argc > 7) {  // 需要同时提供x和y
      custom_origin_x   = std::stod(argv[6]);
      custom_origin_y   = std::stod(argv[7]);
      use_custom_origin = true;
      std::cout << "\033[1;33mUsing custom origin alignment: (" << custom_origin_x << ", "
                << custom_origin_y << ")\033[0m" << std::endl;
      std::cout << "Grid map (0,0) will align to 3D world (" << custom_origin_x << ", "
                << custom_origin_y << ")" << std::endl;
    } else if (argc > 6) {
      std::cout << "\033[1;31mError: Both origin_x and origin_y must be provided\033[0m"
                << std::endl;
      return -1;
    }

    std::cout << "\033[1;32mLoading point cloud from " << input_cloud_filepath << "\033[0m"
              << std::endl;
  }

  if (!fs::exists(input_cloud_filepath)) {
    std::cout << "\033[1;31mERROR: File does not exist: " << input_cloud_filepath << "\033[0m"
              << std::endl;
    return -1;
  }

  // Patchwork++ initialization - Optimized for casbot data
  patchwork::Params patchwork_parameters;
  
  // Sensor parameters - adjust based on your robot
  patchwork_parameters.sensor_height = 1.2;  // Typical mobile robot sensor height
  patchwork_parameters.max_range = 80.0;     // Increased for outdoor scenarios  
  patchwork_parameters.min_range = 0.3;      // Avoid too close points
  
  // Core ground segmentation parameters - more conservative
  patchwork_parameters.num_iter = 3;         // Reduced iterations for speed
  patchwork_parameters.num_lpr = 20;         // Lower point requirement per ring
  patchwork_parameters.num_min_pts = 10;     // Minimum points for ground estimation
  
  // Ground detection thresholds - stricter for better precision
  patchwork_parameters.th_seeds = 0.5;       // Seed threshold for ground initialization
  patchwork_parameters.th_dist = 0.3;        // Distance threshold to ground plane
  patchwork_parameters.th_seeds_v = 0.4;     // Vertical seed threshold
  patchwork_parameters.th_dist_v = 0.3;      // Vertical distance threshold
  
  // Normal vector thresholds
  patchwork_parameters.uprightness_thr = 0.707;  // cos(45°) for stricter ground normal
  patchwork_parameters.adaptive_seed_selection_margin = -1.2;  // Adaptive margin

  // Ring and zone configuration - default proven settings
  patchwork_parameters.num_zones = 4;
  patchwork_parameters.num_sectors_each_zone = {16, 32, 54, 32};
  patchwork_parameters.num_rings_each_zone = {2, 4, 4, 4};
  patchwork_parameters.elevation_thr = {0.523, 0.746, 0.879, 1.125};
  patchwork_parameters.flatness_thr = {0.0005, 0.000725, 0.001, 0.001};

  // Advanced features
  patchwork_parameters.enable_RNR = true;    // Enable Reflected Noise Removal
  patchwork_parameters.enable_TGR = true;    // Enable Temporal Ground Revert
  patchwork_parameters.enable_RVPF = true;   // Enable Regional Vertical Plane Fitting
  patchwork_parameters.RNR_ver_angle_thr = -15.0;
  patchwork_parameters.RNR_intensity_thr = 0.2;
  
  patchwork_parameters.num_rings_of_interest = 4;
  patchwork_parameters.max_flatness_storage = 1000;
  patchwork_parameters.max_elevation_storage = 1000;
  patchwork_parameters.verbose = true;  
  patchwork::PatchWorkpp Patchworkpp(patchwork_parameters);

  // Load point cloud
  Eigen::MatrixXf cloud;
  std::string file_extension = input_cloud_filepath.substr(input_cloud_filepath.find_last_of("."));

  if (file_extension == ".bin") {
    read_bin(input_cloud_filepath, cloud);
  } else if (file_extension == ".pcd") {
    read_pcd(input_cloud_filepath, cloud);
  } else {
    std::cout << "\033[1;31mERROR: Unsupported file format. Use .bin or .pcd\033[0m" << std::endl;
    return -1;
  }

  if (cloud.rows() == 0) {
    std::cout << "\033[1;31mERROR: Failed to load point cloud\033[0m" << std::endl;
    return -1;
  }

  // Estimate Ground
  Patchworkpp.estimateGround(cloud);

  // Get Ground and Nonground
  Eigen::MatrixX3f ground    = Patchworkpp.getGround();
  Eigen::MatrixX3f nonground = Patchworkpp.getNonground();
  double time_taken          = Patchworkpp.getTimeTaken();

  // Get centers and normals for patches
  Eigen::MatrixX3f centers = Patchworkpp.getCenters();
  Eigen::MatrixX3f normals = Patchworkpp.getNormals();

  std::cout << "Original Points  #: " << cloud.rows() << std::endl;
  std::cout << "Ground Points    #: " << ground.rows() << std::endl;
  std::cout << "Nonground Points #: " << nonground.rows() << std::endl;
  std::cout << "Time Taken : " << time_taken / 1000000 << " (sec)" << std::endl;

  // 分析点云，帮助用户理解坐标系
  if (!use_custom_origin) {
    analyzePointCloudForOrigin(cloud);
  }

  // Generate grid map with optimal parameters
  GridMapParams grid_params = calculateOptimalParams(
      ground, nonground, resolution, use_custom_origin, custom_origin_x, custom_origin_y);

  // 应用命令行参数
  grid_params.opening_size = opening_size;
  grid_params.closing_size = closing_size;

  std::cout << "\nGrid map generation with morphological processing:" << std::endl;
  std::cout << "  Height filter: ignore points above " << grid_params.height_threshold << "m"
            << std::endl;
  std::cout << "  Opening kernel: " << grid_params.opening_size << "x" << grid_params.opening_size
            << " (removes isolated regions ~" << grid_params.opening_size * resolution << "m)"
            << std::endl;
  std::cout << "  Closing kernel: " << grid_params.closing_size << "x" << grid_params.closing_size
            << " (fills holes ~" << grid_params.closing_size * resolution << "m)" << std::endl;

  // 打印坐标系信息
  if (use_custom_origin) {
    std::cout << "\n🎯 Using custom origin alignment mode" << std::endl;
  } else {
    std::cout << "\n📍 Using auto-centered mode" << std::endl;
  }
  // printCoordinateSystemInfo(grid_params);

  cv::Mat grid_map = generateGridMap(ground, nonground, grid_params);

  // Calculate grid map statistics
  int ground_cells = 0, obstacle_cells = 0, unknown_cells = 0;
  for (int i = 0; i < grid_map.rows; ++i) {
    for (int j = 0; j < grid_map.cols; ++j) {
      uint8_t value = grid_map.at<uint8_t>(i, j);
      if (value == grid_params.ground_value)  // 100
        ground_cells++;
      else if (value == grid_params.obstacle_value)  // 0
        obstacle_cells++;
      else if (value == grid_params.unknown_value)  // 200
        unknown_cells++;
    }
  }

  std::cout << "\nGrid map statistics:" << std::endl;
  std::cout << "  Total cells: " << grid_map.rows * grid_map.cols << std::endl;
  std::cout << "  Ground cells (value=0): " << ground_cells << " ("
            << 100.0 * ground_cells / (grid_map.rows * grid_map.cols) << "%)" << std::endl;
  std::cout << "  Obstacle cells (value=200): " << obstacle_cells << " ("
            << 100.0 * obstacle_cells / (grid_map.rows * grid_map.cols) << "%)" << std::endl;
  std::cout << "  Unknown cells (value=100): " << unknown_cells << " ("
            << 100.0 * unknown_cells / (grid_map.rows * grid_map.cols) << "%)" << std::endl;

  // Save grid map with updated function signature
  saveGridMap(grid_map, output_path, grid_params);

  // Display grid map with coordinate system correction
  cv::Mat display_map;
  cv::resize(grid_map, display_map, cv::Size(800, 800));  // Resize for better display

  // 翻转y轴用于显示（我们的坐标系是左下角原点，但OpenCV显示是左上角原点）
  cv::Mat flipped_display;
  cv::flip(display_map, flipped_display, 0);  // 垂直翻转

  // Create a colored version for better visualization
  cv::Mat colored_map;
  cv::applyColorMap(flipped_display, colored_map, cv::COLORMAP_JET);

  // Show both grayscale and colored versions
  cv::imshow("Grid Map (Origin: bottom-left, Value: 0=Obstacle, 100=Ground, 200=Unknown)",
             flipped_display);
  cv::imshow("Grid Map (Colored)", colored_map);

  std::cout << "\nGrid map generated successfully!" << std::endl;
  std::cout << "Map size: " << grid_map.cols << " x " << grid_map.rows << " pixels" << std::endl;
  if (grid_params.map_width > 200.0 || grid_params.map_height > 200.0) {
    std::cout
        << "⚠️  Large map detected! Consider using lower resolution if visualization is slow."
        << std::endl;
  }
  if (ground_cells < 100) {
    std::cout
        << "⚠️  Few ground cells detected! Consider using higher resolution or check input "
           "data."
        << std::endl;
  }

  std::cout << "Grid map generated successfully!" << std::endl;
  std::cout << "Press keys:" << std::endl;
  std::cout << "\t V  : open 3D visualization" << std::endl;
  std::cout << "\t S  : save and exit" << std::endl;
  std::cout << "\tESC : exit without 3D visualization" << std::endl;

  // Handle key press
  int key = cv::waitKey(0);
  cv::destroyWindow("Grid Map (Origin: bottom-left, Value: 0=Ground, 100=Unknown, 200=Obstacle)");
  cv::destroyWindow("Grid Map (Colored)");

  if (key == 'v' || key == 'V') {
    std::cout << "Opening 3D visualization..." << std::endl;
    std::cout << "Mouse controls:" << std::endl;
    std::cout << "\t Left click + drag  : rotate view" << std::endl;
    std::cout << "\t Right click + drag : translate view" << std::endl;
    std::cout << "\t Scroll wheel       : zoom in/out" << std::endl;
    std::cout << "\t H                  : help" << std::endl;
    std::cout << "\t ESC                : close window" << std::endl;

    // 3D Visualization with enhanced interactivity
    std::shared_ptr<geometry::PointCloud> geo_ground(new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> geo_nonground(new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> geo_centers(new geometry::PointCloud);

    eigen2geo(ground, geo_ground);
    eigen2geo(nonground, geo_nonground);
    eigen2geo(centers, geo_centers);
    addNormals(normals, geo_centers);

    geo_ground->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));     // Green
    geo_nonground->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));  // Red
    geo_centers->PaintUniformColor(Eigen::Vector3d(1.0, 1.0, 0.0));    // Yellow

    // Create visualizer with interactive controls
    visualization::VisualizerWithKeyCallback visualizer;
    visualizer.CreateVisualizerWindow("Ground Segmentation Results - Interactive View", 1600, 900);

    // Add geometries
    visualizer.AddGeometry(geo_ground);
    visualizer.AddGeometry(geo_nonground);
    visualizer.AddGeometry(geo_centers);

    // Set up camera view for better initial viewing angle
    auto view_control = visualizer.GetViewControl();
    view_control.SetFront(Eigen::Vector3d(0.0, -1.0, -0.3));
    view_control.SetLookat(Eigen::Vector3d(0.0, 0.0, 0.0));
    view_control.SetUp(Eigen::Vector3d(0.0, 0.0, 1.0));
    view_control.SetZoom(0.8);

    // Add keyboard callback for additional controls
    visualizer.RegisterKeyCallback(GLFW_KEY_R, [&](visualization::Visualizer* vis) -> bool {
      std::cout << "Resetting view..." << std::endl;
      view_control.Reset();
      return false;  // false means don't redraw, let the main loop handle it
    });

    visualizer.RegisterKeyCallback(GLFW_KEY_T, [&](visualization::Visualizer* vis) -> bool {
      std::cout << "Toggling center points visibility..." << std::endl;
      static bool centers_visible = true;
      if (centers_visible) {
        vis->RemoveGeometry(geo_centers, false);
      } else {
        vis->AddGeometry(geo_centers, false);
      }
      centers_visible = !centers_visible;
      return true;  // true means redraw
    });

    // Enable mouse controls and run
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
  }

  return 0;
}