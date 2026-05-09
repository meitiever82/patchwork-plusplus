#include <algorithm>
#include <cmath>

#include "patchwork/patchwork.h"

namespace patchwork {

PatchWork::PatchWork(const PatchworkParams& params) : params_(params) {}

double PatchWork::xy2theta(double x, double y) const {
  double a = std::atan2(y, x);
  return (a >= 0) ? a : (a + 2 * M_PI);
}

double PatchWork::xy2radius(double x, double y) const {
  return std::hypot(x, y);
}

void PatchWork::initialize() {
  regionwise_patches_.clear();
  regionwise_patches_.resize(params_.num_zones);
  for (int z = 0; z < params_.num_zones; ++z) {
    regionwise_patches_[z].resize(params_.num_rings_each_zone[z]);
    for (int r = 0; r < params_.num_rings_each_zone[z]; ++r) {
      regionwise_patches_[z][r].resize(params_.num_sectors_each_zone[z]);
    }
  }
}

void PatchWork::flush() {
  for (auto& zone : regionwise_patches_)
    for (auto& ring : zone)
      for (auto& sector : ring)
        sector.clear();
}

void PatchWork::pc2regionwise_patches(const std::vector<PointXYZ>& src) {
  for (int idx = 0; idx < static_cast<int>(src.size()); ++idx) {
    const auto& p = src[idx];
    double r     = xy2radius(p.x, p.y);
    double theta = xy2theta(p.x, p.y);

    if (r < params_.min_range || r > params_.max_range) continue;

    // Determine zone by min_ranges (last index whose min_range <= r)
    int zone = 0;
    for (int z = params_.num_zones - 1; z >= 0; --z) {
      if (r >= params_.min_ranges[z]) { zone = z; break; }
    }
    if (zone < 0 || zone >= params_.num_zones) continue;

    // Within zone, ring index proportional to (r - min_ranges[z]) / ring_width
    double ring_width = (zone + 1 < params_.num_zones)
                        ? (params_.min_ranges[zone + 1] - params_.min_ranges[zone])
                              / params_.num_rings_each_zone[zone]
                        : (params_.max_range - params_.min_ranges[zone])
                              / params_.num_rings_each_zone[zone];
    int ring = std::min<int>(
        static_cast<int>((r - params_.min_ranges[zone]) / ring_width),
        params_.num_rings_each_zone[zone] - 1);

    int sector = std::min<int>(
        static_cast<int>(theta / (2 * M_PI / params_.num_sectors_each_zone[zone])),
        params_.num_sectors_each_zone[zone] - 1);

    regionwise_patches_[zone][ring][sector].push_back(p);
  }
}

void PatchWork::estimate_plane(const std::vector<PointXYZ>& seeds, PCAFeature& out) {
  if (seeds.empty()) return;

  Eigen::MatrixXf pts(seeds.size(), 3);
  for (size_t i = 0; i < seeds.size(); ++i) {
    pts(i, 0) = seeds[i].x;
    pts(i, 1) = seeds[i].y;
    pts(i, 2) = seeds[i].z;
  }
  Eigen::Vector3f mean = pts.colwise().mean();
  Eigen::MatrixXf centered = pts.rowwise() - mean.transpose();
  Eigen::Matrix3f cov = (centered.adjoint() * centered) /
                        std::max<float>(1.0f, static_cast<float>(pts.rows() - 1));

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU);
  out.normal_ = svd.matrixU().col(2);
  if (out.normal_(2) < 0) out.normal_ = -out.normal_;
  out.singular_values_ = svd.singularValues();
  out.mean_  = mean;
  out.d_     = -out.normal_.dot(mean);
  out.th_dist_d_ = static_cast<float>(params_.th_dist) - out.d_;

  const float s0 = out.singular_values_(0);
  const float s1 = out.singular_values_(1);
  const float s2 = out.singular_values_(2);
  const float eps = 1e-12f;
  out.linearity_ = (s0 - s1) / std::max(s0, eps);
  out.planarity_ = (s1 - s2) / std::max(s0, eps);
}

void PatchWork::estimateGround(const Eigen::MatrixXf& /*cloud*/) {}

Eigen::MatrixX3f PatchWork::getGround() const { return ground_mat_; }
Eigen::MatrixX3f PatchWork::getNonground() const { return nonground_mat_; }
std::vector<int> PatchWork::getGroundIndices() const { return ground_idx_; }
std::vector<int> PatchWork::getNongroundIndices() const { return nonground_idx_; }
double PatchWork::getTimeTaken() const { return time_taken_; }
double PatchWork::getHeight() const { return sensor_height_; }

}  // namespace patchwork
