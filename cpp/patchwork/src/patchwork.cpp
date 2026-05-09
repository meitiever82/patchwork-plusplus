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

void PatchWork::estimateGround(const Eigen::MatrixXf& /*cloud*/) {}

Eigen::MatrixX3f PatchWork::getGround() const { return ground_mat_; }
Eigen::MatrixX3f PatchWork::getNonground() const { return nonground_mat_; }
std::vector<int> PatchWork::getGroundIndices() const { return ground_idx_; }
std::vector<int> PatchWork::getNongroundIndices() const { return nonground_idx_; }
double PatchWork::getTimeTaken() const { return time_taken_; }
double PatchWork::getHeight() const { return sensor_height_; }

}  // namespace patchwork
