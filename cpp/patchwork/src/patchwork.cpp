#include "patchwork/patchwork.h"

namespace patchwork {

PatchWork::PatchWork(const PatchworkParams& params) : params_(params) {}

void PatchWork::estimateGround(const Eigen::MatrixXf& /*cloud*/) {}

Eigen::MatrixX3f PatchWork::getGround() const { return ground_mat_; }
Eigen::MatrixX3f PatchWork::getNonground() const { return nonground_mat_; }
std::vector<int> PatchWork::getGroundIndices() const { return ground_idx_; }
std::vector<int> PatchWork::getNongroundIndices() const { return nonground_idx_; }
double PatchWork::getTimeTaken() const { return time_taken_; }
double PatchWork::getHeight() const { return sensor_height_; }

}  // namespace patchwork
