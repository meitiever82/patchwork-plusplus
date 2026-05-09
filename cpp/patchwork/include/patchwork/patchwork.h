#ifndef PATCHWORK_CLASSIC_H
#define PATCHWORK_CLASSIC_H

#include <Eigen/Dense>
#include <vector>

namespace patchwork {

struct PatchworkParams {
  bool verbose = false;
};

class PatchWork {
 public:
  PatchWork() = default;
  explicit PatchWork(const PatchworkParams& params);

  void estimateGround(const Eigen::MatrixXf& cloud);

  Eigen::MatrixX3f getGround() const;
  Eigen::MatrixX3f getNonground() const;
  std::vector<int> getGroundIndices() const;
  std::vector<int> getNongroundIndices() const;
  double getTimeTaken() const;
  double getHeight() const;

 private:
  PatchworkParams params_;
  Eigen::MatrixX3f ground_;
  Eigen::MatrixX3f nonground_;
  std::vector<int> ground_idx_;
  std::vector<int> nonground_idx_;
  double time_taken_ = 0.0;
  double sensor_height_ = 1.723;
};

}  // namespace patchwork

#endif
