#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace {

constexpr int kWidth = 8288;
constexpr int kHeight = 5520;
constexpr int kPlaneWidth = kWidth / 2;
constexpr int kPlaneHeight = kHeight / 2;
constexpr int kChannels = 4;
constexpr int kResidualOffset = 32768;

std::vector<uint16_t> ReadRaw16(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("failed to open input: " + path);
  std::vector<uint16_t> data(static_cast<size_t>(kWidth) * kHeight);
  f.read(reinterpret_cast<char*>(data.data()),
         static_cast<std::streamsize>(data.size() * sizeof(uint16_t)));
  if (f.gcount() != static_cast<std::streamsize>(data.size() * sizeof(uint16_t))) {
    throw std::runtime_error("short read: " + path);
  }
  return data;
}

std::vector<cv::Mat> SplitRggb(const std::vector<uint16_t>& raw) {
  std::vector<cv::Mat> planes;
  for (int c = 0; c < kChannels; ++c) {
    planes.emplace_back(kPlaneHeight, kPlaneWidth, CV_16UC1);
  }
  for (int y = 0; y < kPlaneHeight; ++y) {
    const int src_y0 = y * 2;
    const int src_y1 = src_y0 + 1;
    auto* p0 = planes[0].ptr<uint16_t>(y);
    auto* p1 = planes[1].ptr<uint16_t>(y);
    auto* p2 = planes[2].ptr<uint16_t>(y);
    auto* p3 = planes[3].ptr<uint16_t>(y);
    for (int x = 0; x < kPlaneWidth; ++x) {
      const int src_x0 = x * 2;
      const int src_x1 = src_x0 + 1;
      p0[x] = raw[static_cast<size_t>(src_y0) * kWidth + src_x0];
      p1[x] = raw[static_cast<size_t>(src_y0) * kWidth + src_x1];
      p2[x] = raw[static_cast<size_t>(src_y1) * kWidth + src_x0];
      p3[x] = raw[static_cast<size_t>(src_y1) * kWidth + src_x1];
    }
  }
  return planes;
}

cv::Mat MakeAlignmentImage(const std::vector<cv::Mat>& planes) {
  cv::Mat acc(kPlaneHeight, kPlaneWidth, CV_32FC1, cv::Scalar(0));
  for (const cv::Mat& plane : planes) {
    cv::Mat f;
    plane.convertTo(f, CV_32FC1);
    acc += f;
  }
  acc *= 0.25f;
  return acc;
}

cv::Mat WarpPlane(const cv::Mat& source, const cv::Mat& matrix, int mode) {
  cv::Mat warped;
  if (mode == cv::MOTION_HOMOGRAPHY) {
    cv::warpPerspective(source, warped, matrix, source.size(),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);
  } else {
    cv::warpAffine(source, warped, matrix, source.size(),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
  }
  return warped;
}

void WritePamResidual(const std::string& path,
                      const std::vector<cv::Mat>& target,
                      const std::vector<cv::Mat>& predictor,
                      int* min_residual,
                      int* max_residual) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open output: " + path);
  out << "P7\n"
      << "WIDTH " << kPlaneWidth << "\n"
      << "HEIGHT " << kPlaneHeight << "\n"
      << "DEPTH 4\n"
      << "MAXVAL 65535\n"
      << "TUPLTYPE RGB_ALPHA\n"
      << "ENDHDR\n";

  int local_min = std::numeric_limits<int>::max();
  int local_max = std::numeric_limits<int>::min();
  for (int y = 0; y < kPlaneHeight; ++y) {
    const uint16_t* t[kChannels];
    const uint16_t* p[kChannels];
    for (int c = 0; c < kChannels; ++c) {
      t[c] = target[c].ptr<uint16_t>(y);
      p[c] = predictor[c].ptr<uint16_t>(y);
    }
    for (int x = 0; x < kPlaneWidth; ++x) {
      for (int c = 0; c < kChannels; ++c) {
        const int residual = static_cast<int>(t[c][x]) - static_cast<int>(p[c][x]);
        local_min = std::min(local_min, residual);
        local_max = std::max(local_max, residual);
        const int coded = residual + kResidualOffset;
        if (coded < 0 || coded > 65535) {
          throw std::runtime_error("residual outside uint16 offset range");
        }
        const uint16_t be = static_cast<uint16_t>(coded);
        const unsigned char bytes[2] = {
            static_cast<unsigned char>((be >> 8) & 0xff),
            static_cast<unsigned char>(be & 0xff),
        };
        out.write(reinterpret_cast<const char*>(bytes), 2);
      }
    }
  }
  *min_residual = local_min;
  *max_residual = local_max;
}

void PrintMatrix(const cv::Mat& matrix) {
  for (int y = 0; y < matrix.rows; ++y) {
    for (int x = 0; x < matrix.cols; ++x) {
      if (x) std::cout << ",";
      std::cout << matrix.at<float>(y, x);
    }
    std::cout << (y + 1 == matrix.rows ? "" : ";");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: measure_residual_jxl MODE BASE_RAW TARGET_RAW OUT_PAM\n";
      return 2;
    }
    const std::string mode = argv[1];
    const std::string base_path = argv[2];
    const std::string target_path = argv[3];
    const std::string out_path = argv[4];

    const auto base_raw = ReadRaw16(base_path);
    const auto target_raw = ReadRaw16(target_path);
    const auto base_planes = SplitRggb(base_raw);
    const auto target_planes = SplitRggb(target_raw);

    std::vector<cv::Mat> predictor = base_planes;
    cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
    std::string status = "ok";
    double score = 0.0;

    if (mode == "none") {
      // Keep identity predictor.
    } else if (mode == "translation") {
      const cv::Mat base_align = MakeAlignmentImage(base_planes);
      const cv::Mat target_align = MakeAlignmentImage(target_planes);
      double response = 0.0;
      const cv::Point2d shift = cv::phaseCorrelate(base_align, target_align, cv::noArray(), &response);
      warp_matrix.at<float>(0, 2) = static_cast<float>(shift.x);
      warp_matrix.at<float>(1, 2) = static_cast<float>(shift.y);
      score = response;
      predictor.clear();
      for (const cv::Mat& plane : base_planes) {
        predictor.push_back(WarpPlane(plane, warp_matrix, cv::MOTION_TRANSLATION));
      }
    } else if (mode == "ecc_affine") {
      cv::Mat base_align = MakeAlignmentImage(base_planes);
      cv::Mat target_align = MakeAlignmentImage(target_planes);
      cv::normalize(base_align, base_align, 0.0, 1.0, cv::NORM_MINMAX);
      cv::normalize(target_align, target_align, 0.0, 1.0, cv::NORM_MINMAX);
      try {
        score = cv::findTransformECC(
            target_align, base_align, warp_matrix, cv::MOTION_AFFINE,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-7),
            cv::noArray(), 5);
      } catch (const cv::Exception& e) {
        status = std::string("ecc_failed: ") + e.what();
      }
      predictor.clear();
      for (const cv::Mat& plane : base_planes) {
        predictor.push_back(WarpPlane(plane, warp_matrix, cv::MOTION_AFFINE));
      }
    } else {
      throw std::runtime_error("unknown mode: " + mode);
    }

    int min_residual = 0;
    int max_residual = 0;
    WritePamResidual(out_path, target_planes, predictor, &min_residual, &max_residual);

    std::cout << "mode=" << mode << "\n";
    std::cout << "status=" << status << "\n";
    std::cout << "score=" << score << "\n";
    std::cout << "matrix=";
    PrintMatrix(warp_matrix);
    std::cout << "\n";
    std::cout << "residual_min=" << min_residual << "\n";
    std::cout << "residual_max=" << max_residual << "\n";
    std::cout << "offset=" << kResidualOffset << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
