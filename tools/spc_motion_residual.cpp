#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace {

constexpr int kChannels = 4;
constexpr int kResidualOffset = 32768;

struct Geometry {
  int width = 0;
  int height = 0;
  int plane_width = 0;
  int plane_height = 0;
};

struct PamImage {
  int width = 0;
  int height = 0;
  int depth = 0;
  int maxval = 0;
  std::vector<uint16_t> data;
};

Geometry MakeGeometry(int width, int height) {
  if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) {
    throw std::runtime_error("RAW dimensions must be positive even numbers");
  }
  return Geometry{width, height, width / 2, height / 2};
}

std::vector<uint16_t> ReadRaw16Le(const std::string& path, const Geometry& geom) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("failed to open input: " + path);
  std::vector<uint16_t> data(static_cast<size_t>(geom.width) * geom.height);
  for (uint16_t& value : data) {
    unsigned char bytes[2];
    f.read(reinterpret_cast<char*>(bytes), 2);
    if (!f) throw std::runtime_error("short read: " + path);
    value = static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
  }
  return data;
}

void WriteRaw16Le(const std::string& path, const std::vector<uint16_t>& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open output: " + path);
  for (uint16_t value : data) {
    const unsigned char bytes[2] = {
        static_cast<unsigned char>(value & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
    };
    out.write(reinterpret_cast<const char*>(bytes), 2);
  }
}

std::vector<cv::Mat> SplitRggb(const std::vector<uint16_t>& raw, const Geometry& geom) {
  std::vector<cv::Mat> planes;
  for (int c = 0; c < kChannels; ++c) {
    planes.emplace_back(geom.plane_height, geom.plane_width, CV_16UC1);
  }
  for (int y = 0; y < geom.plane_height; ++y) {
    const int src_y0 = y * 2;
    const int src_y1 = src_y0 + 1;
    auto* p0 = planes[0].ptr<uint16_t>(y);
    auto* p1 = planes[1].ptr<uint16_t>(y);
    auto* p2 = planes[2].ptr<uint16_t>(y);
    auto* p3 = planes[3].ptr<uint16_t>(y);
    for (int x = 0; x < geom.plane_width; ++x) {
      const int src_x0 = x * 2;
      const int src_x1 = src_x0 + 1;
      p0[x] = raw[static_cast<size_t>(src_y0) * geom.width + src_x0];
      p1[x] = raw[static_cast<size_t>(src_y0) * geom.width + src_x1];
      p2[x] = raw[static_cast<size_t>(src_y1) * geom.width + src_x0];
      p3[x] = raw[static_cast<size_t>(src_y1) * geom.width + src_x1];
    }
  }
  return planes;
}

std::vector<uint16_t> MergeRggb(const std::vector<cv::Mat>& planes, const Geometry& geom) {
  std::vector<uint16_t> raw(static_cast<size_t>(geom.width) * geom.height);
  for (int y = 0; y < geom.plane_height; ++y) {
    const uint16_t* p0 = planes[0].ptr<uint16_t>(y);
    const uint16_t* p1 = planes[1].ptr<uint16_t>(y);
    const uint16_t* p2 = planes[2].ptr<uint16_t>(y);
    const uint16_t* p3 = planes[3].ptr<uint16_t>(y);
    for (int x = 0; x < geom.plane_width; ++x) {
      const int dst_y0 = y * 2;
      const int dst_y1 = dst_y0 + 1;
      const int dst_x0 = x * 2;
      const int dst_x1 = dst_x0 + 1;
      raw[static_cast<size_t>(dst_y0) * geom.width + dst_x0] = p0[x];
      raw[static_cast<size_t>(dst_y0) * geom.width + dst_x1] = p1[x];
      raw[static_cast<size_t>(dst_y1) * geom.width + dst_x0] = p2[x];
      raw[static_cast<size_t>(dst_y1) * geom.width + dst_x1] = p3[x];
    }
  }
  return raw;
}

cv::Mat MakeAlignmentImage(const std::vector<cv::Mat>& planes, const Geometry& geom) {
  cv::Mat acc(geom.plane_height, geom.plane_width, CV_32FC1, cv::Scalar(0));
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

std::vector<cv::Mat> BuildPredictor(const std::vector<cv::Mat>& base_planes,
                                    const cv::Mat& matrix,
                                    int motion_mode) {
  std::vector<cv::Mat> predictor;
  predictor.reserve(base_planes.size());
  for (const cv::Mat& plane : base_planes) {
    predictor.push_back(WarpPlane(plane, matrix, motion_mode));
  }
  return predictor;
}

void WritePamResidual(const std::string& path,
                      const Geometry& geom,
                      const std::vector<cv::Mat>& target,
                      const std::vector<cv::Mat>& predictor,
                      int* min_residual,
                      int* max_residual) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open output: " + path);
  out << "P7\n"
      << "WIDTH " << geom.plane_width << "\n"
      << "HEIGHT " << geom.plane_height << "\n"
      << "DEPTH 4\n"
      << "MAXVAL 65535\n"
      << "TUPLTYPE RGB_ALPHA\n"
      << "ENDHDR\n";

  int local_min = std::numeric_limits<int>::max();
  int local_max = std::numeric_limits<int>::min();
  for (int y = 0; y < geom.plane_height; ++y) {
    const uint16_t* t[kChannels];
    const uint16_t* p[kChannels];
    for (int c = 0; c < kChannels; ++c) {
      t[c] = target[c].ptr<uint16_t>(y);
      p[c] = predictor[c].ptr<uint16_t>(y);
    }
    for (int x = 0; x < geom.plane_width; ++x) {
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

PamImage ReadPam(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open PAM: " + path);

  std::string line;
  std::getline(in, line);
  if (line != "P7") throw std::runtime_error("unsupported PAM magic");

  PamImage image;
  while (std::getline(in, line)) {
    if (line == "ENDHDR") break;
    std::istringstream iss(line);
    std::string key;
    iss >> key;
    if (key == "WIDTH") iss >> image.width;
    else if (key == "HEIGHT") iss >> image.height;
    else if (key == "DEPTH") iss >> image.depth;
    else if (key == "MAXVAL") iss >> image.maxval;
  }
  if (image.width <= 0 || image.height <= 0 || image.depth != 4 || image.maxval != 65535) {
    throw std::runtime_error("unsupported PAM geometry");
  }

  const size_t count = static_cast<size_t>(image.width) * image.height * image.depth;
  image.data.resize(count);
  for (uint16_t& value : image.data) {
    unsigned char bytes[2];
    in.read(reinterpret_cast<char*>(bytes), 2);
    if (!in) throw std::runtime_error("short PAM data");
    value = static_cast<uint16_t>((bytes[0] << 8) | bytes[1]);
  }
  return image;
}

cv::Mat IdentityMatrix() {
  return cv::Mat::eye(2, 3, CV_32F);
}

std::string MatrixToString(const cv::Mat& matrix) {
  std::ostringstream out;
  out << std::setprecision(9);
  for (int y = 0; y < matrix.rows; ++y) {
    for (int x = 0; x < matrix.cols; ++x) {
      if (x) out << ",";
      out << matrix.at<float>(y, x);
    }
    if (y + 1 != matrix.rows) out << ";";
  }
  return out.str();
}

cv::Mat ParseAffineMatrix(const std::string& text) {
  cv::Mat matrix(2, 3, CV_32F);
  std::stringstream rows(text);
  std::string row;
  int y = 0;
  while (std::getline(rows, row, ';')) {
    std::stringstream cols(row);
    std::string col;
    int x = 0;
    while (std::getline(cols, col, ',')) {
      if (y >= 2 || x >= 3) throw std::runtime_error("invalid affine matrix");
      matrix.at<float>(y, x) = std::stof(col);
      ++x;
    }
    if (x != 3) throw std::runtime_error("invalid affine matrix");
    ++y;
  }
  if (y != 2) throw std::runtime_error("invalid affine matrix");
  return matrix;
}

void PrintKeyValue(const std::string& key, const std::string& value) {
  std::cout << key << "=" << value << "\n";
}

void Encode(int argc, char** argv) {
  if (argc != 8) {
    throw std::runtime_error("usage: spc_motion_residual encode MODE WIDTH HEIGHT BASE_RAW TARGET_RAW OUT_PAM");
  }
  const std::string mode = argv[2];
  const Geometry geom = MakeGeometry(std::stoi(argv[3]), std::stoi(argv[4]));
  const auto base_raw = ReadRaw16Le(argv[5], geom);
  const auto target_raw = ReadRaw16Le(argv[6], geom);
  const auto base_planes = SplitRggb(base_raw, geom);
  const auto target_planes = SplitRggb(target_raw, geom);

  cv::Mat warp_matrix = IdentityMatrix();
  int motion_mode = cv::MOTION_AFFINE;
  double score = 0.0;
  std::string status = "ok";

  if (mode == "none") {
    motion_mode = cv::MOTION_TRANSLATION;
  } else if (mode == "translation") {
    const cv::Mat base_align = MakeAlignmentImage(base_planes, geom);
    const cv::Mat target_align = MakeAlignmentImage(target_planes, geom);
    double response = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(base_align, target_align, cv::noArray(), &response);
    warp_matrix.at<float>(0, 2) = static_cast<float>(shift.x);
    warp_matrix.at<float>(1, 2) = static_cast<float>(shift.y);
    motion_mode = cv::MOTION_TRANSLATION;
    score = response;
  } else if (mode == "ecc_affine") {
    cv::Mat base_align = MakeAlignmentImage(base_planes, geom);
    cv::Mat target_align = MakeAlignmentImage(target_planes, geom);
    cv::normalize(base_align, base_align, 0.0, 1.0, cv::NORM_MINMAX);
    cv::normalize(target_align, target_align, 0.0, 1.0, cv::NORM_MINMAX);
    try {
      score = cv::findTransformECC(
          target_align, base_align, warp_matrix, cv::MOTION_AFFINE,
          cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-7),
          cv::noArray(), 5);
    } catch (const cv::Exception& e) {
      status = std::string("ecc_failed: ") + e.what();
      warp_matrix = IdentityMatrix();
    }
    motion_mode = cv::MOTION_AFFINE;
  } else {
    throw std::runtime_error("unknown mode: " + mode);
  }

  const std::vector<cv::Mat> predictor =
      mode == "none" ? base_planes : BuildPredictor(base_planes, warp_matrix, motion_mode);

  int min_residual = 0;
  int max_residual = 0;
  WritePamResidual(argv[7], geom, target_planes, predictor, &min_residual, &max_residual);

  PrintKeyValue("status", status);
  PrintKeyValue("score", std::to_string(score));
  PrintKeyValue("matrix", MatrixToString(warp_matrix));
  PrintKeyValue("residual_min", std::to_string(min_residual));
  PrintKeyValue("residual_max", std::to_string(max_residual));
  PrintKeyValue("offset", std::to_string(kResidualOffset));
}

void Restore(int argc, char** argv) {
  if (argc != 9) {
    throw std::runtime_error("usage: spc_motion_residual restore MODE WIDTH HEIGHT MATRIX BASE_RAW RESIDUAL_PAM OUT_RAW");
  }
  const std::string mode = argv[2];
  const Geometry geom = MakeGeometry(std::stoi(argv[3]), std::stoi(argv[4]));
  const cv::Mat warp_matrix = ParseAffineMatrix(argv[5]);
  const auto base_raw = ReadRaw16Le(argv[6], geom);
  const auto base_planes = SplitRggb(base_raw, geom);
  const PamImage residual = ReadPam(argv[7]);
  if (residual.width != geom.plane_width || residual.height != geom.plane_height) {
    throw std::runtime_error("PAM dimensions do not match RAW geometry");
  }

  int motion_mode = cv::MOTION_AFFINE;
  if (mode == "none" || mode == "translation") {
    motion_mode = cv::MOTION_TRANSLATION;
  } else if (mode != "ecc_affine") {
    throw std::runtime_error("unknown mode: " + mode);
  }
  const std::vector<cv::Mat> predictor =
      mode == "none" ? base_planes : BuildPredictor(base_planes, warp_matrix, motion_mode);

  std::vector<cv::Mat> target_planes;
  for (int c = 0; c < kChannels; ++c) {
    target_planes.emplace_back(geom.plane_height, geom.plane_width, CV_16UC1);
  }
  for (int y = 0; y < geom.plane_height; ++y) {
    uint16_t* out[kChannels];
    const uint16_t* pred[kChannels];
    for (int c = 0; c < kChannels; ++c) {
      out[c] = target_planes[c].ptr<uint16_t>(y);
      pred[c] = predictor[c].ptr<uint16_t>(y);
    }
    for (int x = 0; x < geom.plane_width; ++x) {
      const size_t base_idx = (static_cast<size_t>(y) * geom.plane_width + x) * kChannels;
      for (int c = 0; c < kChannels; ++c) {
        const int residual_value = static_cast<int>(residual.data[base_idx + c]) - kResidualOffset;
        const int restored = static_cast<int>(pred[c][x]) + residual_value;
        if (restored < 0 || restored > 65535) {
          throw std::runtime_error("restored RAW value outside uint16 range");
        }
        out[c][x] = static_cast<uint16_t>(restored);
      }
    }
  }

  WriteRaw16Le(argv[8], MergeRggb(target_planes, geom));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: spc_motion_residual encode|restore ...\n";
      return 2;
    }
    const std::string command = argv[1];
    if (command == "encode") {
      Encode(argc, argv);
    } else if (command == "restore") {
      Restore(argc, argv);
    } else {
      throw std::runtime_error("unknown command: " + command);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
