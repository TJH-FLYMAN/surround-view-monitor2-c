#include "common.h"
#include "prms.hpp"
#include <array>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <vector>

void undist_by_remap(const cv::Mat &src, cv::Mat &dst, const CameraPrms &prms) {
  // get new camera matrix
  cv::Mat new_camera_matrix = prms.camera_matrix.clone();
  double *matrix_data = (double *)new_camera_matrix.data;

  const auto scale = (const float *)(prms.scale_xy.data);
  const auto shift = (const float *)(prms.shift_xy.data);

  if (!matrix_data || !scale || !shift) {
    return;
  }

  matrix_data[0] *= (double)scale[0];
  matrix_data[3 * 1 + 1] *= (double)scale[1];
  matrix_data[2] += (double)shift[0];
  matrix_data[1 * 3 + 2] += (double)shift[1];
  // std::cout << new_camera_matrix;
  // undistort
  cv::Mat map1, map2;
  cv::fisheye::initUndistortRectifyMap(prms.camera_matrix, prms.dist_coff,
                                       cv::Mat(), new_camera_matrix, prms.size,
                                       CV_16SC2, map1, map2);

  cv::remap(src, dst, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}
void rgb_info_statics(cv::Mat &src, BgrSts &sts) {
  int nums = src.rows * src.cols;

  sts.b = sts.r = sts.g = 0;

  for (int h = 0; h < src.rows; ++h) {
    uchar *uc_pixel = src.data + h * src.step;
    for (int w = 0; w < src.cols; ++w) {
      sts.b += uc_pixel[0];
      sts.g += uc_pixel[1];
      sts.r += uc_pixel[2];
      uc_pixel += 3;
    }
  }

  sts.b /= nums;
  sts.r /= nums;
  sts.g /= nums;
}

// r g b digtial gain
void rgb_dgain(cv::Mat &src, float r_gain, float g_gain, float b_gain) {
  if (src.empty()) {
    return;
  }
  for (int h = 0; h < src.rows; ++h) {
    uchar *uc_pixel = src.data + h * src.step;
    for (int w = 0; w < src.cols; ++w) {
      uc_pixel[0] = clip<uint8_t>(uc_pixel[0] * b_gain, 255);
      uc_pixel[1] = clip<uint8_t>(uc_pixel[1] * g_gain, 255);
      uc_pixel[2] = clip<uint8_t>(uc_pixel[2] * r_gain, 255);
      uc_pixel += 3;
    }
  }
}

bool read_prms(const std::string &path, CameraPrms &prms) {
  cv::FileStorage fs(path, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    throw std::string("error open file");
    return false;
  }
  prms.camera_matrix = fs["camera_matrix"].mat();
  prms.dist_coff = fs["dist_coeffs"].mat();
  prms.project_matrix = fs["project_matrix"].mat();
  prms.shift_xy = fs["shift_xy"].mat();
  prms.scale_xy = fs["scale_xy"].mat();
  auto size_ = fs["resolution"].mat();
  prms.size = cv::Size(size_.at<int>(0), size_.at<int>(1));

  fs.release();

  return true;
}

void merge_image(cv::Mat src1, cv::Mat src2, cv::Mat w, cv::Mat out) {
  if (src1.size() != src2.size()) {
    return;
  }

  int p_index = 0;
  auto *weights = (float *)(w.data);
  for (int h = 0; h < src1.rows; ++h) {
    uchar *p1 = src1.data + h * src1.step;
    uchar *p2 = src2.data + h * src2.step;
    uchar *o = out.data + h * out.step;
    for (int w = 0; w < src1.cols; ++w) {
      o[0] = clip<uint8_t>(
          p1[0] * weights[p_index] + p2[0] * (1 - weights[p_index]), 255);
      o[1] = clip<uint8_t>(
          p1[1] * weights[p_index] + p2[1] * (1 - weights[p_index]), 255);
      o[2] = clip<uint8_t>(
          p1[2] * weights[p_index] + p2[2] * (1 - weights[p_index]), 255);
      p1 += 3;
      p2 += 3;
      o += 3;
      ++p_index;
    }
  }
}

void awb_and_lum_banlance(std::vector<cv::Mat *> srcs) {
  BgrSts sts[4]; // b g r
  int gray[4] = {0, 0, 0, 0};
  float gray_ave = 0;

  if (srcs.size() != 4) {
    return;
  }

  for (int i = 0; i < 4; ++i) {
    if (srcs[i] == nullptr) {
      return;
    }
    rgb_info_statics(*srcs[i], sts[i]); //计算rgb三通道均值
    gray[i] = sts[i].r * 20 + sts[i].g * 60 + sts[i].b; //加权计算
    gray_ave += gray[i];
  }

  gray_ave /= 4; //计算色图像像素灰度均值的平均值
  //计算每个通道的增益系数r/g/b_gain，进行白平衡和亮度均衡处理
  for (int i = 0; i < 4; ++i) {
    float lum_gain = gray_ave / gray[i];
    float r_gain = sts[i].g * lum_gain / sts[i].r;
    float g_gain = lum_gain;
    float b_gain = sts[i].g * lum_gain / sts[i].b;
    std::cout << "gains : " << r_gain << " | " << g_gain << " | " << b_gain
              << "\r\n";
    rgb_dgain(*srcs[i], r_gain, g_gain, b_gain); //增益应用到图像像素值
  }
}
auto main() -> int {
  std::array<cv::Mat, 4> origin_dir_img;
  std::array<CameraPrms, 4> prms;
  std::vector<cv::Mat *> srcs;
  std::array<cv::Mat, 4> undist_dir_img;
  std::array<cv::Mat, 4> merge_weights_img;
  cv::Mat out_put_img;
  std::array<float *, 4> w_ptr = {nullptr, nullptr, nullptr, nullptr};
  const std::string weightpath = "/home/tjh/hello/hello-test/yaml/weights.png";
  cv::Mat weights = cv::imread(weightpath, -1);
  if (weights.channels() != 4) {
    std::cerr << "imread weights failed " << weights.channels() << "\r\n";
    return -1;
  }

  for (int i = 0; i < 4; ++i) {
    merge_weights_img[i] =
        cv::Mat(weights.size(), CV_32FC1, cv::Scalar(0, 0, 0));
    w_ptr[i] = (float *)merge_weights_img[i].data;
  }
  int pixel_index = 0;
  for (int h = 0; h < weights.rows; ++h) {
    uchar *uc_pixel = weights.data + h * weights.step;
    for (int w = 0; w < weights.cols; ++w) {
      w_ptr[0][pixel_index] = uc_pixel[0] / 255.0f;
      w_ptr[1][pixel_index] = uc_pixel[1] / 255.0f;
      w_ptr[2][pixel_index] = uc_pixel[2] / 255.0f;
      w_ptr[3][pixel_index] = uc_pixel[3] / 255.0f;
      uc_pixel += 4;
      ++pixel_index;
    }
  }
  out_put_img =
      cv::Mat(cv::Size(total_w, total_h), CV_8UC3, cv::Scalar(0, 0, 0));
  // 1.读内参、畸变系数等数据
  for (int i = 0; i < 4; ++i) {
    auto &prm = prms[i];
    prm.name = camera_names[i];
    auto ok =
        read_prms("/home/tjh/hello/hello-test/yaml/" + prm.name + ".yaml", prm);
    std::cout << ok << std::endl;
    if (!ok) {
      return -1;
    }
  }
  // 2.读图存数据
  for (int i = 0; i < 4; ++i) {
    auto &prm = prms[i];
    prm.name = camera_names[i];
    origin_dir_img[i] =
        cv::imread("/home/tjh/hello/hello-test/images/" + prm.name + ".png",
                   cv::IMREAD_COLOR);
    srcs.push_back(&origin_dir_img[i]);
  }
  // 3.亮度均衡、自动白平衡
  awb_and_lum_banlance(srcs);
  //   for (int i = 0; i < 4; ++i) {
  //     auto &prm = prms[i];
  //     cv::imwrite("/home/tjh/hello/hello-test/images/" + prm.name +
  //                     "_awb_and_lub" + ".png ",
  //                 *srcs[i]);
  //   }
  // 4.去畸变，图像旋转
  for (int i = 0; i < 4; i++) {
    auto &prm = prms[i];
    cv::Mat &src = origin_dir_img[i];
    undist_by_remap(src, src, prm);
    cv::warpPerspective(src, src, prm.project_matrix, project_shapes[prm.name]);
    if (camera_flip_mir[i] == "r+") {
      cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
    } else if (camera_flip_mir[i] == "r-") {
      cv::rotate(src, src, cv::ROTATE_90_COUNTERCLOCKWISE);
    } else if (camera_flip_mir[i] == "m") {
      cv::rotate(src, src, cv::ROTATE_180);
    }
    // display_mat(src, "project");
    // cv::imwrite("/home/tjh/hello/hello-test/images/" + prm.name +
    // "_rotate.png",
    //             src);
    undist_dir_img[i] = src.clone();
  }
  // 5.图像拼接
  // 5.1 拼接非重合区域
  for (int i = 0; i < 4; ++i) {
    cv::Rect roi;
    bool is_cali_roi = false;
    if (std::string(camera_names[i]) == "front") {
      roi = cv::Rect(xl, 0, xr - xl, yt);
      // std::cout << "\nfront" << roi;
      undist_dir_img[i](roi).copyTo(out_put_img(roi));
    } else if (std::string(camera_names[i]) == "left") {
      roi = cv::Rect(0, yt, xl, yb - yt);
      // std::cout << "\nleft" << roi << out_put_img.size();
      undist_dir_img[i](roi).copyTo(out_put_img(roi));
    } else if (std::string(camera_names[i]) == "right") {
      roi = cv::Rect(0, yt, xl, yb - yt);
      // std::cout << "\nright" << roi << out_put_img.size();
      undist_dir_img[i](roi).copyTo(
          out_put_img(cv::Rect(xr, yt, total_w - xr, yb - yt)));
    } else if (std::string(camera_names[i]) == "back") {
      roi = cv::Rect(xl, 0, xr - xl, yt);
      // std::cout << "\nright" << roi << out_put_img.size();
      undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xl, yb, xr - xl, yt)));
    }
  }
  // cv::imwrite("/home/tjh/hello/hello-test/images/output1.png ", out_put_img);
  // 5.2 拼接四个重合角落
  cv::Rect roi;
  //左上
  roi = cv::Rect(0, 0, xl, yt);
  merge_image(undist_dir_img[0](roi), undist_dir_img[1](roi),
              merge_weights_img[2], out_put_img(roi));
  cv::imwrite("/home/tjh/hello/hello-test/images/output2.png ", out_put_img);
  //右上
  roi = cv::Rect(xr, 0, xl, yt);
  merge_image(undist_dir_img[0](roi), undist_dir_img[3](cv::Rect(0, 0, xl, yt)),
              merge_weights_img[1], out_put_img(cv::Rect(xr, 0, xl, yt)));
  cv::imwrite("/home/tjh/hello/hello-test/images/output3.png ", out_put_img);
  //左下
  roi = cv::Rect(0, yb, xl, yt);
  merge_image(undist_dir_img[2](cv::Rect(0, 0, xl, yt)), undist_dir_img[1](roi),
              merge_weights_img[0], out_put_img(roi));
  cv::imwrite("/home/tjh/hello/hello-test/images/output4.png ", out_put_img);
  //右下
  roi = cv::Rect(xr, 0, xl, yt);
  merge_image(undist_dir_img[2](roi),
              undist_dir_img[3](cv::Rect(0, yb, xl, yt)), merge_weights_img[3],
              out_put_img(cv::Rect(xr, yb, xl, yt)));
  cv::imwrite("/home/tjh/hello/hello-test/images/output5.png ", out_put_img);
  //   for (int i = 0; i < 4; ++i) {
  //     auto &prm = prms[i];
  //     cv::imwrite("/home/tjh/hello/hello-test/images/" + prm.name +
  //                     "_warpPerspective" + ".png ",
  //                 *srcs[i]);
  //   }
  // 5.图像按左右后旋转

  return 0;
}