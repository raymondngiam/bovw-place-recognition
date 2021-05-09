// @file      utils.hpp
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#ifndef UTILS_HPP_
#define UTILS_HPP_
#include <algorithm>
#include <cmath>

#include <opencv2/core/mat.hpp>

template <typename Tp>
bool inline mat_are_equal(const cv::Mat& m1, const cv::Mat& m2) {
  return std::equal(m1.begin<Tp>(), m1.end<Tp>(), m2.begin<Tp>());
}

template <typename Tp>
bool mat_almost_equal(const cv::Mat& m1, const cv::Mat& m2, float epsilon) {
  bool almostEqual = true;
  for (int r=0; r<m1.rows; r++){
    for (int c=0; c<m1.cols; c++){
      bool status = std::fabs(m1.at<Tp>(r,c) - m2.at<Tp>(r,c))< epsilon;
      almostEqual=&status;
    }
  }
  return almostEqual;
}

#endif  // UTILS_HPP_
