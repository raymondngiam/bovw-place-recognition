#pragma once
#include <vector>
#include <set>
#include <utility>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core_c.h>

class Tfidf{
private:
  cv::Mat data_;
  cv::Mat multiplier_;
  cv::Mat signum(cv::Mat src);
  cv::Mat compute_one_per_nd(cv::Mat src);
  cv::Mat compute_lg_n_ni(cv::Mat src);
  cv::Mat remove_unused_words(cv::Mat src, std::vector<int> indices);

public:
  Tfidf() = default;
  Tfidf(cv::Mat hist);
  cv::Mat get_full_histograms();
  cv::Mat get_multiplier();
  void set_full_histograms(cv::Mat full_hist);
  void set_multiplier(cv::Mat multiplier);
  cv::Mat transform_hist(cv::Mat hist);
  std::pair<cv::Mat,cv::Mat>  query_matches(cv::Mat input, int k);
};
