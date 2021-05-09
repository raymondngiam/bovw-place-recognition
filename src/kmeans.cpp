#include "kmeans.hpp"
#include <iostream>

cv::Mat ipb::kMeans(const std::vector<cv::Mat> &descriptors, int k, int max_iter){
  auto feature_size = descriptors[0].cols;
  // concatenate all descriptors into a cv::Mat
  using Matf = cv::Mat_<float>;
  Matf input = Matf::zeros(1,feature_size);  
  for(const auto& d : descriptors){
    cv::vconcat(input, d, input);
  }
  // remove the first empty row
  cv::Rect rect(0, 1, input.cols, input.rows-1);
  Matf selections = input(rect);
  // execute k-means clustering
  cv::Mat labels, centers;
  cv::TermCriteria termCrit(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS , max_iter,0.1);
  cv::kmeans(selections, k, labels, termCrit, 15, cv::KMEANS_RANDOM_CENTERS, centers);
  // scale to 4 decimal places
  //cv::Mat tmp, centers_scaled;
  //centers.convertTo(tmp, CV_32S, 10000);
  //tmp.convertTo(centers_scaled, CV_32F, 0.0001);
  return centers;
}
