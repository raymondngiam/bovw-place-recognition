#include <gtest/gtest.h>

#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "tfidf.hpp"
#include "utils.hpp"

namespace {
cv::Mat M = (cv::Mat_<uint16_t>(4,5)<<\
    5,2,1,0,0,\
    4,0,1,1,0,\
    3,1,1,0,2,\
    1,2,1,0,0\
  );
  
cv::Mat hist_result = (cv::Mat_<double>(4,5)<<\
    -0.1395, 0., -0.0279, 0.    , 0.    ,\
    -0.1488, 0., -0.0372, 0.1155, 0.    ,\
    -0.0956, 0., -0.0319, 0.    , 0.1980,\
    -0.0558, 0., -0.0558, 0.    , 0.     \
  );
Tfidf tfidf(M);
}  // namespace

TEST(Tfidf, ComputeHistogram) {
  auto hist_final = tfidf.get_full_histograms();
  std::cout<<hist_result<<std::endl;
  EXPECT_TRUE(mat_almost_equal<double>(hist_final,hist_result,1e-4))      
      << "target histogram:\n"
      << hist_result << "\ncomputed histogram:\n"
      << hist_final;
}

TEST(Tfidf, SingleTransform) {
  cv::Mat M2 = (cv::Mat_<uint16_t>(1,5)<<\
    5,2,1,0,0
  );
  cv::Mat M2_result = (cv::Mat_<double>(1,5)<<\
    -0.1395, 0., -0.0279, 0.    , 0.    
  );
  cv::Mat trans = tfidf.transform_hist(M2);
  EXPECT_TRUE(mat_almost_equal<double>(trans,M2_result,1e-4))      
      << "target histogram:\n"
      << M2_result << "\ncomputed histogram:\n"
      << trans;
}



TEST(Tfidf, MultipleTransform) {
  cv::Mat M2 = (cv::Mat_<uint16_t>(2,5)<<\
    5,2,1,0,0,\
    3,1,1,0,2\
  );
  cv::Mat M2_result = (cv::Mat_<double>(2,5)<<\
    -0.1395, 0., -0.0279, 0.    , 0.    ,\
    -0.0956, 0., -0.0319, 0.    , 0.1980\
  );
  cv::Mat trans = tfidf.transform_hist(M2);
  EXPECT_TRUE(mat_almost_equal<double>(trans,M2_result,1e-4))      
      << "target histogram:\n"
      << M2_result << "\ncomputed histogram:\n"
      << trans;
}

/*
Input are item of index 0 and 3 from the histogram list
*/
TEST(Tfidf, SearchHistogram) {
  cv::Mat M2 = (cv::Mat_<double>(2,5)<<\
    5,2,1,0,0,\
    1,2,1,0,0
  );
  std::vector target_indices {0,3};
  cv::Mat target_distances = (cv::Mat_<double>(2,4)<<\
    0, 0.1679, 0.2022, 0.5499,\
    0, 0.1679, 0.3151, 0.5943\
  );
  cv::Mat trans = tfidf.transform_hist(M2);
  // compute cosine distance
  auto queries = tfidf.query_matches(trans,4);
  auto indices = queries.first;
  auto distances = queries.second;
  std::cout<<"Selected index: \n"<<indices<<std::endl;
  std::cout<<"distances: \n"<<distances<<std::endl;
  EXPECT_TRUE(indices.at<int>(0,0)==target_indices[0])
      << "target index:\n"
      << target_indices[0] << "\ncomputed index:\n"
      << indices.at<int>(0,0);
  EXPECT_TRUE(indices.at<int>(1,0)==target_indices[1])
      << "target index:\n"
      << target_indices[1] << "\ncomputed index:\n"
      << indices.at<int>(1,0);
  EXPECT_TRUE(mat_almost_equal<double>(distances,target_distances,1e-4))      
      << "target histogram:\n"
      << target_distances << "\ncomputed histogram:\n"
      << distances;
}  
  
