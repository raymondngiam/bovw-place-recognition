#include "tfidf.hpp"
#include <iostream>

Tfidf::Tfidf(cv::Mat hist){
  hist.convertTo(hist, CV_64F);
  multiplier_ = compute_lg_n_ni(hist);
  
  //std::cout<<"Multiplier: "<<multiplier_<<std::endl;  //debug: multiplier cannot have inf due to div-by-zero
  cv::Mat lg_n_ni = cv::repeat(multiplier_,hist.rows,1);
  
  auto one_per_nd = compute_one_per_nd(hist);
  
  data_ = hist.mul(one_per_nd.mul(lg_n_ni));
}

cv::Mat Tfidf::signum(cv::Mat src){
  cv::Mat z = cv::Mat::zeros(src.size(), src.type()); 
  cv::Mat a = (z < src) & 1;
  cv::Mat b = (src < z) & 1;

  cv::Mat dst;
  cv::addWeighted(a,1.0,b,-1.0,0.0,dst, CV_32F);
  return dst;
}

cv::Mat Tfidf::compute_one_per_nd(cv::Mat src){
  src.convertTo(src, CV_16U);
  cv::Mat_<double> tmp;
  cv::reduce(src, tmp, 1, CV_REDUCE_SUM,CV_64FC1);
  auto nd = cv::repeat(tmp,1,src.cols);
  cv::Mat one_per_nd = 1.0/nd;
  return one_per_nd;
}

cv::Mat Tfidf::compute_lg_n_ni(cv::Mat src){
  int n = src.rows;
  cv::Mat N = cv::Mat(cv::Mat_<double>::ones(1,src.cols)*n);
  
  src.convertTo(src, CV_64F);
  cv::Mat_<double> ni;
  cv::reduce(signum(src), ni, 0, CV_REDUCE_SUM,CV_64FC1);

  // N/ni
  auto tmp = cv::Mat(N/(ni+1));  //IMPORTANT: AFTER SUMMING NI, NEED TO PLUS 1 TO AVOID DIVISION BY ZERO

  // log(N/ni)
  cv::Mat lg;
  cv::log(tmp, lg);
  
  return lg;
}

void Tfidf::set_multiplier(cv::Mat multiplier){
  multiplier_ = multiplier;
}

cv::Mat Tfidf::get_multiplier(){
  return multiplier_;
}

void Tfidf::set_full_histograms(cv::Mat full_hist){
  data_ = full_hist;
}

cv::Mat Tfidf::get_full_histograms(){
  return data_;
}

cv::Mat Tfidf::transform_hist(cv::Mat hist){
  hist.convertTo(hist, CV_64F);
  cv::Mat_<double> tmp;
  cv::reduce(hist, tmp, 1, CV_REDUCE_SUM,CV_64FC1);
  cv::Mat one_per_nd = cv::repeat(1.0/tmp,1,hist.cols);
  int n = hist.rows;
  cv::Mat mult = cv::repeat(multiplier_,n,1);
  return one_per_nd.mul(hist.mul(mult));
}

std::pair<cv::Mat,cv::Mat> Tfidf::query_matches(cv::Mat input, int k){
  auto var = input.mul(input);
  cv::Mat tmp1;
  cv::reduce(var, tmp1, 1, CV_REDUCE_SUM,CV_64FC1);
  cv::Mat norm1;
  cv::sqrt(tmp1, norm1);
  auto var2 = data_.mul(data_);
  cv::Mat tmp2;
  cv::reduce(var2, tmp2, 1, CV_REDUCE_SUM,CV_64FC1);
  cv::Mat norm2;
  cv::sqrt(tmp2, norm2);
  cv::Mat dotProducts = input*(data_.t());
  cv::Mat full_distances = cv::Mat_<double>::zeros(1,data_.rows);
  for (int i = 0; i<dotProducts.rows; i++){
    cv::Mat distances = cv::Mat_<double>::zeros(1,1);
    for (int j = 0; j<dotProducts.cols; j++){
      double val = 1.0 - dotProducts.at<double>(i,j)/(norm1.at<double>(i)*norm2.at<double>(j));
      cv::Mat d = (cv::Mat_<double>(1,1)<<val);
      cv::hconcat(distances, d, distances);
    }
    cv::Rect mask(1,0,distances.cols-1,1);
    auto valid_distances = distances(mask);
    cv::vconcat(full_distances, valid_distances, full_distances);
  }
  cv::Rect mask_full(0,1,full_distances.cols,full_distances.rows-1);
  auto valid_distance_rows = full_distances(mask_full);
  //std::cout<<"Valid distances: "<<valid_distance_rows<<std::endl;  //debug: to examine distance vectors with small dataset
  cv::Mat sorted_indices;
  cv::sortIdx(valid_distance_rows,sorted_indices,CV_SORT_ASCENDING+CV_SORT_EVERY_ROW);
  mask_full = cv::Rect(0,0,k,valid_distance_rows.rows);
  cv::Mat result = sorted_indices(mask_full);
  cv::Mat distances_final = cv::Mat_<double>(result.rows,result.cols);
  for (int i=0; i<result.rows; i++){
    for (int j=0; j<result.cols; j++){
      auto current_col_index = result.at<int>(i,j);
      distances_final.at<double>(i,j) = valid_distance_rows.at<double>(i,current_col_index);
    }
  }
  return std::make_pair(result,distances_final);
}

