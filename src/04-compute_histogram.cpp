#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "tfidf.hpp"
#include "histogram.hpp"
#include "bow_dictionary.hpp"
#include "serialize.hpp"
#include "../thirdparty/matplotlib-cpp/matplotlibcpp.h"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main(){
  std::cout<<"Loading vocabulary"<<std::endl;
  auto vocab = ipb::serialization::Deserialize("../data/vocabulary.bin");
  std::cout<<"Loaded vocabulary: Rows["<<vocab.rows<<"] Cols["<<vocab.cols<<"]"<<std::endl;

  std::cout<<std::endl;
  std::cout<<"Extracting raw visual word histogram from dataset..."<<std::endl;  
  auto dictionary = ipb::BowDictionary::GetInstance();
  dictionary.set_vocabulary(vocab);

  cv::Mat m_tmp(1,vocab.rows, CV_32SC1);
  {
  std::ofstream f("../data/img_path_full.txt");
  auto descriptor_path = "../data/freiburg-full/bin/";
  for (const auto& entry : fs::directory_iterator(descriptor_path)){
    auto d = ipb::serialization::Deserialize(entry.path());
    ipb::Histogram h(d, dictionary);
    cv::Mat m(1,vocab.rows, CV_32SC1);
    memcpy(m.data, h.data().data(),h.size()*sizeof(int));
    cv::vconcat(m_tmp, m, m_tmp);

    //write img paths to file
    f<<std::string(entry.path())<<std::endl;
  }
  }
  cv::Rect rect(0, 1, m_tmp.cols, m_tmp.rows-1);
  cv::Mat M = m_tmp(rect);
  std::cout<<"Raw histogram size size: rows["<<M.rows<<"] cols["<<M.cols<<"]"<<std::endl;

  std::cout<<std::endl;
  std::cout<<"Performing tf-idf reweighting..."<<std::endl;  
  Tfidf tfidf(M);
  auto hist_tfidf = tfidf.get_full_histograms();
  std::cout<<"Tfidf histogram: rows["<<hist_tfidf.rows<<"] cols["<<hist_tfidf.cols<<"]"<<std::endl;
  auto multiplier_tfidf = tfidf.get_multiplier();
  std::cout<<"Tfidf multiplier: rows["<<multiplier_tfidf.rows<<"] cols["<<multiplier_tfidf.cols<<"]"<<std::endl;

  //saving tfidf into files
  std::cout<<std::endl;
  std::cout<<"Saving serialized file..."<<std::endl;  
  std::string hist_path = "../data/tfidf_hist_full.bin";
  std::string multiplier_path = "../data/tfidf_multiplier_full.bin";
  ipb::serialization::Serialize(hist_tfidf, hist_path);
  ipb::serialization::Serialize(multiplier_tfidf, multiplier_path);
  std::cout<<"hist_tfidf saved to "<<hist_path<<std::endl;
  std::cout<<"multiplier_tfidf saved to "<<multiplier_path<<std::endl;
  
  std::cout<<std::endl;
  std::cout<<"Visualizing histogram..."<<std::endl;  
  std::ifstream f("../data/img_path_full.txt");
  std::string first_image_path;
  std::getline(f,first_image_path);
  std::cout<<"First image path: "<<first_image_path<<std::endl;
  
  //plot raw and tf-idf reweighted histogram
  std::vector<int> x;
  std::vector<int32_t> y_raw;
  std::vector<double> y_reweighted;
  for (int i=0; i<M.cols; i++){
    x.emplace_back(i);
    y_raw.emplace_back(M.at<int32_t>(0,i));
    y_reweighted.emplace_back(hist_tfidf.at<double>(0,i));
  }
  plt::suptitle("Visual Word Histogram for First Image in the Dataset");
  plt::subplot(2, 1, 1);
  plt::plot(x,y_raw,"r-");
  plt::title("Raw Histogram");
  plt::subplot(2, 1, 2);
  plt::plot(x,y_reweighted,"g-");
  plt::title("Tf-idf Reweighted Histogram");
  std::map<std::string,double> subplot_param;
  subplot_param["hspace"]=0.5;
  subplot_param["wspace"]=0.1;
  plt::subplots_adjust(subplot_param);
  plt::save("../images/BOVW_histogram.png");
  plt::show();
}

