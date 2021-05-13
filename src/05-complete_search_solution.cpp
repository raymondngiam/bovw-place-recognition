#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "tfidf.hpp"
#include "histogram.hpp"
#include "bow_dictionary.hpp"
#include "serialize.hpp"
#include "image_browser.hpp"

namespace fs = std::filesystem;

int main(int argc , char const *argv []){
  if (argc!=2){
    std::cout<<"Invalid input argument."<<std::endl;
    std::cout<<"Please input file full path as program argument."<<std::endl;
    return -1;
  }
  
  fs::path input_file = fs::path(std::string(argv[1]));
  if (!fs::exists(input_file)){
    std::cout<<"Invalid file: "<<input_file<<std::endl;
    return -1;
  }

  // program input argument ok, can proceed...

  // load full dataset and image paths
  std::vector<std::string> image_paths;
  fs::path img_path("../data/freiburg-full/images");
  std::ifstream f("../data/img_path_full.txt");
  std::string line;
  while(std::getline(f,line)){
    auto path_bin = fs::path(line);
    auto path_img = std::string(img_path / path_bin.stem()) + std::string(".png");
    image_paths.emplace_back(path_img);
  }
  auto tfidf_hist = ipb::serialization::Deserialize("../data/tfidf_hist_full.bin");
  auto tfidf_multiplier = ipb::serialization::Deserialize("../data/tfidf_multiplier_full.bin");
  Tfidf tfidf;
  tfidf.set_full_histograms(tfidf_hist);
  tfidf.set_multiplier(tfidf_multiplier);

  // compute sift descriptor
  auto detector = cv::SIFT::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptor;
  cv::Mat img = cv::imread(input_file,cv::IMREAD_COLOR);
  detector->detectAndCompute(img, cv::Mat(), keypoints, descriptor);
  
  // load BowDictionary and generate histogram
  auto vocab = ipb::serialization::Deserialize("../data/vocabulary.bin");
  auto dictionary = ipb::BowDictionary::GetInstance();
  dictionary.set_vocabulary(vocab);
  ipb::Histogram h(descriptor, dictionary);

  // tranform raw histogram to tf-idf
  cv::Mat m(1,vocab.rows, CV_32SC1);
  memcpy(m.data, h.data().data(),h.size()*sizeof(int));
  cv::Mat input = tfidf.transform_hist(m);  

  // query first entry from full dataset for 9 closest matches
  std::pair<cv::Mat,cv::Mat> query = tfidf.query_matches(input,9);
  auto indices = query.first;
  auto distances = query.second;
  cv::Mat scores = 1.0f-distances;

  image_browser::ScoredImage im1=std::make_tuple(image_paths[indices.at<int>(0)], scores.at<double>(0));
  image_browser::ScoredImage im2=std::make_tuple(image_paths[indices.at<int>(1)], scores.at<double>(1));
  image_browser::ScoredImage im3=std::make_tuple(image_paths[indices.at<int>(2)], scores.at<double>(2));
  image_browser::ImageRow row1{im1, im2, im3};
  std::vector<image_browser::ImageRow> rows{row1};

  image_browser::ScoredImage im4=std::make_tuple(image_paths[indices.at<int>(3)], scores.at<double>(3));
  image_browser::ScoredImage im5=std::make_tuple(image_paths[indices.at<int>(4)], scores.at<double>(4));
  image_browser::ScoredImage im6=std::make_tuple(image_paths[indices.at<int>(5)], scores.at<double>(5));
  image_browser::ImageRow row2{im4, im5, im6};
  rows.emplace_back(row2);

  image_browser::ScoredImage im7=std::make_tuple(image_paths[indices.at<int>(6)], scores.at<double>(6));
  image_browser::ScoredImage im8=std::make_tuple(image_paths[indices.at<int>(7)], scores.at<double>(7));
  image_browser::ScoredImage im9=std::make_tuple(image_paths[indices.at<int>(8)], scores.at<double>(8));
  image_browser::ImageRow row3{im7, im8, im9};
  rows.emplace_back(row3);

  image_browser::CreateImageBrowser("title", "style.css", rows);

}

