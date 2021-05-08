#include "convert_dataset.hpp"
#include "serialize.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

void ipb::serialization::sifts::ConvertDataset(const fs::path& img_path){
  auto destination_path = img_path / fs::path("..") / fs::path("bin");
  if (!fs::exists(destination_path)){
    fs::create_directory(destination_path);
  }
  int count=0;
  for (const auto& entry : fs::directory_iterator(img_path)){
    auto p= fs::path(entry.path());
    if (p.extension() == ".png"){
      count+=1;
    
      auto tmp = std::string(p.stem()) + std::string(".bin");
      auto out_path = destination_path / fs::path(tmp);

      auto img = cv::imread(p, cv::IMREAD_COLOR);

      auto detector = cv::SIFT::create();
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptor;
      detector->detectAndCompute(img, cv::Mat(), keypoints, descriptor);
      ipb::serialization::Serialize(descriptor, out_path);
    }
  }
  std::cout<<"Processed count: "<<count<<std::endl;
}

std::vector<cv::Mat> ipb::serialization::sifts::LoadDataset(const fs::path& bin_path){
  std::vector<cv::Mat> m;
  for (const auto& entry : fs::directory_iterator(bin_path)){
    auto p= fs::path(entry.path());
    if (p.extension() != ".bin")
      break;
    m.emplace_back(ipb::serialization::Deserialize(p));
  }
  return m;
}
