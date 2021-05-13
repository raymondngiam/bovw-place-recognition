#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <fmt/core.h>

namespace fs = std::filesystem;

namespace {
  const fs::path image_path = fs::path("../data/freiburg/images/");
  const std::string image_prefix = "imageCompressedCam0_0000";
} // namespace
  
int main(){  
  std::cout<<std::endl;
  std::cout<<"Testing SIFT"<<std::endl;
  
  auto detector = cv::SIFT::create();
  
  std::vector<std::string> image_indices {"000","300","700"};
  
  for (const auto& index : image_indices){
    auto image_name = fmt::format("{}{}.png",image_prefix,index);
    auto full_image_path = image_path / fs::path(image_name);
    auto scene = cv::imread(full_image_path, cv::IMREAD_COLOR);
  
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    detector->detectAndCompute(scene, cv::Mat(), keypoints, descriptor);
    cv::Mat img_keypoints;
    cv::drawKeypoints(scene, keypoints, img_keypoints );
    auto image_out_path = fmt::format("../images/scene_sift_{}.png",index);
    cv::imwrite(image_out_path, img_keypoints);
    std::cout<<"Saved image to "<< image_out_path <<std::endl;
  }
  return 0;
}
