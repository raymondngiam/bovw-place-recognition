#include "../thirdparty/matplotlib-cpp/matplotlibcpp.h"
#include <string>
#include <vector>
#include <iostream>
#include "serialize.hpp"
#include "convert_dataset.hpp"
#include "bow_dictionary.hpp"
#include <opencv2/core/mat.hpp>

namespace plt = matplotlibcpp;
using Matf = cv::Mat_<float>;

namespace{
  const int NUM_OF_WORDS = 3;
  std::vector<std::string> plot_formats{"r-","g-","b-"};
} // namespace

int main() {
  Matf vocab = ipb::serialization::Deserialize("../data/vocabulary.bin");
  std::cout<<"Loaded vocabulary: Rows["<<vocab.rows<<"] Cols["<<vocab.cols<<"]"<<std::endl;
  
  // plot words from vocabulary

  for (int i=0; i<NUM_OF_WORDS; i++){
    std::vector<float> x,y;
    for (int j=0; j<vocab.cols; j++){
      x.emplace_back(j);
      y.emplace_back(vocab.at<float>(i,j));
    }
    plt::plot(x,y,plot_formats[i]);
  }
  plt::title("3 words from the vocabulary");  
  plt::save("../images/vocabulary.png");
  plt::show();
}
