#include "image_browser.hpp"

int main() {

  image_browser::ScoredImage im1=std::make_tuple("../data/000000.png", 0.98f);
  image_browser::ScoredImage im2=std::make_tuple("../data/000100.png", 0.98f);
  image_browser::ScoredImage im3=std::make_tuple("../data/000200.png", 0.98f);
  image_browser::ImageRow row1{im1, im2, im3};
  std::vector<image_browser::ImageRow> rows{row1};

  image_browser::ScoredImage im4=std::make_tuple("../data/000300.png", 0.98f);
  image_browser::ScoredImage im5=std::make_tuple("../data/000400.png", 0.98f);
  image_browser::ScoredImage im6=std::make_tuple("../data/000500.png", 0.98f);
  image_browser::ImageRow row2{im4, im5, im6};
  rows.emplace_back(row2);

  image_browser::ScoredImage im7=std::make_tuple("../data/000600.png", 0.98f);
  image_browser::ScoredImage im8=std::make_tuple("../data/000700.png", 0.98f);
  image_browser::ScoredImage im9=std::make_tuple("../data/000800.png", 0.98f);
  image_browser::ImageRow row3{im7, im8, im9};
  rows.emplace_back(row3);

  image_browser::CreateImageBrowser("title", "style.css", rows);
  return 0;
}
