#include <iostream>
#include "serialize.hpp"
#include "convert_dataset.hpp"
#include "bow_dictionary.hpp"

namespace fs = std::filesystem;

const int KMEANS_MAX_ITER = 30;
const int KMEANS_DICT_SIZE = 1000;

int main(){
  std::cout<<"Converting full dataset..."<<std::endl;
  fs::path p_full("../data/freiburg-full/images");
  std::cout<<"Image path: "<<p_full<<std::endl;
  ipb::serialization::sifts::ConvertDataset(p_full);
  auto descriptors = ipb::serialization::sifts::LoadDataset(fs::path("../data/freiburg-full/bin"));
  std::cout<<"Loaded descriptor: "<<descriptors.size()<<std::endl;

  auto dict = ipb::BowDictionary::GetInstance();
  dict.build(KMEANS_MAX_ITER, KMEANS_DICT_SIZE, descriptors);

  auto vocab = dict.vocabulary();
  std::cout<<"Dict vocabulary: "<<std::endl;
  std::cout<<"Rows: "<<vocab.rows<<std::endl;
  std::cout<<"Cols: "<<vocab.cols<<std::endl;

  ipb::serialization::Serialize(vocab, "../data/vocabulary.bin");
}
