#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "gtest/gtest.h"
#include "serialize.hpp"
#include "utils.hpp"
#include "convert_dataset.hpp"

namespace fs = std::filesystem;
const std::string lenna_path = "../data/lenna.png";
const std::string lenna_bin_path = "../data/lenna.bin";

TEST(SerializeTest,SerializeImage){
  auto lenna = cv::imread(lenna_path, cv::IMREAD_GRAYSCALE);
  ipb::serialization::Serialize(lenna, lenna_bin_path);
  EXPECT_EQ(fs::exists(lenna_bin_path),true);
  fs::remove(lenna_bin_path);
}

TEST(SerializeTest,DeserializeImage){
  auto lenna = cv::imread(lenna_path, cv::IMREAD_GRAYSCALE);
  ipb::serialization::Serialize(lenna, lenna_bin_path);
  auto lenna_bin = ipb::serialization::Deserialize(lenna_bin_path);
  fs::remove(lenna_bin_path);
  EXPECT_EQ(lenna_bin.empty(),false);
  EXPECT_EQ(lenna.size(),lenna_bin.size());
  EXPECT_EQ(mat_are_equal<uchar>(lenna, lenna_bin),true);
}

TEST(ConvertDatasetTest,ComputeSIFTSAndSerialize){
  const std::string img_path = "../data/freiburg/images/";
  const std::string bin_path = "../data/freiburg/bin/";
  ipb::serialization::sifts::ConvertDataset(img_path);
  for (const auto& entry : fs::directory_iterator(img_path)) {
    const auto& stem = entry.path().stem().string();
    const auto& extension = entry.path().extension();
    if (extension == ".png") {
      const auto& descriptors_filename = bin_path + stem + ".bin";
      EXPECT_EQ(fs::exists(descriptors_filename),true);
    }
  }
}

TEST(ConvertDatasetTest,DeserializeSIFTSFromBinaries){
  const std::string bin_path = "../data/freiburg/bin/";
  auto descriptors = ipb::serialization::sifts::LoadDataset(bin_path);
  for (const auto& descriptor : descriptors) {
    EXPECT_EQ(descriptor.empty(),false);
  }
  fs::remove_all(bin_path);
}
