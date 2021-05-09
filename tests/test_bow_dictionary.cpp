// @file      test_bow_dictionary.cpp
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include <gtest/gtest.h>
#include <iostream>
#include <filesystem>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "convert_dataset.hpp"
#include "kmeans.hpp"
#include "bow_dictionary.hpp"
#include "utils.hpp"

using ipb::serialization::sifts::ConvertDataset;
using ipb::serialization::sifts::LoadDataset;

namespace {
const int max_iter = 10;
const int dict_size = 1000;
auto& dictionary = ipb::BowDictionary::GetInstance();
}  // namespace

TEST(BowDictionary, Serialization) {
  namespace fs = std::filesystem;
  const std::string img_path = "../data/freiburg/images";
  const std::string bin_path = "../data/freiburg/bin";
  ipb::serialization::sifts::ConvertDataset(img_path);

  for (const auto& entry : fs::directory_iterator(img_path)) {
    const auto& stem = entry.path().stem().string();
    const auto& extension = entry.path().extension();
    if (extension == ".png") {
      const auto& descriptors_filename = (fs::path(bin_path) / stem).string() + ".bin";
      std::cout<<descriptors_filename<<std::endl;
      ASSERT_TRUE(fs::exists(descriptors_filename));
    }
  }
}

TEST(BowDictionary, SetDictionaryParams) {
  const auto descriptors = LoadDataset("../data/freiburg/bin");
  dictionary.set_params(max_iter, dict_size, descriptors);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_EQ(dictionary.size(), dict_size);
}

// Cluster and dataset are identical
TEST(BowDictionary, TransparentCluster) {
  const auto descriptors = LoadDataset("../data/freiburg/bin/");
  int total_features = 0;
  for (const auto& descriptor : descriptors) {
    total_features += descriptor.rows;
  }
  dictionary.set_params(max_iter, total_features, descriptors);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_EQ(dictionary.total_features(), total_features);
}

TEST(BowDictionary, ChangeSizeDictionary) {
  const auto descriptors = LoadDataset("../data/freiburg/bin/");
  dictionary.set_params(max_iter, dict_size, descriptors);
  const int kNewSize = 100;
  dictionary.set_size(kNewSize);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_EQ(dictionary.size(), kNewSize);
}

TEST(BowDictionary, ChangeMaxIterations) {
  const auto descriptors = LoadDataset("../data/freiburg/bin/");
  dictionary.set_params(max_iter, dict_size, descriptors);
  const int kNewIterations = 100;
  dictionary.set_max_iterations(kNewIterations);
  ASSERT_EQ(dictionary.max_iterations(), kNewIterations);
  ASSERT_TRUE(!dictionary.empty());
  ASSERT_EQ(dictionary.size(), dict_size);
}

TEST(BowDictionary, ChangeInputDescriptors) {
  const auto descriptors = LoadDataset("../data/freiburg/bin/");
  dictionary.set_params(max_iter, dict_size, descriptors);

  // copy the first half of the descriptors
  std::vector<cv::Mat> new_descriptors(descriptors.size() / 2);
  new_descriptors.reserve(descriptors.size() / 2);
  for (size_t i = 0; i < descriptors.size() / 2; i++) {
    new_descriptors[i] = descriptors[i];
  }

  dictionary.set_descriptors(new_descriptors);
  ASSERT_NE(dictionary.descriptors().size(), descriptors.size());
  ASSERT_EQ(dictionary.descriptors().size(), new_descriptors.size());
  ASSERT_TRUE(!dictionary.empty());

  for (size_t i = 0; i < descriptors.size() / 2; i++) {
    ASSERT_TRUE(mat_are_equal<uint8_t>(new_descriptors[i],
                                       dictionary.descriptors()[i]));
  }
}
