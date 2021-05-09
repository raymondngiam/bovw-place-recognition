#include "bow_dictionary.hpp"

using namespace ipb;

BowDictionary* BowDictionary::instance = nullptr;

int BowDictionary::max_iterations() const {
  return _maxIter; 
}

int BowDictionary::size() const {
  return _dictionary.rows; 
}

std::vector<cv::Mat> BowDictionary::descriptors() const {
  return _inputDescriptors; 
}

cv::Mat BowDictionary::vocabulary() const {
  return _dictionary; 
}

void BowDictionary::set_max_iterations(int value){
  _maxIter = value;
  _dictionary = ipb::kMeans(_inputDescriptors, _dictSize, _maxIter);
}

void BowDictionary::set_size(int value){
  _dictSize = value;
  _dictionary = ipb::kMeans(_inputDescriptors, _dictSize, _maxIter);
}

void BowDictionary::set_descriptors(std::vector<cv::Mat> value){
  _inputDescriptors = value;
  _dictionary = ipb::kMeans(_inputDescriptors, _dictSize, _maxIter);
}

void BowDictionary::set_params(int max_iter, int size, std::vector<cv::Mat> descriptors){
  _maxIter = max_iter;
  _dictSize = size;
  _inputDescriptors = descriptors;
  _dictionary = ipb::kMeans(_inputDescriptors, _dictSize, _maxIter);
}

int BowDictionary::total_features() const{
  int length=0;
  for (const auto& d : _inputDescriptors){
    length+=d.rows;
  }
  return length;
}

BowDictionary& BowDictionary::GetInstance(){
  if (instance==nullptr){
    instance = new BowDictionary();
  }
  return *instance;
}

bool BowDictionary::empty(){
  return _dictionary.rows==0;
}

