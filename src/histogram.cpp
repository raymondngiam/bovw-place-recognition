#include "histogram.hpp"

using namespace ipb;

Histogram::Histogram(){

}

Histogram::Histogram(std::vector<int> data){
  data_ = data;
}

Histogram::Histogram(cv::Mat descriptors, BowDictionary& dictionary){
  if (descriptors.rows==0){
    data_ = std::vector<int>();
    return;
  }

  auto vocab = dictionary.vocabulary();
  data_ = std::vector<int>(vocab.rows,0);
  cv::flann::KDTreeIndexParams index_params;
  cv::flann::Index kdtree(vocab, index_params);
  int k = 1;

  for (int i=0; i<descriptors.rows; i++){
    cv::Mat nearest_vector_idx (1, k, cv::DataType<int>::type);
    cv::Mat nearest_vector_dist (1, k, cv::DataType<float>::type);

    cv::Rect rect(0, i, descriptors.cols, 1);
    auto selections = descriptors(rect);

    kdtree.knnSearch(selections, nearest_vector_idx, nearest_vector_dist, k);
    int id = nearest_vector_idx.at<int>(0);
    data_[id]+=1;
  }
}

int& Histogram::operator[](int pos){
  return data_[pos];
}

const int& Histogram::operator[](int pos) const{
  return data_[pos];
}

std::vector<int> Histogram::data(){
  return data_;
}

int Histogram::size() const{
  return data_.size();
}

bool Histogram::empty() const{
  return data_.size()==0;
}

std::vector<int>::iterator Histogram::begin(){
  return data_.begin();
}

std::vector<int>::iterator Histogram::end(){
  return data_.end();
}

std::vector<int>::const_iterator Histogram::begin() const{
  return data_.cbegin();
}

std::vector<int>::const_iterator Histogram::end() const{
  return data_.cend();
}

const std::vector<int>::const_iterator Histogram::cbegin() const{
  return data_.cbegin();
}

const std::vector<int>::const_iterator Histogram::cend() const{
  return data_.cend();
}

bool Histogram::WriteToCSV(const std::string& path){
  std::ofstream f(path);
  f<<"Index, Frequency\n";
  int count = 0;
  for (const auto& d : data_){
    f<<count<<", "<<d<<std::endl;
    count+=1;
  }
  if (count == data_.size())
    return true;
  return false;
}

Histogram Histogram::ReadFromCSV(const std::string& path){
  auto data = std::vector<int>();
  std::ifstream f(path, std::ios_base::in);
  std::string line;
  std::smatch match;
  std::getline(f,line); //ignore header
  while(std::getline(f,line)){
    std::regex re("(\\w+), (\\w+)");
    if (std::regex_search(line, match, re) && match.size() > 1) {
      data.emplace_back(std::stoi(match.str(2)));
    }
  }
  Histogram hist(data);
  return hist;
}

std::ostream& ipb::operator<<(std::ostream& os, const Histogram& hist)
{
    std::stringstream ss;
    size_t length = hist.data_.size();
    for (size_t i=0; i<length-1; i++){
      ss << hist.data_[i] <<", ";
    }
    ss << hist.data_[length-1];
    auto output = ss.str();
    os << output;
    return os;
}

