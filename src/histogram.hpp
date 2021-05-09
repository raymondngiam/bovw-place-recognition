#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
#include <opencv2/core/mat.hpp>
#include "bow_dictionary.hpp"
#include <opencv2/flann.hpp>

namespace ipb{
  class Histogram{
  private:
    std::vector<int> data_;
    
  public:
    Histogram();
    Histogram(std::vector<int> data);
    Histogram(cv::Mat descriptors, BowDictionary& dictionary);
    friend std::ostream& operator<<(std::ostream& os, const Histogram& hist);

    int& operator[](int pos);
    const int& operator[](int pos) const;
    std::vector<int> data();
    int size() const;
    bool empty() const;

    std::vector<int>::iterator begin();
    std::vector<int>::iterator end();
    std::vector<int>::const_iterator begin() const;
    std::vector<int>::const_iterator end() const;

    const std::vector<int>::const_iterator cbegin() const;
    const std::vector<int>::const_iterator cend() const;

    bool WriteToCSV(const std::string& path);
    static Histogram ReadFromCSV(const std::string& path);
  };
  std::ostream& operator<<(std::ostream& os, const Histogram& hist);

}
