#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "kmeans.hpp"

namespace ipb{
  class BowDictionary{
  
  private:
    int _maxIter;
    int _dictSize;
    std::vector<cv::Mat> _inputDescriptors;
    cv::Mat _dictionary;
    static BowDictionary* instance;   //IMPORTANT: static member needs to be initialized from .cpp file.
    BowDictionary() = default;
  public:
    // getters
    int max_iterations() const;
    int size() const; // number of centroids / codewords
    std::vector<cv::Mat> descriptors() const;

    // setters
    void set_max_iterations(int value);
    void set_size(int value);
    void set_descriptors(std::vector<cv::Mat> value);
    void set_params(int max_iter, int size, std::vector<cv::Mat> descriptors);
    void set_vocabulary(cv::Mat);
    void build(int max_iter, int size, std::vector<cv::Mat> descriptors);
    
    // result
    cv::Mat vocabulary() const;

    // utilities
    int total_features() const; // number of input features
    bool empty(); // is result dictionary empty?

    // singleton access
    static BowDictionary& GetInstance();
  };
}
