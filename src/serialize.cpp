#include "serialize.hpp"
#include <iostream>
#include <fstream>
#include <unordered_map>

using std::ios_base;

/*
 * dictionary of cv::Mat.type() enum to byte size
 */
std::unordered_map<uint8_t,uint8_t> size_dict{
{0,1},{1,1},{8,1},{9,1},{16,1},{17,1},{24,1},{25,1},
{2,2},{3,2},{10,2},{11,2},{18,2},{19,2},{26,2},{27,2},
{4,4},{5,4},{12,4},{13,4},{20,4},{21,4},{28,4},{29,4},
{6,8},{14,8},{22,8},{30,8}
};

void ipb::serialization::Serialize(cv::Mat m, const std::string& filename){
  std::ofstream file(filename, ios_base::out | ios_base::binary);
  if (!file) {std::cerr<<"Failed opening file."<<std::endl;}
  int type = m.type();
  int channels = m.channels();

  file.write(reinterpret_cast<char*>(&m.rows), sizeof(int));
  file.write(reinterpret_cast<char*>(&m.cols), sizeof(int));
  file.write(reinterpret_cast<char*>(&type), sizeof(int));
  file.write(reinterpret_cast<char*>(&channels), sizeof(int));

  int size = size_dict[type];
  auto pointer_start = m.ptr(0);
  file.write(reinterpret_cast<char*>(pointer_start), size*m.rows*m.cols*channels);
}

cv::Mat ipb::serialization::Deserialize(const std::string& filename){

  std::ifstream file(filename, ios_base::in | ios_base::binary);
  if (!file) {std::cerr<<"Failed opening file."<<std::endl;}

  int rows, cols, type, channels;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int));
  file.read(reinterpret_cast<char*>(&type), sizeof(int));
  file.read(reinterpret_cast<char*>(&channels), sizeof(int));

  int size = size_dict[type];

  auto m_out = cv::Mat(rows, cols, type);
  auto pointer_start = m_out.ptr(0);

  file.read(reinterpret_cast<char*>(pointer_start), size*rows*cols*channels);
  return m_out;
}
