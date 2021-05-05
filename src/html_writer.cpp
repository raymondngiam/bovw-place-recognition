#include "html_writer.hpp"
#include <fmt/core.h>
#include <regex>

void html_writer::OpenDocument(){
  fmt::print("<!DOCTYPE html>\n<html>\n");
}

void html_writer::CloseDocument(){
  fmt::print("</html>");
}

void html_writer::AddCSSStyle(const std::string& stylesheet){
  fmt::print("\t<head>\n\t\t<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\"/>\n\t</head>\n",stylesheet);
}

void html_writer::AddTitle(const std::string& title){
  fmt::print("\t<title>{}</title>\n",title);
}

void html_writer::OpenBody(){
  fmt::print("\t<body>\n");
}

void html_writer::CloseBody(){
  fmt::print("\t</body>\n");
}

void html_writer::OpenRow(){
  fmt::print("\t\t<div class=\"row\">\n");
}

void html_writer::CloseRow(){
  fmt::print("\t\t</div>\n");
}

void html_writer::AddImage(const std::string& img_path, float score, bool highlight){
  if (highlight){
    fmt::print("\t\t\t<div class=\"column\" style=\"border: 5px solid green;\">\n");
  }
  else{
    fmt::print("\t\t\t<div class=\"column\">\n");
  }  
  std::smatch match;
  std::regex re("\\w+/(\\w+.png)");
  std::string filename;
  if (std::regex_search(img_path, match, re) && match.size() > 1) {
    filename = match.str(1);
  }
  else{
    filename = img_path;
  }
  fmt::print("\t\t\t\t<h2>{}</h2>\n",filename);
  fmt::print("\t\t\t\t<img src=\"{}\" />\n",img_path);
  fmt::print("\t\t\t\t<p>score = {:.2f}</p>\n",score);     
  fmt::print("\t\t\t</div>\n");      
}

