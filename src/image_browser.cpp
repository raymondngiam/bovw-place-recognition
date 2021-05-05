#include "image_browser.hpp"

void image_browser::AddFullRow(const image_browser::ImageRow& row, bool first_row){
  html_writer::OpenRow();
  size_t i = 0;
  for (const auto& item : row){
    image_browser::ScoredImage si = item;
    html_writer::AddImage(std::get<0>(si), std::get<1>(si), (i==0&&first_row));
    i+=1;
  }
  html_writer::CloseRow();
}

void image_browser::CreateImageBrowser(const std::string& title, const std::string& stylesheet,
                        const std::vector<image_browser::ImageRow>& rows){
  html_writer::OpenDocument();
  html_writer::AddTitle(title);
  html_writer::AddCSSStyle(stylesheet);
  html_writer::OpenBody();
  size_t i = 0;
  for (const auto& row : rows){
    AddFullRow(row, i==0);
    i+=1;
  }
  html_writer::CloseBody();
  html_writer::CloseDocument();
}
