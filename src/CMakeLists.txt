add_library(html_writer STATIC html_writer.cpp)
add_library(image_browser STATIC image_browser.cpp)

add_executable(01-create_image_browser 01-create_image_browser.cpp)
target_link_libraries(html_writer fmt)
target_link_libraries(image_browser html_writer)
target_link_libraries(01-create_image_browser html_writer image_browser)
