add_library(html_writer STATIC html_writer.cpp)
add_library(image_browser STATIC image_browser.cpp)
add_library(serialize STATIC serialize.cpp)
add_library(convert_dataset STATIC convert_dataset.cpp)
add_library(bow_dictionary STATIC bow_dictionary.cpp)
add_library(kmeans STATIC kmeans.cpp)
add_library(histogram STATIC histogram.cpp)
add_library(tfidf STATIC tfidf.cpp)

add_executable(00-create_image_browser 00-create_image_browser.cpp)
target_link_libraries(html_writer fmt)
target_link_libraries(image_browser html_writer)
target_link_libraries(00-create_image_browser image_browser)

add_executable(01-visualizing_sift_keypoints 01-visualizing_sift_keypoints.cpp)
target_link_libraries(01-visualizing_sift_keypoints fmt ${OpenCV_LIBS}) 

add_executable(02-compute_vocab 02-compute_vocab.cpp)
target_link_libraries(02-compute_vocab serialize convert_dataset bow_dictionary kmeans ${OpenCV_LIBS})

add_executable(03-plot_vocab 03-plot_vocab.cpp)
target_link_libraries(03-plot_vocab Python3::Python Python3::NumPy ${OpenCV_LIBS} serialize)

add_executable(04-compute_histogram 04-compute_histogram.cpp)
target_link_libraries(04-compute_histogram 
  tfidf
  histogram
  bow_dictionary
  kmeans
  convert_dataset
  serialize
  ${OpenCV_LIBS}
  Python3::Python 
  Python3::NumPy )
  
add_executable(05-complete_search_solution 05-complete_search_solution.cpp)
target_link_libraries(05-complete_search_solution
  tfidf
  histogram
  bow_dictionary
  kmeans
  serialize
  image_browser
  html_writer
  fmt
  ${OpenCV_LIBS}
)
