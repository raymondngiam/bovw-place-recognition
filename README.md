## Place recognition using Bag of Visual Words in C++

---

### Overview

- Final project for University of Bonn's `Modern C++ for Computer Vision and Image Processing (2020)` course.
- Task: Realize a visual place recognition system using Bag of Visual Words (BoVW).

### Dataset

- Download the Freiburg dataset from <a href='https://uni-bonn.sciebo.de/s/c2d0a1ebbe575fdba2a35a8033f1e2ab'>here</a>.

- Extract it to the path `<repo_root>/data/freiburg-full/images/`.

### Dependencies

- CMake > 3.18
- OpenCV == 4.5.1

### Pull third party submodules

```
$ cd <repo_root>
$ git submodule update --init --recursive
```

### Build

```
$ cd <repo_root>
$ mkdir build && cd build
$ cmake ..
$ make
```

### Run tests

```
$ cd <repo_root>/bin
$ ./bovw_place_recognition_test 
```

### Visualizing SIFT keypoints in dataset images
```
$ cd <repo_root>/bin
$ ./01-visualizing_sift_keypoints 

Testing SIFT
Saved image to ../images/scene_sift_000.png
Saved image to ../images/scene_sift_300.png
Saved image to ../images/scene_sift_700.png

```
<img src='./images/scene_sift_000.png' width=480/>
<img src='./images/scene_sift_300.png' width=480/>
<img src='./images/scene_sift_700.png' width=480/>

### Generating BoVW vocabulary

- Building the vocabulary dictionary with parameter `KMEANS_MAX_ITER` = 30, `KMEANS_DICT_SIZE` = 1000.
- These parameters can be modified in <a href='./src/02-compute_vocab.cpp'>02-compute_vocab.cpp</a>.

    ```
    $ cd <repo_root>/bin
    $ ./02_compute_vocab 
    Converting full dataset...
    Image path: "../data/freiburg-full/images"
    Processed count: 692
    Loaded descriptor: 692
    Dict vocabulary: 
    Rows: 1000
    Cols: 128
    ```

### Visualizing BoVW vocabulary
```
$ cd <repo_root>/bin
$ ./03_plot_vocab 
Loaded vocabulary: Rows[1000] Cols[128]
```
<img src='./images/vocabulary.png'>

### Generating BoVW histogram
```
$ cd <repo_root>/bin
$ ./04_compute_histogram 
Loading vocabulary
Loaded vocabulary: Rows[1000] Cols[128]

Extracting raw visual word histogram from dataset...
Raw histogram size size: rows[692] cols[1000]

Performing tf-idf reweighting...
Tfidf histogram: rows[692] cols[1000]
Tfidf multiplier: rows[1] cols[1000]

Saving serialized file...
hist_tfidf saved to ../data/tfidf_hist_full.bin
multiplier_tfidf saved to ../data/tfidf_multiplier_full.bin

Visualizing histogram...
First image path: ../data/freiburg-full/bin/imageCompressedCam0_0003730.bin
```
**Selected image for visualization of BoVW histogram:**
<img src='./images/imageCompressedCam0_0003730.png' width=480/>

<img src='./images/BOVW_histogram.png'>

### Running the complete solution

- Run the executable with the desired image path as first argument, and pipe the output to an html file in `<repo_root>/web_app/`.

    ```shell
    $ cd <repo_root>/bin
    $ ./05-complete_search_solution ../data/freiburg/images/imageCompressedCam0_0000000.png > ../web_app/output.html
    ```

- Open the html with a web browser.
- The output is an array of 9 images sorted by similarity score in descending order.
- The image with the highest matching score is at index 1 with a green bounding box.
- Each of the displayed images are labeled with their respective filename and matching score relative to the query image.

### Results

1. **Input image:** `imageCompressedCam0_0000000.png`

    <img src='./data/freiburg/images/imageCompressedCam0_0000000.png' width=240/>

    **Output:**

    <img src='./images/Result_0000.png'>

2. **Input image:** `imageCompressedCam0_0000300.png`

    <img src='./data/freiburg/images/imageCompressedCam0_0000300.png' width=240/>

    **Output:**

    <img src='./images/Result_0300.png'>

3. **Input image:** `imageCompressedCam0_0000700.png`

    <img src='./data/freiburg/images/imageCompressedCam0_0000700.png' width=240/>

    **Output:**

    <img src='./images/Result_0700.png'>

4. **Input image:** `imageCompressedCam0_0003730.png`

    <img src='./images/imageCompressedCam0_0003730.png' width=240/>

    **Output:**

    <img src='./images/Result_3730.png'>