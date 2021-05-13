## Place recognition using Bag of Visual Words in C++

---

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
<img src='./images/scene_sift_000.png'>
<img src='./images/scene_sift_300.png'>
<img src='./images/scene_sift_700.png'>

### Generating BOVW vocabulary
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

### Visualizing BOVW vocabulary
```
$ cd <repo_root>/bin
$ ./03_plot_vocab 
Loaded vocabulary: Rows[1000] Cols[128]
```
<img src='./images/vocabulary.png'>

### Generating BOVW histogram
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
**Selected image for visualization of BOVW histogram:**
<img src='./images/imageCompressedCam0_0003730.png'>

<img src='./images/BOVW_histogram.png'>

