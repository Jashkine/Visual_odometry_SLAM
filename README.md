# Visual SLAM Demo - Bag of Words & Trajectory Visualization

This notebook demonstrates a Visual SLAM system with loop closure detection using Bag of Words (BoW) and trajectory visualization comparing ground truth vs estimated paths.

## Overview

The demo consists of two main parts:

1. **Bag of Words Image Retrieval**: Demonstrates loop closure detection by finding similar images in a sequence
2. **SLAM Trajectory Visualization**: Compares ground truth poses with estimated trajectories and visualizes errors

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

### 1. Create Virtual Environment

```bash
cd /home/lg/Desktop/Python_files/1_visualslam
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install numpy opencv-python matplotlib seaborn jupyter ipykernel
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.2+ | Numerical computations |
| opencv-python | 4.12+ | Computer vision (ORB feature detection) |
| matplotlib | 3.10+ | Plot visualization |
| seaborn | 0.13+ | Heatmap visualization |
| jupyter | Latest | Interactive notebook environment |
| ipykernel | Latest | Jupyter kernel for Python |

## Running the Demo

### Option 1: Jupyter Notebook (Interactive)

```bash
cd /home/lg/Desktop/Python_files/1_visualslam/VisualSLAM
jupyter notebook demo.ipynb
```

Then open your browser to `http://localhost:8888` and navigate to `demo.ipynb`.

### Option 2: Command Line Execution

```bash
cd /home/lg/Desktop/Python_files/1_visualslam/VisualSLAM
jupyter nbconvert --to notebook --execute demo.ipynb
```

## Notebook Structure

### Part 1: Bag of Words (Cells 1-13)

**Purpose**: Demonstrates image retrieval and loop closure detection

**Key Cells**:
1. **Import libraries** - numpy, cv2, matplotlib, dbow
2. **Load images** - Reads `.png` images from `./images/` directory
3. **Create vocabulary** - Builds BoW vocabulary from ORB descriptors (10 clusters, depth=2)
4. **ORB detector** - Initializes OpenCV ORB feature detector
5. **Compute BoW vectors** - Converts image descriptors to bag-of-words representation
6. **Query database** - Finds best matching images with similarity scores
7. **Visualize keypoints** - Shows detected ORB features (green circles)
8. **Similarity matrix** - Heatmap showing image-to-image similarity scores
9. **Save/load** - Demonstrates vocabulary and database persistence

**Output**:
- Query image pairs with similarity scores
- ORB keypoint visualization (500+ features per image)
- YlOrRd color heatmap of all pairwise similarities

### Part 2: SLAM Trajectory (Cells 14-18)

**Purpose**: Visualizes localization accuracy by comparing GT vs estimated paths

**Key Cells**:
1. **Load KITTI poses** - Reads 51 ground truth poses from `poses.txt`
2. **Simulate trajectory** - Creates estimated path with 0.5m Gaussian noise
3. **2D trajectory plot** - Top-down view with error lines between GT (blue) and estimated (green)
4. **Error metrics** - 4-subplot analysis (absolute, cumulative, X-axis, Z-axis errors)
5. **3D visualization** - Interactive 3D trajectory comparison

**Output**:
- 18×10 inch 2D trajectory plot with:
  - Blue solid line: Ground truth path
  - Green dashed line: Estimated path
  - Red dotted lines: Error vectors at each frame
  - Circle markers: Start positions
  - Square markers: End positions
- Error statistics: mean, std, min, max, cumulative
- 3D trajectory visualization with start/end markers

## Dataset Structure

```
KITTI_sequence_1/
├── calib.txt          # Camera calibration parameters
├── poses.txt          # Ground truth poses (51 frames)
├── image_l/           # Left stereo images
└── image_r/           # Right stereo images

images/               # Query images for BoW demo (must be .png format)
```

## Key Dependencies

### dbow.py (Pure Python Fallback)

A Python shim that provides Bag of Words functionality:

```python
import dbow

# Create vocabulary
vocabulary = dbow.Vocabulary(images, n_clusters=10, depth=2)

# Create database and query
db = dbow.Database(vocabulary)
db.add(descriptors)
scores = db.query(query_descriptors)

# Save/load
vocabulary.save('vocabulary.pickle')
db.save('database.pickle')
```

### ORB Feature Detection

```python
import cv2
orb = cv2.ORB_create()
kps, descs = orb.detectAndCompute(image, None)
descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
```

## Troubleshooting

### Issue: No images found for BoW demo

**Solution**: Add `.png` images to the `./images/` directory before running cells 1-13

### Issue: KITTI data not found

**Solution**: Ensure `KITTI_sequence_1/` directory exists with `poses.txt` file in the same directory as `demo.ipynb`

### Issue: Module 'dbow' not found

**Solution**: Ensure you're running from `/home/lg/Desktop/Python_files/1_visualslam/VisualSLAM/` directory where `dbow.py` is located

### Issue: Jupyter kernel not responding

**Solution**: 
```bash
source .venv/bin/activate
pip install --upgrade jupyter ipykernel
jupyter notebook demo.ipynb
```

## Output Description

### Bag of Words Results

- **Query Results**: Shows 5 sample queries with best matching images and similarity scores (0.09-0.12 range typical)
- **ORB Keypoints**: Green circles overlaid on images, ~500 features per image
- **Similarity Matrix**: Yellow (low) to Red (high) heatmap showing all pairwise image matches

### Trajectory Results

- **Error Statistics**:
  - Mean Error: ~0.58m
  - Max Error: ~1.37m
  - Cumulative Error: ~29.67m

- **2D Plot**: Shows trajectory in XZ plane with error deviation lines
- **Error Subplots**: 
  - Absolute error per frame
  - Cumulative error growth
  - X-axis component error
  - Z-axis component error

## Performance Notes

- **BoW Computation**: Pure Python k-means (~2-5 seconds for full sequence)
- **Visualization**: Matplotlib rendering (~1-2 seconds per plot)
- **Memory Usage**: ~200-300 MB for 51-frame KITTI sequence

## Advanced Usage

### Modify BoW Parameters

Edit cell 3 (Create Vocabulary):
```python
n_clusters = 10    # Number of visual words (increase for finer vocabulary)
depth = 2          # BoW tree depth (increase for hierarchical structure)
vocabulary = dbow.Vocabulary(images, n_clusters, depth)
```

### Change Noise Level

Edit cell 15 (Simulate trajectory):
```python
estimated_path += np.random.randn(*gt_path.shape) * 0.5  # Adjust noise sigma
```

### Adjust Plot Size

Edit cell 16 (2D trajectory):
```python
fig, ax = plt.subplots(figsize=(18, 10))  # Change width, height in inches
```

## References

- **ORB Features**: Rublee et al., "ORB: An Efficient Alternative to SIFT or SURF" (2011)
- **Bag of Words**: Sivic & Zisserman, "Video Google" (2003)
- **KITTI Dataset**: Geiger et al., "Vision meets Robotics: The KITTI Dataset"

## License

See LICENSE file in project root.

## Author

Visual SLAM Demo - Based on KITTI dataset and OpenCV framework

---

**Last Updated**: January 2026  
**Python Version**: 3.8+  
**Status**: Production Ready
