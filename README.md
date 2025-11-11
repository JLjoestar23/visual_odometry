# Visual Odometry
By: Jojo Liu and Andrew Kurtz 

## Introduction
In this project, we implement monocular visual odometry in Python using OpenCV, applying it to the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). We explain the math behind the core concept, the essential matrix, and wrote a custom implementation of it.

Taking a step back, visual odometry is the process of determining a robot's position and orientation velocity using one or more cameras. Monocular visual odometry is the case when you use one camera. This is frequently used as a core part of visual SLAM, in drone localization, and/or in combination with other forms of odometry using a Kalman filter.

Here is a demo of our algorithm running. The main window shows the camera feed and the smaller window shows the current estimated position. The final few frames show the estimated trajectory compared to the ground truth. 
<p align="center">
  <img src="./assets/demo.gif" width="600" alt="Demo">
</p>

### Running Code

To run the code locally, clone the repo, then install the dependencies with:

```
pip install -r requirements.txt
```

Next, download the Grayscale Odometry Dataset and Ground Truth Poses from the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). You will need to create an account. After unzipping both, we move each sequence’s ground truth file into its corresponding folder within the odometry dataset and rename it to `truth.txt. The structure for a single sequence should look like this:

```
└── sequences
    ├── 00
    │   ├── calib.txt
    │   ├── image_0
    │   ├── image_1
    │   ├── times.txt
    │   └── truth.txt
```

Next, you can update the path `DATA_DIR` at the top of `main.py` to match the path to the sequence you would like to test. Finally, you can run `main.py`.


## Algorithmic Overview

## Calculating Essential Matrix

### Math

### Comparison with OpenCV Implementation

## Conclusion

### Limitations

### Lessons

### Future Work

## Sources
