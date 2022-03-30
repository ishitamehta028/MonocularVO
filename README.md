# Monocular Visual Odometry

Estimating the location and orientation of a camera by analyzing a sequence of images using OpenCV and Computer Vision

### Requirements
  - Python 3.9
  - Numpy
  - OpenCV
  - Matplotlib

### Dataset
[KITTI odometry data set (grayscale, 22 GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

### Files 
   -  trajectory.py and monocularVO.py
   -  output.txt - contains the terminal output of about 2000 frames
   -  trajectory.png - Ground truth vs Estimated Odometry position plotted.

## Running Program
  1.  First clone repository
  2.  Modify the pose_path and file_path in trajectory.py and monocularVO.py both to your image sequences and ground truth trajectories.
  3.  Make sure to have 2 backslashes in their paths.
  4.  Ensure focal length and principal point information is correct
  5.  Adjust Lucas Kanade Parameters as needed
  6.  Run trajectory.py
 

> MVO has a disadvantage wherein scale factor is difficult to compute and hence the ground truth may be vastly different than the estimated position.
### References
1.    [Monocular Visual Odometry using OpenCV](http://avisingh599.github.io/vision/monocular-vo/) and its related project [report Monocular Visual Odometry | Avi Singh](http://avisingh599.github.io/assets/ugp2-report.pdf)
    
2.    [Github Repository](https://github.com/avisingh599/mono-vo) | Avi Singh


______________________________
![img](https://github.com/ishitamehta028/MonocularVO/blob/main/trajectory.png)
