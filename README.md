# rebuttal

| Model           | 7Scenes-Chess Mean Error (m/degree) |
|-----------------|-------------------------------------|
| PoseNet         | 0.32/6.6                            |
| AtLoc           | 0.10/4.1                            |
| AnchorPoint     | 0.06/3.9                            |
| GNNMapNet       | 0.08/2.8                            |
| RobustLoc(ours) | 0.08/2.6                            |



| Category | Model           | Loop Mean   | Speed  | Inference DB Space |
|----------|-----------------|-------------|--------|--------------------|
| GPS      | GPS             | 7.03/-      | 50 FPS | -                  |
| VO       | Stereo Odometry | 40.20/12.85 | 15 FPS | 10 MB              |
| IR       | NetVLAD         | 3.32/1.90   | 2 FPS  | 10 GB              |
| CPR      | MapNet          | 9.84/3.96   | 75 FPS | -                  |
| CPR      | MapNet+GPS      | 6.78/2.72   | 50 FPS | -                  |
| CPR      | MapNet+VO       | 6.73/2.23   | 15 FPS | 10 MB              |
| CPR      | RobustLoc       | 2.23/1.37   | 50 FPS | -                  |


| Model                     | 3D Model   | Inference Database | Global Accuracy | Local Accuracy | Speed  |
|---------------------------|------------|--------------------|-----------------|----------------|--------|
| 2D-3D key points matching | need       | need               | high            | high           | slow   |
| 2D-2D key points matching | -          | need               | high            | high           | slow   |
| Visual Odometry           | -          | increasing         | low             | high           | fast   |
| SLAM                      | increasing | increasing         | low             | high           | medium |
| Image Retrieval           | -          | need               | medium          | medium         | slow   |
| Camera Pose Regression    | -          | -                  | medium          | medium         | fast   |


