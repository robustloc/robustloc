
| Model           | 7Scenes-Chess Mean Error (m/degree) |
|-----------------|-------------------------------------|
| PoseNet         | 0.32/6.6                            |
| AtLoc           | 0.10/4.1                            |
| AnchorPoint     | 0.06/3.9                            |
| GNNMapNet       | 0.08/2.8                            |
| RobustLoc(ours) | 0.08/2.6                            |

Table R1: Performance comparison on the 7Scenes-Chess dataset. Noted our model is without much finetuning, but can still achieve comparable performance.





| Category | Model           | Loop Mean   | Speed  | Inference DB Space |
|----------|-----------------|-------------|--------|--------------------|
| GPS      | GPS             | 7.03/-      | 50 FPS | -                  |
| VO       | Stereo Odometry | 40.20/12.85 | 15 FPS | 10 MB              |
| IR       | NetVLAD         | 3.32/1.90   | 2 FPS  | 10 GB              |
| CPR      | MapNet          | 9.84/3.96   | 75 FPS | -                  |
| CPR      | MapNet+GPS      | 6.78/2.72   | 50 FPS | -                  |
| CPR      | MapNet+VO       | 6.73/2.23   | 15 FPS | 10 MB              |
| CPR      | RobustLoc       | 2.23/1.37   | 50 FPS | -                  |

Table R2: Performance comparison for different kinds of technologies for localization




| Model                     | 3D Model   | Inference Database | Global Accuracy | Local Accuracy | Speed  |
|---------------------------|------------|--------------------|-----------------|----------------|--------|
| 2D-3D key points matching | need       | need               | high            | high           | slow   |
| 2D-2D key points matching | -          | need               | high            | high           | slow   |
| Visual Odometry           | -          | increasing         | low             | high           | fast   |
| SLAM                      | increasing | increasing         | low             | high           | medium |
| Image Retrieval           | -          | need               | medium          | medium         | slow   |
| Camera Pose Regression    | -          | -                  | medium          | medium         | fast   |

Table R3: Advantages and disadvantages for different kinds of technologies for localization. 


| Model            | Loop (cross.) | Loop (cross.) | Loop (within.) | Loop (within.) | Full       | Full       |
|:----------------:|:-------------:|:-------------:|:--------------:|:--------------:|:----------:|:----------:|
|                  | Mean          | Median        | Mean           | Median         | Mean       | Median     |
| GNNMapNet + post | 7.96/2.56     | -             | -              | -              | 17.35/3.47 | -          |
| ADPoseNet        | -             | -             | -              | 6.40/3.09      | -          | 33.82/6.77 |
| ADMapNet         | -             | -             | -              | 6.45/2.98      | -          | 19.18/4.60 |
| MapNet+          | 8.17/2.62     | -             | -              | -              | 30.3/7.8   |            |
| MapNet + post    | 6.73/2.23     | -             | -              | -              | 29.5/7.8   |            |
| GeoPoseNet       | 27.05/18.54   | 6.34/2.06     | -              | -              | 125.6/27.1 | 107.6/22.5 |
| MapNet           | 9.30/3.71     | 5.35/1.61     | -              | -              | 41.4/12.5  | 17.94/6.68 |
| LsG              | 9.08/3.43     | -             | -              | -              | 31.65/4.51 |            |
| AtLoc            | 8.74/4.63     | 5.37/2.12     | -              | -              | 29.6/12.4  | 11.1/5.28  |
| AtLoc+           | 7.53/3.61     | 4.06/1.98     | -              | -              | 21.0/6.15  | 6.40/1.50  |
| CoordiNet        | -             | -             | 4.06/1.44      | 2.42/0.88      | 14.96/5.74 | 3.55/1.14  |
| RobustLoc(ours)  | 4.68/2.67     | 3.70/1.50     | 2.49/1.40      | 1.97/0.84      | 9.37/2.47  | 5.93/1.06  |

Table 1: Performance Comparision on the Oxford RobotCar dataset.




