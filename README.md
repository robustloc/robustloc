# RobustLoc

| Model           | 7Scenes-Chess Mean Error (m/degree) |
|:---------------:|:-----------------------------------:|
| PoseNet         | 0.32/6.6                            |
| AtLoc           | 0.10/4.1                            |
| AnchorPoint     | 0.06/3.9                            |
| GNNMapNet       | 0.08/2.8                            |
| RobustLoc(ours) | 0.08/2.6                            |


Table 1: Performance comparison on the 7Scenes-Chess dataset. Noted our model is without much finetuning, but can still achieve comparable performance.





| Category | Model           | Loop Mean Error | Speed  | Inference DB Space |
|:--------:|:---------------:|:-----------:|:------:|:------------------:|
| GPS      | GPS             | 7.03/-      | 50 FPS | -                  |
| VO       | Stereo Odometry | 40.20/12.85 | 15 FPS | 10 MB              |
| IR       | NetVLAD         | 3.32/1.90   | 2 FPS  | 10 GB              |
| CPR      | MapNet          | 9.84/3.96   | 75 FPS | -                  |
| CPR      | MapNet+GPS      | 6.78/2.72   | 50 FPS | -                  |
| CPR      | MapNet+VO       | 6.73/2.23   | 15 FPS | 10 MB              |
| CPR      | RobustLoc       | 2.23/1.37   | 50 FPS | -                  |


Table 2: Performance comparison for different kinds of technologies for localization




| Model                     | 3D Model   | Inference Database | Global Accuracy | Local Accuracy | Speed  |
|:-------------------------:|:----------:|:------------------:|:---------------:|:--------------:|:------:|
| 2D-3D key points matching | need       | need               | high            | high           | slow   |
| 2D-2D key points matching | -          | need               | high            | high           | slow   |
| Visual Odometry           | -          | increasing         | low             | high           | fast   |
| SLAM                      | increasing | increasing         | low             | high           | medium |
| Image Retrieval           | -          | need               | medium          | medium         | slow   |
| Camera Pose Regression    | -          | -                  | medium          | medium         | fast   |


Table 3: Advantages and disadvantages for different kinds of technologies for localization. 


| Model            | Loop (cross.) | Loop (cross.) | Loop (within.) | Loop (within.) | Full       | Full       |
|:----------------:|:-------------:|:-------------:|:--------------:|:--------------:|:----------:|:----------:|
|                  | Mean Error         | Median Error        | Mean Error           | Median Error         | Mean Error       | Median Error     |
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

Table 4: Performance comparision on the Oxford RobotCar dataset.



| Model           | Business.  | Business. | Neighborhood | Neighborhood | Old Town   | Old Town   |
|:---------------:|:----------:|:---------:|:------------:|:------------:|:----------:|:----------:|
|                 | Mean Error       | Median Error    | Mean Error         | Median Error       | Mean Error       | Median Error     |
| GeoPoseNet      | 11.04/5.78 | 5.93/2.03 | 2.87/1.30    | 1.92/0.88    | 64.81/6.67 | 15.03/1.57 |
| MapNet          | 10.35/3.78 | 5.66/1.83 | 2.81/1.05    | 1.89/0.92    | 46.56/7.14 | 16.52/2.12 |
| GNNMapNet       | 7.69/4.34  | 5.52/2.16 | 3.02/2.92    | 2.14/1.45    | 41.54/7.30 | 19.23/3.26 |
| AtLoc           | 11.53/4.84 | 5.81/1.50 | 2.80/1.16    | 1.83/0.93    | 84.17/7.81 | 17.10/1.73 |
| AtLoc+          | 13.70/6.41 | 5.58/1.94 | 2.33/1.39    | 1.61/0.88    | 68.40/5.51 | 14.52/1.69 |
| IRPNet          | 10.95/5.38 | 5.91/1.82 | 3.17/2.85    | 1.98/0.90    | 55.86/6.97 | 17.33/3.11 |
| CoordiNet       | 11.52/3.44 | 6.44/1.38 | 1.75/0.86    | 0.37/0.69    | 43.68/3.58 | 11.83/1.36 |
| RobustLoc(ours) | 4.28/2.04  | 2.55/1.50 | 1.36/0.83    | 1.00/0.65    | 21.65/2.41 | 5.52/1.05  |

Table 5: Performance comparision on the 4Seasons dataset.


| Model           | Medium        | Medium      | Hard          | Hard            | Hard + Noisy Training | Hard + Noisy Training |
|:---------------:|:-------------:|:-----------:|:-------------:|:---------------:|:---------------------:|:---------------------:|
|                 | Mean Error          | Median Error      | Mean Error          | Median Error          | Mean Error                  | Median Error                |
| GeoPoseNet      | 20.47 / 8.76  | 8.70 / 2.30 | 41.71 / 17.63 | 14.02 / 3.13    | 24.03 / 11.14         | 7.14 / 1.70           |
| MapNet          | 17.93 / 7.01  | 6.89 / 2.00 | 49.36 / 20.01 | 18.37 / 2.58    |  21.22 / 8.38         | 6.38 / 1.97           |
| GNNMapNet       | 16.17 / 7.24  | 8.02 / 2.35 | 73.97 / 35.57 |  61.47 / 19.73  | 14.55 / 7.62          | 6.69 / 1.57           |
| AtLoc           | 19.92 / 7.25  | 7.26 / 1.74 | 52.56 / 23.46 | 15.01 / 3.17    | 23.48 / 11.43         | 7.42 / 2.38           |
| AtLoc+          | 17.68 / 7.48  | 6.19 / 1.80 | 37.92 / 18.65 | 12.17 / 2.93    | 22.61 / 11.23         | 6.21 / 1.83           |
| IRPNet          | 16.35 / 7.56  | 8.71 / 2.28 | 45.72 / 21.84 | 17.99 / 3.50    | 24.73 / 11.20         |  6.73 / 1.82          |
| CoordiNet       | 17.67 / 6.66  | 7.63 / 1.79 | 44.11 / 16.42 | 17.21 / 2.70    | 24.06 / 12.27         | 6.25 / 1.61           |
| RobustLoc(ours) | 8.12 / 3.83   | 5.34 / 1.53 | 27.75 / 9.70  | 11.59 / 2.64    | 10.06 / 4.95          | 5.18 / 1.43           |

Table 6: Performance comparision on the Perturbed RobotCar dataset.




| Method                    | Mean Error (meter/degree) on Loop (c.) |
|:-------------------------:|:--------------------------------------:|
| base model                | 8.38 / 4.29                            |
|  + feature map graph      | 7.01 / 3.86                            |
|  + vector embedding graph | 6.24 / 3.21                            |
|  + diffusion              | 5.53 / 2.95                            |
|  +  branched decoder      | 5.14 / 2.79                            |
|  +  multi-level decoding  | 4.68 / 2.67                            |
| diffusion at stage 3      | 5.27 / 2.90                            |
| diffusion at stage 3,4    | 4.86 / 3.18                            |
| diffusion at stage 4      | 4.68 / 2.67                            |
| multi-layer concatenation | 5.80 / 3.26                            |
| more augmentation         | 4.68 / 2.67                            |
| less augmentation         | 5.32 / 3.17                            |

Table 7: Main ablation study.



| Method           | Mean Error (meter/degree) on Full      |
|:----------------:|:--------------------------------------:|
| grid graph       | 15.67 / 2.95                           |
| self-cross graph | 15.31 / 3.28                           |
| complete graph   | 9.37 / 2.47                            |
|                  | Mean Error (degree) on Business Campus |
| quaternion       | 2.23                                   |
| Lie group        | 2.2                                    |
| rotation matrix  | 2.25                                   |
| log (quaternion) | 2.04                                   |

Table 8: Graph design comparison.


| #frames         | 3    | 5    | 7    | 9    | 11   |
|:---------------:|:----:|:----:|:----:|:----:|:----:|
| Speed (iters/s) | 56   | 55   | 53   | 52   | 50   |
| Mean Error (m)  | 5.28 | 5.09 | 4.96 | 4.68 | 4.72 |

Table 9: Performance using different number of frames.
