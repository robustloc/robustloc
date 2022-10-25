This is the source code of RobustLoc, which supports the training/testing on the Oxford RobotCar dataset.  Our data is constructed mainly on AtLoc https://github.com/BingCS/AtLoc You can refer to the repo for more detailed instructions.

1. Download the relevant data from https://robotcar-dataset.robots.ox.ac.uk/datasets/

2. Change the data directory --data_dir in tools/options.py. E.g. I save the dataset in /data/abc/loc/RobotCar/, so my --data_dir is set as /data/abc/loc/.

3. To process the Oxford RobotCar dataset, you may need the official toolkit at: https://github.com/ori-mrg/robotcar-dataset-sdk 

4. For other settings, you can just maintain as default.

5. main requirements:

   ```
   colour_demosaicing==0.2.2
   matplotlib==3.3.4
   numpy==1.19.2
   Pillow==9.2.0
   scipy==1.5.4
   torch==1.10.0
   torchdiffeq==0.2.3
   torchvision==0.11.1
   tqdm==4.64.0
   transforms3d==0.3.1
   ```

6. The train file is train.py, while the test file is eval.py. The network's main components are in network/.

7. The code is based on some relevant repositories:
   AtLoc: https://github.com/BingCS/AtLoc

   Vid-ODE: https://github.com/psh01087/Vid-ODE

   NeuralODE: https://github.com/rtqichen/torchdiffeq

8. Hope you enjoy it :)
