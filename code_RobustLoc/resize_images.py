import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms.functional as TVF

images_folder = '/data/abc/loc/RobotCar/loop/2014-06-23-15-36-04/stereo/semantic_960_psp_r50_d8_769x769_40k/'
images_list = os.listdir(images_folder)
output_folder = '/data/abc/loc/RobotCar/loop/2014-06-23-15-36-04/stereo/semantic_256_psp_r50_d8_769x769_40k/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for image_name in tqdm(images_list):

    img = Image.open(images_folder + image_name )

    img = transforms.Resize(256, TVF.InterpolationMode.NEAREST)(img)

    img.save(output_folder + image_name)
