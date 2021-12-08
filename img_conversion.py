from PIL import Image
import os
import sys
import glob

from_img_dir = r'K:\Research\Noise_map\NAIP_clipped'

to_img_dir = r'K:\Research\Noise_map\NAIP_clipped_jpg'
os.makedirs(to_img_dir, exist_ok=True)


imgs = glob.glob(os.path.join(from_img_dir, '*.tif'))
print("Got image count: ", len(imgs))

import shutil

for idx, img in enumerate(imgs):
    print_interval = 1000
    img_pil = Image.open(img)
    base_name = os.path.basename(img)
    base_name = base_name[:-3] + 'jpg'
    old_tfw_name = img[:-3] + 'tfw'

    new_name = os.path.join(to_img_dir, base_name)
    new_tfw_name = os.path.join(to_img_dir, os.path.basename(old_tfw_name))
    new_tfw_name = new_tfw_name[:-3] + 'jgw'

    shutil.copyfile(old_tfw_name, new_tfw_name)

    img_pil.save(new_name)
    if idx % print_interval == 0:
        print(idx, img, new_name)