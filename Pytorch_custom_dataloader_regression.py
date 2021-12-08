from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import glob
from tqdm import tqdm
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()   # interactive mode

class ImageListDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, img_dir, transform=None, verify_files=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        img_list_df = pd.read_csv(csv_file)
        img_list_df.iloc[:, 0] = img_list_df.iloc[:, 0] + "_2.jpg"
        # img_list_df = img_list_df[['panoId', 'class_id', 'noise']]

        if verify_files:
            not_exists_files = []
            for idx in range(len(img_list_df)):
                img_name = os.path.join(img_dir,
                                        img_list_df.iloc[idx, 0])
                if not os.path.exists(img_name):
                    not_exists_files.append(idx)
            dropped_rows = img_list_df.iloc[not_exists_files].copy()
            img_list_df = img_list_df.drop(not_exists_files)
            print(f"Dropped {len(not_exists_files)} rows:\n", dropped_rows)
            img_list_df.to_csv(csv_file, index=False)


        self.img_list = img_list_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        basename = self.img_list.iloc[idx, 0]
        basename = basename[:22] + "_2.jpg"
        img_name = os.path.join(self.img_dir,
                                basename)
        # image = io.imread(img_name)
        class_id = self.img_list.loc[idx, 'noise']
        try:
            image = Image.open(img_name)
        except Exception as e:
            print("Error in __getitem__(), using the first row:", e, img_name)
            basename = self.img_list.iloc[0, 0]
            basename = basename[:22] + "_2.jpg"
            img_name = os.path.join(self.img_dir,
                                    basename)
            image = Image.open(img_name)
            class_id = self.img_list.loc[0,  'noise']
            # print("First row: ", img_name, class_id)


        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'class_id': class_id}

        if self.transform:
            image = self.transform(image)

        # return sample
        return image, sample['class_id'], img_name


class Image_list_dataloader():
    def __init__(self, csv_file, img_dir):
        self.csv_file = csv_file
        self.img_dir = img_dir
        image_list_dataset = ImageListDataset(csv_file, img_dir)

        return image_list_dataset

def test_image_list_data_loader():
    csv_file = r'K:\Research\Tagme\tagme\src\assets\class_id.csv'
    img_dir = r'K:\Research\Tagme\tagme\src\assets\Image'
    dataset = ImageListDataset(csv_file, img_dir)
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, class_id = sample['image'], sample['class_id']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'class_id': class_id}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, class_id = sample['image'], sample['class_id']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return {'image': image, 'class_id': class_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, class_id = sample['image'], sample['class_id']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'class_id':class_id}

def verfiy_im():
    imgs = glob.glob(r'K:\Research\Noise_map\panoramas4_jpg_half\*.jpg')
    for img in tqdm(imgs):
        try:
            im = Image.open(img)
            im.verify() #I perform also verify, don't know if he sees other types o defects
            im.close() #reload is necessary in my case


        except Exception as e:
          #manage excetions here
            print(f"Image file {img} is broken. {e}")

if __name__ == "__main__":
    # image_list_dataset = test_image_list_data_loader()
    # print(len(image_list_dataset))
    verfiy_im()