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
from torchvision import datasets, models, transforms


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

        pano_list_df = pd.read_csv(csv_file).iloc[:]  # csv with class name
        pano_list_df = pano_list_df[['panoId', 'class_id', 'noise']]  # change the column order
        # pano_list_df = pano_list_df.set_index('panoId')


        # img_list = glob.glob(os.path.join(img_dir, "*.jpg"))
        img_path_csv = r'K:\Research\Noise_map\thumbnails.csv'  # temporally avoid read the disk.
        img_list = pd.read_csv(img_path_csv)['thumbnail_path'].to_list()
        img_list = sorted(img_list)
        img_list_df = pd.DataFrame(img_list, columns=['img_path'])
        img_list_df['panoId'] = img_list_df['img_path'].apply(os.path.basename).str[:22]
        img_list_df = img_list_df.set_index('panoId')




        # Error: img_list_df[['img_path']].apply(os.path.basename)
        # panoId_list = img_list_df['panoId'].unique()


        #

        # if verify_files:
        #     not_exists_files = []
        #     for idx in range(len(img_list_df)):
        #         img_name = os.path.join(img_dir,
        #                                 img_list_df.iloc[idx, 0])
        #         if not os.path.exists(img_name):
        #             not_exists_files.append(idx)
        #     dropped_rows = img_list_df.iloc[not_exists_files].copy()
        #     img_list_df = img_list_df.drop(not_exists_files)
        #     print(f"Dropped {len(not_exists_files)} rows:\n", dropped_rows)
        #     img_list_df.to_csv(csv_file, index=False)


        self.img_list_df = img_list_df
        # self.panoId_list = panoId_list
        self.img_dir = img_dir
        self.transform = transform
        self.pano_list_df = pano_list_df

    def __len__(self):
        return len(self.pano_list_df)

    def __getitem__(self, idx):

        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            panoId = self.pano_list_df.iloc[idx]['panoId']

            # img_dict = {}
            img_orders = ["L", "R", "F", "B"] # order of thumbnails: left, right, front, back
            img_paths = self.img_list_df.loc[panoId]['img_path'].to_list()

            if len(img_paths) != 4:
                idx = 0
                old_panoId = panoId
                panoId = self.panoId_list[idx]
                print(f"Can not find 4 image for panoId {old_panoId}, used the first row ({panoId}) instead.")
                img_paths = self.img_list_df.loc[panoId]['img_path'].to_list()

            img_dict = {p[-5:-4]:p for p in img_paths}

            img_ts_list = []  # ts: tensor
            for direction in img_orders:
                img_path = img_dict[direction]
                img_pil = Image.open(img_path)
                if self.transform:
                    img_ts = self.transform(img_pil)
                else:
                    img_ts = img_pil
                img_ts_list.append(img_ts)

            img_ts_list = torch.cat(img_ts_list, dim=0)

            class_id = self.pano_list_df.loc[idx, 'class_id']


            # sample = {'image': image, 'class_id': class_id}

            # return sample
            return img_ts_list, class_id, img_paths

        except Exception as e:
            print("Error __get_item__():", e, idx, img_paths)
            print("Use first row instead.")
            self.__getitem__(0)

class Image_list_dataloader():
    def __init__(self, csv_file, img_dir):
        self.csv_file = csv_file
        self.img_dir = img_dir
        image_list_dataset = ImageListDataset(csv_file, img_dir)


        return image_list_dataset

def test_image_list_data_loader():
    csv_file = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\noise_map\val.csv'
    img_dir = r'K:\Research\Noise_map\thumnails176k'
    input_size = (512, 512)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5)
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    dataset = ImageListDataset(csv_file, img_dir, transform=data_transforms['val'])
    dataloader = DataLoader(dataset, batch_size=1)
    for idx, (inputs, labels, paths) in enumerate(dataloader):
        print(idx, len(inputs), labels, paths)
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
    image_list_dataset = test_image_list_data_loader()
    # print(len(image_list_dataset))
    # verfiy_im()