import torch
import os
import random
import utils as U
import torchvision.transforms as transforms
from PIL import Image


# Dataset class
class ZebrafishDataset(torch.utils.data.Dataset):
    # Use only if the size(img_dir) <= size(mask_dir)
    # and if you have three different files train, mask, and test
    def __init__(self, img_dir, mask_dir):
        """Initialzes a dataset with all the images in the directory img_dir."""
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.transform = transforms.ToTensor()

        self.dataset = [file for file in os.listdir(self.imgs)]

        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        # When the extensions are different
        m_file = file[:-4] + ".tif"
        img = Image.open(os.path.join(self.imgs, file))
        mask = Image.open(os.path.join(self.masks, m_file))
        return self.transform(img), self.transform(mask), file


class ZebrafishDataset_v2(torch.utils.data.Dataset):
    # Use only if the size(img_dir) >= size(mask_dir)
    def __init__(self, img_dir, mask_dir):
        """Initialzes a dataset with all the images in the directory img_dir."""
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.transform = transforms.ToTensor()

        self.dataset = [file for file in os.listdir(self.masks)]

        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        # When the extensions are different
        m_file = file[:-4] + ".tif"
        img = Image.open(os.path.join(self.imgs, file))
        mask = Image.open(os.path.join(self.masks, m_file))
        return self.transform(img), self.transform(mask), file


class ZebrafishDataset_KFold(torch.utils.data.Dataset):
    # Use only if the size(img_dir) <= size(mask_dir)
    # Needs only two folders: data and masks
    # Can output the 3 different types of dataset
    # actual fold between 0 and folds - 1
    def __init__(self, img_dir, mask_dir, actual_fold, dataset="train", folds=5):
        """Initialzes a dataset with all the images in the directory img_dir."""
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.transform = transforms.ToTensor()
        self.fold_div = folds + 1

        #self.files = [file for file in os.listdir(self.imgs) if "lat" in file]
        self.files = [file for file in os.listdir(self.imgs)]
        self.files = list(U.split(self.files, self.fold_div))

        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
        elif dataset == "pre-test":
            self.dataset = self.files[-2]
        elif dataset == "test":
            self.dataset = self.files[-1]

        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        # When the extensions are different
        m_file = file[:-4] + ".tif"
        img = Image.open(os.path.join(self.imgs, file))
        mask = Image.open(os.path.join(self.masks, m_file))
        return self.transform(img), self.transform(mask), file


class ZebrafishDataset_KFold_v2(torch.utils.data.Dataset):
    # Use only if the size(img_dir) >= size(mask_dir)
    # Needs only two folders: data and masks
    # Can output the 3 different types of dataset
    # actual fold between 0 and folds - 1
    def __init__(self, img_dir, mask_dir, actual_fold, dataset="train", folds=5):
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.transform = transforms.ToTensor()
        self.fold_div = folds + 2

        self.files = [file for file in os.listdir(self.masks) if file in os.listdir(self.imgs)]
        self.files = [file for file in os.listdir(self.masks) if ((file in os.listdir(self.imgs)) and ("v" in file))]
        self.files = list(U.split(self.files, self.fold_div))
        #print(self.files)

        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
        elif dataset == "pre-test":
            self.dataset = self.files[-2]
            print("Pre-testing set length: {}".format(len(self.dataset)))
        elif dataset == "test":
            self.dataset = self.files[-1]
            print("Testing set length: {}".format(len(self.dataset)))
        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        img = Image.open(os.path.join(self.imgs, file))
        mask = Image.open(os.path.join(self.masks, file))
        im = self.transform(img)
        img.close()
        ms = self.transform(mask)
        mask.close()
        return im[:3, :, :], ms, file
    