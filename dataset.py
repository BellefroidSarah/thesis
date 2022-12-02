import torch
import os
import random
import utils as U
import torchvision.transforms as transforms
from PIL import Image
from torchvision.ops import masks_to_boxes


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
        self.fold_div = folds + 1

        #self.files = [file for file in os.listdir(self.masks) if file in os.listdir(self.imgs)]
        self.files = [file for file in os.listdir(self.masks) if ((file in os.listdir(self.imgs)) and ("v" in file))]
        self.files = list(U.split(self.files, self.fold_div))

        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
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
    
    
class ZebrafishDataset_KFold_crop(torch.utils.data.Dataset):
    # Use only if the size(img_dir) >= size(mask_dir)
    # Needs only two folders: data and masks
    # Can output the 3 different types of dataset
    # actual fold between 0 and folds - 1
    
    # Images are cropped around the fish and padded
    def __init__(self, img_dir, mask_dir, actual_fold, model, device, dataset="train", folds=5):
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.fish_model = model
        self.SIZE = (384, 512)
        self.DEVICE = device
        self.pretransform = transforms.Compose([transforms.Resize(self.SIZE),
                                                transforms.Pad((0, 64, 0, 64))])
        self.untransform = transforms.Compose([transforms.CenterCrop(self.SIZE),
                                               transforms.Resize((1932, 2576))])
        self.transform = transforms.ToTensor()
        self.fold_div = folds + 1 

        self.files = [file for file in os.listdir(self.masks) if file in os.listdir(self.imgs)]
        self.files = list(U.split(self.files, self.fold_div))

        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
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
        ms = Image.open(os.path.join(self.masks, file))
        original_image = self.transform(img)
        original_image = original_image[:3, :, :]
        image = self.pretransform(original_image)
        mask = self.transform(ms)
        
        #print(image.shape)
        fish_mask = U.predict_img(self.fish_model, image.unsqueeze(dim=0), self.DEVICE, self.untransform)
        fish_mask_image = Image.fromarray(fish_mask)
        fish_mask_tensor = self.transform(fish_mask_image)
        
        obj_ids = torch.unique(fish_mask_tensor)
        obj_ids = obj_ids[1:]
        
        fish_masks = fish_mask_tensor == obj_ids[:, None, None]
        boxes = masks_to_boxes(fish_masks)
        #print(boxes)
        
        h_length = boxes[0, 2]+1 - boxes[0, 0]
        v_length = boxes[0, 3]+1 - boxes[0, 1]
        h1 = int(boxes[0, 0])
        h2 = int(boxes[0, 2])+1
        v1 = int(boxes[0, 1])
        v2 = int(boxes[0, 3])+1
        
        if h_length%2!=0:
            h1 = h1-1
            h_length += 1
        if v_length%2!=0:
            v1 = v1-1
            v_length += 1
            
        v1 = int(v1)
        v2 = int(v2)
        h1 = int(h1)
        h2 = int(h2)
        cropped = original_image[:, v1:v2, h1:h2]
        mask = mask[:, v1:v2, h1:h2]
        
        if h_length>v_length:
            padding = int((h_length-v_length)/2)
            post_tr = transforms.Compose([transforms.Pad((0, padding, 0, padding)),
                                          transforms.Resize((512,512))])
            untr = transforms.Compose([transforms.Resize((h_length, h_length)),
                                       transforms.CenterCrop((v_length, h_length))])
        elif h_length>v_length:
            padding = int((v_length-h_length)/2)
            post_tr = transforms.Compose([transforms.Pad((padding, 0, padding, 0)),
                                          transforms.Resize((512,512))])
            untr = transforms.Compose([transforms.Resize((v_length, v_length)),
                                       transforms.CenterCrop((v_lentgth, h_length))])
        else:
            post_tr = transforms.Compose([transforms.Resize((512,512))])
            untr = transforms.Compose([transforms.Resize((h_length, h_length))])
            
        image = post_tr(cropped)
        mask = post_tr(mask)
                                          
        img.close()
        ms.close()
        return image, mask, file, (h_length, v_length)
    
    
class ZebrafishDataset_KFold_crop_head(torch.utils.data.Dataset):
    # Use only if the size(img_dir) >= size(mask_dir)
    # Needs only two folders: data and masks
    # Can output the 3 different types of dataset
    # actual fold between 0 and folds - 1
    
    # Images are cropped around the head of the fish and padded
    def __init__(self, img_dir, mask_dir, actual_fold, model, device, dataset="train", folds=5):
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.fish_model = model
        self.SIZE = (384, 512)
        self.DEVICE = device
        self.pretransform = transforms.Compose([transforms.Resize(self.SIZE),
                                                transforms.Pad((0, 64, 0, 64))])
        self.untransform = transforms.Compose([transforms.CenterCrop(self.SIZE),
                                               transforms.Resize((1932, 2576))])
        self.transform = transforms.ToTensor()
        self.fold_div = folds + 1 

        self.files = [file for file in os.listdir(self.masks) if file in os.listdir(self.imgs)]
        self.files = list(U.split(self.files, self.fold_div))

        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
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
        ms = Image.open(os.path.join(self.masks, file))
        original_image = self.transform(img)
        original_image = original_image[:3, :, :]
        image = self.pretransform(original_image)
        mask = self.transform(ms)
        
        fish_mask = U.predict_img(self.fish_model, image.unsqueeze(dim=0), self.DEVICE, self.untransform)
        fish_mask_image = Image.fromarray(fish_mask)
        fish_mask_tensor = self.transform(fish_mask_image)
        
        obj_ids = torch.unique(fish_mask_tensor)
        obj_ids = obj_ids[1:]
        
        fish_masks = fish_mask_tensor == obj_ids[:, None, None]
        boxes = masks_to_boxes(fish_masks)
        
        h_length = boxes[0, 2]+1 - boxes[0, 0]
        v_length = boxes[0, 3]+1 - boxes[0, 1]
        h1 = int(boxes[0, 0])
        h2 = int(boxes[0, 2])+1
        v1 = int(boxes[0, 1])
        v2 = int(boxes[0, 3])+1
        
        if h_length%10!=0:
            mod = 10 - (h_length%4)
            h_length += mod
            
        h_length = (3*h_length)/5
        h2 = int(h1 + h_length)
            
        if v_length%2==1:
            v1 = v1-1
            v_length += 1
        
        cropped = original_image[:, v1:v2, h1:h2]
        mask = mask[:, v1:v2, h1:h2]
        
        if h_length>v_length:
            padding = int((h_length-v_length)/2)
            post_tr = transforms.Compose([transforms.Pad((0, padding, 0, padding)),
                                          transforms.Resize((512,512))])
        elif h_length>v_length:
            padding = int((v_length-h_length)/2)
            post_tr = transforms.Compose([transforms.Pad((padding, 0, padding, 0)),
                                          transforms.Resize((512,512))])
        else:
            post_tr = transforms.Compose([transforms.Resize((512,512))])
            
        image = post_tr(cropped)
        mask = post_tr(mask)
                                          
        img.close()
        ms.close()
        return image, mask, file, (h_length, v_length)