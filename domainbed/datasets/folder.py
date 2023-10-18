from PIL import Image, ImageFile

from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder,ImageFolder
import os.path as osp
import random
import torch
import domainbed.datasets.transforms as DBT

class ImageFolderwithDomain(ImageFolder):
    def __init__(self,root,env_i,transform=None):
        super(ImageFolderwithDomain,self).__init__(root,transform=transform)
        self.env_i = env_i
        self.data = self.samples
        self.imgloader = self.loader

        self.length = len(self.samples)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.env_i
class Mydataset(Dataset):
    def __init__(self,datasets):
        assert isinstance(datasets,list)
        self.samples=[]
        self.envs = []
        for i,dataset in enumerate(datasets):
            envs = [i]*len(dataset.underlying_dataset.data)
            self.envs.extend(envs)
            self.samples.extend(dataset.underlying_dataset.data)
        self.length = len(self.samples)
        self.transform = DBT.aug
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = read_image(path)
        env_i = self.envs[index]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, env_i
    def __len__(self):
        return self.length


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

