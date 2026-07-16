# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import pickle

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def paired_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # Train transform
    if is_train == True:
        basic_transforms = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        
        random_transforms = [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=(-15, 15)),
            # # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            # transforms.RandomResizedCrop(size=224, scale=(0.85, 1.15)),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        
        return random_transforms, basic_transforms
    else:
        return None, [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]


def age_to_group(age):
    if age < 18:
        return 0
    elif age < 44:
        return 1
    elif age < 59:
        return 2
    elif age < 75:
        return 3
    else:  
        return 4
    
def gender_to_int(gender):
    if gender == "Male":
        return 0
    else:
        return 1



def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
    ])
    else:
        
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    
    return transform



class ODIRDatasetSingle_diffusion(Dataset):
    def __init__(self, dataframe, pkl_file, img_dir, is_train, args):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.pkl_file = pkl_file
        self.is_train = is_train
        self.random_transforms, _ = paired_transform(is_train, args) # for diffusion
        _, self.basic_transforms_encoder_cls = paired_transform_cls(is_train, args)  # for causal
        self.basic_transforms_diffusion = [
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]

        self.dataframe['gender'] = self.dataframe['Patient Sex'].apply(gender_to_int)
        self.dataframe['age_group'] = self.dataframe['Patient Age'].apply(age_to_group)

        self.concept_list = ['Age', 'Gender', 'Diabetes', 'Glaucoma', 'Cataract', 'Hypertension']
        self.concept_counts = [5, 2, 2, 2, 2, 2]
        self.cdag = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [1,1,0,0,0,1,0], [1,0,1,0,0,0,0], [1,1,0,1,0,0,0], [1,0,0,0,0,0,0], [0,0,0,0,0,0,0]]

        desired_order = ['ID', 'Fundus', 'Type', 'age_group', 'gender', 'D', 'G', 'C', 'H']
        self.dataframe = self.dataframe[desired_order]

        with open(self.pkl_file, 'rb') as f:
            self.loaded_features = pickle.load(f)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # values = self.dataframe.iloc[idx][3:].values.astype(np.float32)
        values = self.dataframe.iloc[idx][3:].values.astype(np.int64)
        labels = torch.tensor(values)

        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Fundus'])
        image = Image.open(img_name)

        if self.random_transforms:
            random_transform = transforms.Compose(self.random_transforms)
            image_random = random_transform(image)
        else:
            image_random = image

        basic_transforms_diffusion = transforms.Compose(self.basic_transforms_diffusion)
        image_diff = basic_transforms_diffusion(image_random)

        image_id = int(self.dataframe.iloc[idx]['Fundus'].split('_')[0])
        image_feat = np.squeeze(self.loaded_features[int(image_id)])  # get ViT features
        image_type = self.dataframe.iloc[idx]['Type']
        if image_type=='Left':
            image_feat = image_feat[:1024]
        elif image_type=='Right':
            image_feat = image_feat[1024:]

        return image_diff, image_feat, labels
    


class TestDataset_diffusion(Dataset):
    def __init__(self, dataframe, pkl_file, img_dir, is_train, args):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.pkl_file = pkl_file
        self.random_transforms, _ = paired_transform(is_train, args) # for diffusion
        _, self.basic_transforms_encoder_cls = paired_transform_cls(is_train, args)  # for causal
        self.basic_transforms_diffusion = [
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
        
        self.dataframe['gender'] = self.dataframe['Patient Sex'].apply(gender_to_int)
        self.dataframe['age_group'] = self.dataframe['Patient Age'].apply(age_to_group)

        desired_order = ['ID', 'Left-Fundus', 'Right-Fundus', 'age_group', 'gender', 'D', 'G', 'C', 'H']
        self.dataframe = self.dataframe[desired_order]

        self.concept_list = ['Age', 'Gender', 'Diabetes', 'Glaucoma', 'Cataract', 'Hypertension']
        self.concept_counts = [5, 2, 2, 2, 2, 2]
        self.cdag = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [1,1,0,0,0,1,0], [1,0,1,0,0,0,0], [1,1,0,1,0,0,0], [1,0,0,0,0,0,0], [0,0,0,0,0,0,0]]

        with open(self.pkl_file, 'rb') as f:
            self.loaded_features = pickle.load(f)

    def __len__(self):
        return len(self.dataframe)*2

    def __getitem__(self, idx):
        if idx%2 == 0:  # load left
            new_idx = int(idx/2)
            img_name = os.path.join(self.img_dir, self.dataframe.iloc[new_idx]['Left-Fundus'])
            image_id = int(self.dataframe.iloc[new_idx]['Left-Fundus'].split('_')[0])
            encoder_feat = np.squeeze(self.loaded_features[int(image_id)])  # get ViT features
            encoder_feat = encoder_feat[:1024]
            image_name = str(image_id)+'_left'
        else:
            new_idx = int((idx-1)/2)
            img_name = os.path.join(self.img_dir, self.dataframe.iloc[new_idx]['Right-Fundus'])
            image_id = int(self.dataframe.iloc[new_idx]['Left-Fundus'].split('_')[0])
            encoder_feat = np.squeeze(self.loaded_features[int(image_id)])  # get ViT features
            encoder_feat = encoder_feat[1024:]
            image_name = str(image_id)+'_right'

        # values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        values = self.dataframe.iloc[new_idx][3:].values.astype(np.int64)
        labels = torch.tensor(values)

        image_org = Image.open(img_name)
        basic_transforms_diffusion = transforms.Compose(self.basic_transforms_diffusion)
        image_diff = basic_transforms_diffusion(image_org)

        return image_diff, encoder_feat, image_id, labels, image_name
    

class TestDataset_genCounterfactual(Dataset):
    def __init__(self, dataframe, pkl_file, img_dir, is_train, args):
        # self.dataframe = dataframe
        self.img_dir = img_dir
        self.pkl_file = pkl_file
        self.random_transforms, _ = paired_transform(is_train, args) # for diffusion
        _, self.basic_transforms_encoder_cls = paired_transform_cls(is_train, args)  # for causal
        self.basic_transforms_diffusion = [
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
        
        select_list = dataframe[(dataframe['D']==0) & (dataframe['C']==0) & (dataframe['H']==0) & (dataframe['G']==0)]['Fundus'].values.tolist()   # select healthy, generate sick
        # select_list = dataframe[(dataframe['D']==1)]['Fundus'].values.tolist()   # select D, generate normal image
        # # select_list = dataframe[(dataframe['C']==1)]['Fundus'].values.tolist()     # select C, generate normal image
        # # select_list = dataframe[(dataframe['H']==1)]['Fundus'].values.tolist()     # select H, generate normal image
        # # select_list = dataframe[(dataframe['G']==1)]['Fundus'].values.tolist()     # select G, generate normal image

        self.dataframe = []
        for _, row in dataframe.iterrows():
            if row['Fundus'] in select_list:
                self.dataframe.append(row)
        self.dataframe = pd.DataFrame(self.dataframe)
        
        self.dataframe['gender'] = self.dataframe['Patient Sex'].apply(gender_to_int)
        self.dataframe['age_group'] = self.dataframe['Patient Age'].apply(age_to_group)

        desired_order = ['ID', 'Fundus', 'age_group', 'gender', 'D', 'G', 'C', 'H']
        self.dataframe = self.dataframe[desired_order]

        self.concept_list = ['Age', 'Gender', 'Diabetes', 'Glaucoma', 'Cataract', 'Hypertension']
        self.concept_counts = [5, 2, 2, 2, 2, 2]
        self.cdag = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [1,1,0,0,0,1,0], [1,0,1,0,0,0,0], [1,1,0,1,0,0,0], [1,0,0,0,0,0,0], [0,0,0,0,0,0,0]]

        with open(self.pkl_file, 'rb') as f:
            self.loaded_features = pickle.load(f)


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_name = self.dataframe.iloc[idx]['Fundus']
        img_name = os.path.join(self.img_dir, image_name)
        image_id = int(self.dataframe.iloc[idx]['Fundus'].split('_')[0])
        encoder_feat = np.squeeze(self.loaded_features[int(image_id)])  # get ViT features
        if 'left' in image_name:
            encoder_feat = encoder_feat[:1024]
        else:
            encoder_feat = encoder_feat[1024:] # right

        # values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        values = self.dataframe.iloc[idx][2:].values.astype(np.int64)
        labels = torch.tensor(values)

        image_org = Image.open(img_name)
        basic_transforms_diffusion = transforms.Compose(self.basic_transforms_diffusion)
        image_diff = basic_transforms_diffusion(image_org)

        return image_diff, encoder_feat, image_id, labels, image_name.split('.')[0]
    



def save_batch_images(dataloader, num_images=8, filename="batch_visualization.png"):
    data_iter = iter(dataloader)
    (left_images, right_images), labels = next(data_iter)  
    concatenated_images = torch.cat((left_images, right_images), 0)[:2*num_images] 
    print(left_images.shape, right_images.shape)
    plt.figure(figsize=(15, 7))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(concatenated_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



def build_ViT_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    else:
        
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform





def paired_transform_cls(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # Train transform
    if is_train == True:
        basic_transforms = [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        
        random_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 15)),
            # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        
        return random_transforms, basic_transforms
    else:
        return None, [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    

def build_transform_cls(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.15)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    else:
        
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform


class ODIRDatasetSingle_cls(Dataset):
    def __init__(self, dataframe, pkl_file, is_train, args):
        self.dataframe = dataframe
        self.pkl_file = pkl_file
        self.is_train = is_train
        self.random_transforms, self.basic_transforms = paired_transform_cls(is_train, args)

        # sex -> 0/1
        self.dataframe['gender'] = self.dataframe['Patient Sex'].apply(gender_to_int)
        # age -> 0-18，18-44，45-59，60-75，75+
        self.dataframe['age_group'] = self.dataframe['Patient Age'].apply(age_to_group)

        self.concept_list = ['Age', 'Gender', 'Diabetes', 'Glaucoma', 'Cataract', 'Hypertension']
        self.concept_counts = [5, 2, 2, 2, 2, 2]
        self.cdag = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [1,1,0,0,0,1,0], [1,0,1,0,0,0,0], [1,1,0,1,0,0,0], [1,0,0,0,0,0,0], [0,0,0,0,0,0,0]]

        desired_order = ['ID', 'Fundus', 'Type', 'age_group', 'gender', 'D', 'G', 'C', 'H']
        self.dataframe = self.dataframe[desired_order]

        with open(self.pkl_file, 'rb') as f:
            self.loaded_features = pickle.load(f)


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # values = self.dataframe.iloc[idx][3:].values.astype(np.float32)
        values = self.dataframe.iloc[idx][3:].values.astype(np.int64)
        labels = torch.tensor(values)

        image_id = int(self.dataframe.iloc[idx]['Fundus'].split('_')[0])
        image_feat = np.squeeze(self.loaded_features[int(image_id)])

        image_type = self.dataframe.iloc[idx]['Type']
        if image_type=='Left':
            image_feat = image_feat[:1024]
        elif image_type=='Right':
            image_feat = image_feat[1024:]

        return image_feat, labels
    


class TestDataset_cls(Dataset):
    def __init__(self, dataframe, pkl_file, is_train, args):
        self.dataframe = dataframe
        self.pkl_file = pkl_file
        self.transform = build_transform_cls(is_train, args)

        self.dataframe['gender'] = self.dataframe['Patient Sex'].apply(gender_to_int)
        self.dataframe['age_group'] = self.dataframe['Patient Age'].apply(age_to_group)

        desired_order = ['ID', 'Left-Fundus', 'Right-Fundus', 'age_group', 'gender', 'D', 'G', 'C', 'H']
        self.dataframe = self.dataframe[desired_order]

        self.concept_list = ['Age', 'Gender', 'Diabetes', 'Glaucoma', 'Cataract', 'Hypertension']
        self.concept_counts = [5, 2, 2, 2, 2, 2]
        self.cdag = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [1,1,0,0,0,1,0], [1,0,1,0,0,0,0], [1,1,0,1,0,0,0], [1,0,0,0,0,0,0], [0,0,0,0,0,0,0]]

        with open(self.pkl_file, 'rb') as f:
            self.loaded_features = pickle.load(f)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = int(self.dataframe.iloc[idx]['Left-Fundus'].split('_')[0])
        image_feat = np.squeeze(self.loaded_features[int(image_id)])

        # values = self.dataframe.iloc[idx][5:].values.astype(np.float32)
        values = self.dataframe.iloc[idx][3:].values.astype(np.int64)
        labels = torch.tensor(values)

        return image_feat, image_id, labels
    