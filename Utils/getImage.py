import torch
import cv2 as cv
import os
import numpy as np
import glob
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

class getImage(Dataset):
    def __init__(self, folder="../../zdatasets/Pneu2 4.11.24/hasil_augmentasi/", augmentation="RGBShift", state="TRAINING", pkl=True):
        self.dataset, to_onehot = [], np.eye(2)
        if pkl==True:
            with open('PKL/' + augmentation + state + '_dataset.pkl', 'rb') as file:
                self.dataset = pickle.load(file)
        else:
            for _, i in enumerate(os.listdir(folder + augmentation + "/" + state)):
                print(_, "_", i)
                print("AUGMENTATION = ", augmentation, "STATE = ", state)
                for j in glob.glob(folder + augmentation + "/" + state + "/" + i + "/*.jpg"):
                    image = cv.resize(cv.imread(j), (224, 224))/255
                    self.dataset.append([image, to_onehot[_]])

            # with open('PKL/' + augmentation + state + '_dataset.pkl', 'wb') as file:
            #     pickle.dump(self.dataset, file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        features, label = self.dataset[item]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

if __name__=="__main__":
    data = getImage()