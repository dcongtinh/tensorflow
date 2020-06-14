#!/usr/bin/env python
import os
import cv2
import numpy as np
from matplotlib.pyplot import imread
from sklearn.utils import shuffle
from PIL import Image, ImageDraw, ImageFont


def LoadTrainingData(Dir, Img_Shape):
    print("Training Data is Loading ...")
    Images, Labels = [], []

    for (_, Dirs, _) in os.walk(Dir):
        Dirs = sorted(Dirs)
        for SubDir in Dirs:
            SubjectPath = os.path.join(Dir, SubDir)
            for FileName in os.listdir(SubjectPath):
                path = SubjectPath + '/' + FileName
                # Img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                Img = imread(path)
                height, width = Img.shape

                if width != Img_Shape[0] or height != Img_Shape[1]:
                    Img = Img.resize((Img_Shape[0], Img_Shape[1]))

                Images.append(Img)
                Labels.append(int(SubDir[1:])-1)

    Images, Labels = shuffle(Images, Labels, random_state=2020)

    Images = np.asarray(Images, dtype='float32').reshape(
        [-1, Img_Shape[0], Img_Shape[1]]) / 255.
    print("Training Data is Loaded.")
    return (Images, np.array(Labels))


def LoadTestingData(Dir, Img_Shape):
    print("Testing Data is Loaing ...")
    (Images, Labels, ID) = ([], [], 0)

    for (_, Dirs, _) in os.walk(Dir):
        Dirs = sorted(Dirs)
        for SubDir in Dirs:
            SubjectPath = os.path.join(Dir, SubDir)
            for FileName in os.listdir(SubjectPath):
                path = SubjectPath + "/" + FileName
                # Img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                Img = imread(path)
                height, width = Img.shape
                if width != Img_Shape[0] or height != Img_Shape[1]:
                    Img = Img.resize((Img_Shape[0], Img_Shape[1]))

                Images.append(Img)
                Labels.append(int(SubDir[1:])-1)
    Images = np.asarray(Images, dtype='float32').reshape(
        [-1, Img_Shape[0], Img_Shape[1]]) / 255.
    print("Testing Data is Loaded.")
    return Images, np.array(Labels)
