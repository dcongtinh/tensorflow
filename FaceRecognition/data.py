#!/usr/bin/env python
from sklearn.utils import shuffle
from scipy import misc
import os
import cv2
import numpy as np

# Created by Mohamed Elsayed


(TrainingData, TrainingLables, start) = ([], [], 0)
(TestingData, TestingLables, startT) = ([], [], 0)


def LoadTrainingData(Dir, Img_Shape):
    Images, Labels = [], []

    for (_, Dirs, _) in os.walk(Dir):
        Dirs = sorted(Dirs)
        for SubDir in Dirs:
            SubjectPath = os.path.join(Dir, SubDir)
            for FileName in os.listdir(SubjectPath):
                if '.pgm' in FileName:
                    path = SubjectPath + '/' + FileName
                    Img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    height, width = Img.shape

                    if width != Img_Shape[0] or height != Img_Shape[1]:
                        Img = cv2.resize(Img, (Img_Shape[0], Img_Shape[1]))

                    Images.append(Img)
                    Labels.append(int(SubDir[1:]))

    Images, Labels = shuffle(Images, Labels)

    Images = np.asarray(Images, dtype='float32').reshape(
        [-1, Img_Shape[0], Img_Shape[1], 1]) / 255.

    return (Images, np.array(Labels))


def LoadTestingData(Dir, Img_Shape):
    (Images, Labels, ID) = ([], [], 0)

    for (_, Dirs, _) in os.walk(Dir):
        Dirs = sorted(Dirs)
        for SubDir in Dirs:
            SubjectPath = os.path.join(Dir, SubDir)
            for FileName in os.listdir(SubjectPath):
                if '.pgm' in FileName:
                    path = SubjectPath + "/" + FileName
                    Img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                    height, width = Img.shape
                    if width != Img_Shape[0] or height != Img_Shape[1]:
                        Img = cv2.resize(Img, (Img_Shape[0], Img_Shape[1]))

                    Images.append(Img)
                    Labels.append(int(SubDir[1:]))
    Images = np.asarray(Images, dtype='float32').reshape(
        [-1, Img_Shape[0], Img_Shape[1], 1]) / 255.
    return Images, np.array(Labels)
