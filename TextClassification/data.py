import os
import numpy
import logging as logger
import numpy as np


def loadFolder(mainDir):
    result = []
    _, dirs, _ = next(os.walk(mainDir))

    labelIndex = 0
    for dir in dirs:
        _, subdirs, subfiles = next(os.walk(mainDir+dir))
        for file in subfiles:
            path = mainDir + dir + '/' + file
            logger.debug(path)
            with open(path, 'r', encoding='utf8') as f:
                result.append([f.read(), labelIndex])
        labelIndex += 1
    return np.array(result)


def loadData():
    _, dirs, _ = next(os.walk('./data/train'))

    return {
        'train': loadFolder('./data/train/'),
        'test': loadFolder('./data/test/'),
        'labels': np.array(dirs)
    }


if __name__ == '__main__':
    data = loadData()
    print(data['train'][:, 1])
