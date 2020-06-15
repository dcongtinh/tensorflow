import os
import numpy
import logging as logger
import numpy as np

def loadFolder(mainDir):
    result = []
    _, dirs, __ = next(os.walk(mainDir))

    labelIndex = 0
    for dir in dirs:
        _, subdirs, subfiles = next(os.walk(mainDir+dir))
        for file in subfiles:
            logger.debug(mainDir+dir+'/'+file)
            with open(mainDir+dir+'/'+file, 'r', encoding='utf8') as f:
                dat = f.read().replace('\n','').replace('\ufeff','')
                result.append([dat, labelIndex])
        labelIndex += 1
    return np.array(result)

def loadData():
    _, dirs, __ = next(os.walk('./data/train'))

    return {
        'train': loadFolder('./data/train/'),
        'test': loadFolder('./data/test/'),
        'labels': np.array(dirs)
    }

if __name__=='__main__':
    data = loadData()
    print(data['train'][:3])
