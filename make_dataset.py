import os
import scipy.io
import random
import numpy as np
import shutil

file_list = os.listdir('./color_img')
file_list.remove('datasets')

origin_prefix = './color_img'
test_prefix = './test_color_img'
file_list.sort()


def data_split(fullList, ratio):
    length = len(fullList)
    offset = int(length * ratio)
    random.shuffle(fullList)
    sub1 = fullList[:offset]
    sub2 = fullList[offset:]
    return sub1, sub2


train, temp = data_split(file_list, 0.8)
test, validate = data_split(temp, 0.5)
trainDb, trainQ = data_split(train, 0.9)
testDb, testQ = data_split(test, 0.9)
validateDb, validateQ = data_split(validate, 0.9)

poseFile = open('./vins_result_loop.txt', 'r')
poses = poseFile.readlines()
poseFile.close()
poseDb = []
for pose in poses:
    poseDb.append(np.array(pose.split(',')[1:-1]).astype(float))

# for fn in trainDb:
s3eDb = []
trainQ.sort()
print(trainQ)
'''

for index, i in enumerate(trainDb):
    print(index)
    print(trainDb[index])
    s3eDb.append(poseDb[int(i[:-10])])

s3eQ = []

for i in trainQ:
    s3eQ.append(poseDb[int(i[:-10])])


trainMat = {}
valMat = {}
testMat = {}
temp = np.array((
    ['train'],
    [trainDb],
    [s3eDb],
    [trainQ],
    [s3eQ],
    [len(trainDb)],
    [len(trainQ)],
    [3],
    [25],
    [16]
))

trainMat['dbStruct'] = temp

scipy.io.savemat('./s3e_train.mat', trainMat)

s3eDb = []
for i in validateDb:
    s3eDb.append(poseDb[int(i[:-10])])

s3eQ = []

for i in validateQ:
    s3eQ.append(poseDb[int(i[:-10])])

trainMat = {}
temp = np.array((
    ['val'],
    [validateDb],
    [s3eDb],
    [validateQ],
    [s3eQ],
    [len(validateDb)],
    [len(validateQ)],
    [3],
    [25],
    [16]
))
trainMat['dbStruct'] = temp
scipy.io.savemat('./s3e_val.mat', trainMat)

s3eDb = []
for i in testDb:
    s3eDb.append(poseDb[int(i[:-10])])

s3eQ = []

for i in testQ:
    s3eQ.append(poseDb[int(i[:-10])])

trainMat = {}
temp = np.array((
    ['val'],
    [testDb],
    [s3eDb],
    [testQ],
    [s3eQ],
    [len(testDb)],
    [len(testQ)],
    [3],
    [25],
    [16]
))
trainMat['dbStruct'] = temp
scipy.io.savemat('./s3e_test.mat', trainMat)


trainMat = scipy.io.loadmat('./test_mat.mat')
struct = trainMat['dbStruct']
print((struct[0][0]))

whichSet = struct[0][0]

dbImage = [f for f in struct[1][0]]
print(dbImage)

s3eDb = struct[2][0]

qImage = [f for f in struct[3][0]]
print(qImage)
s3eQ = struct[4][0]

numDb = struct[5][0]
numQ = struct[6][0]

posDistThr = struct[7][0][0][0]
posDistSqThr = struct[8][0][0][0]
nonTrivPosDistSqThr = struct[9][0][0][0]
print(posDistSqThr)


'''
dtype=[('whichSet', 'O'), ('dbImageFns', 'O'), ('utmDb', 'O'), ('qImageFns', 'O'), ('utmQ', 'O'), ('numImages', 'O'), ('numQueries', 'O'), ('posDistThr', 'O'), ('posDistSqThr', 'O'), ('nonTrivPosDistSqThr', 'O')]

for i in trainQ:
    origin_path = os.path.join(origin_prefix, i)
    path = os.path.join(test_prefix, i)
    shutil.move(origin_path, path)

for i in testQ:
    origin_path = os.path.join(origin_prefix, i)
    path = os.path.join(test_prefix, i)
    shutil.move(origin_path, path)

for i in validateQ:
    origin_path = os.path.join(origin_prefix, i)
    path = os.path.join(test_prefix, i)
    shutil.move(origin_path, path)
'''

# print(trainDb)

'''

