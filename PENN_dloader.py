import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from h5py import File
import os
import scipy.io
import statistics
import random
from PIL import Image
from torchvision import transforms
from six.moves import xrange
# Penn Action Official Joints Info, Menglong
# 0.  head
# 1.  left_shoulder  2.  right_shoulder
# 3.  left_elbow     4.  right_elbow
# 5.  left_wrist     6.  right_wrist
# 7.  left_hip       8.  right_hip
# 9.  left_knee      10. right_knee
# 11. left_ankle     12. right_ankle

# Penn-Crop Joints Info for Dataset, Yuwei
# 0.  head
# 1.  right_shoulder  2.  left_shoulder
# 3.  right_elbow     4.  left_elbow
# 5.  right_wrist     6.  left_wrist
# 7.  right_hip       8.  left_hip
# 9.  right_knee      10. left_knee
# 11. right_ankle     12. left_ankle


def get_train_test_annot(data_root):
    imgPath = os.path.join(data_root, 'frames')
    use_baseline = False

    if use_baseline:
        'dataPath = your baseline path'
        'make your baseline as the same format as labels'
    else:

        dataPath = os.path.join(data_root, 'labels')

    skeletonPath = dataPath

    dataList = os.listdir(imgPath)
    dataList.sort()
    trainAnnot = []
    testAnnot = []

    for i in range(0, len(dataList)):
        mat_name = dataList[i] + '.mat'
        annot = scipy.io.loadmat(os.path.join(skeletonPath, mat_name))
        imgFolderPath = os.path.join(imgPath, dataList[i])
        dict = {'imgFolderPath': imgFolderPath, 'annot': annot}
        if annot['train'] == 1:
            trainAnnot.append(dict)
        else:
            testAnnot.append(dict)
    # print(len(trainAnnot), len(testAnnot))
    return trainAnnot, testAnnot

class pennDataset(data.Dataset):
    def __init__(self, trainAnnot, testAnnot, T, split):
        # self.dataRoot = data_root

        self.trainSet = trainAnnot[0:1000]
        self.testSet = testAnnot
        self.valSet = trainAnnot[1000:]
        self.split = split
        self.inputLen = T
        self.numJoint = 13
        if self.split == 'train':
            self.dataLen = len(self.trainSet)
        elif self.split == 'val':
            self.dataLen = len(self.valSet)
        else:
            self.dataLen = len(self.testSet)


        numData = len(self.trainSet)

        X = []
        Y = []
        for i in xrange(numData):
            annot = self.trainSet[i]['annot']
            x = annot['x'].astype(float)
            y = annot['y'].astype(float)
            X.append(x)
            Y.append(y)

            # X[i] = x
            # Y[i] = y
        X = np.concatenate((X), axis=0)
        Y = np.concatenate((Y), axis=0)
        self.meanX = np.mean(X, axis=0)
        self.stdX = np.std(X, axis=0)
        self.meanY = np.mean(Y, axis=0)
        self.stdY = np.std(Y, axis=0)


    def __len__(self):
        return self.dataLen

        # return 1   # for debug

    def read_annot(self, annotSet):
        imgFolderPath = annotSet['imgFolderPath']
        x = annotSet['annot']['x'].astype(float)
        y = annotSet['annot']['y'].astype(float)

        skeleton = np.concatenate((np.expand_dims(x, 2),np.expand_dims(y, 2)), axis=2).astype(float) # T x 13 x 2

        visibility = annotSet['annot']['visibility'].astype(float)
        nframes = annotSet['annot']['nframes']
        bbox = annotSet['annot']['bbox'].astype(float)
        # self.numJoint = skeleton.shape[1]

        xnorm = (x - self.meanX)/self.stdX
        ynorm = (y - self.meanY)/self.stdY

        skeleton_norm = np.concatenate((np.expand_dims(xnorm, 2), np.expand_dims(ynorm, 2)), axis=2).astype(float)

        return imgFolderPath, skeleton, skeleton_norm, visibility, np.double(nframes), bbox

    def unnormData(self, skeletonNorm):
        meanX_mat = np.expand_dims(self.meanX, 1).repeat(self.inputLen, 1).transpose(1, 0) # T x 13
        meanY_mat = np.expand_dims(self.meanY, 1).repeat(self.inputLen, 1).transpose(1, 0)
        stdX_mat = np.expand_dims(self.stdX,1).repeat(self.inputLen, 1).transpose(1, 0)
        stdY_mat = np.expand_dims(self.stdY,1).repeat(self.inputLen, 1).transpose(1, 0)

        unnorm_X = skeletonNorm[:,:,:,0].double() * torch.DoubleTensor(np.expand_dims(stdX_mat, 0)) + torch.DoubleTensor(np.expand_dims(meanX_mat, 0))
        unnorm_Y = skeletonNorm[:,:,:,1].double() * torch.DoubleTensor(np.expand_dims(stdY_mat, 0)) + torch.DoubleTensor(np.expand_dims(meanY_mat, 0))

        skeleton_unnorm = torch.cat((unnorm_X.unsqueeze(3), unnorm_Y.unsqueeze(3)), 3)

        return skeleton_unnorm

    def preProcessImage(self, imgFolderPath):
        imgList = os.listdir(imgFolderPath)
        imgList.sort()  # list image in a descent order
        imgSequence = []
        for i in xrange(len(imgList)):
            imgPath = os.path.join(imgFolderPath, imgList[i])
            input_image = Image.open(imgPath)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = preprocess(input_image)
            imgSequence.append(img_tensor.unsqueeze(0))

        imgSequence = torch.cat((imgSequence), 0)

        return imgSequence

    def data_to_use(self, nframes, skeleton, visibility, bbox, imgSequence, shiftTimes):
        """""
        'shift times: we used T=40 for each training or testing sequence, if given video is more than 40, for training, we randomly pick 
         concecutive 40 frames; for testing, we shifted sequence until the getting the last frame. 
         
         'if the sequence is less than 40, we added dummy frames behind'
        """""
        random.seed(1234567890)
        useLen = self.inputLen
        stepSize = 10
        if len(skeleton.shape) == 4 and len(visibility.shape) == 3:
            skeleton = skeleton.squeeze(0)
            visibility = visibility.squeeze(0)
            bbox = bbox.squeeze(0)

        if len(nframes.shape) == 1: # if nframes is a tensor
            nframes = nframes.int()
        else:
            nframes = int(nframes)

        if nframes > useLen + stepSize*shiftTimes:

            idx = random.randint(0, nframes - self.inputLen - 10*shiftTimes)
            sequence = []
            vis = []
            bbx = []
            sequence_img = []
            for i in range(0, shiftTimes):
                data_sel = skeleton[idx+i*stepSize:idx+i*stepSize+self.inputLen]
                vis_sel = visibility[idx+i*stepSize:idx+i*stepSize+self.inputLen]
                bbx_sel = bbox[idx+i*stepSize:idx+i*stepSize+self.inputLen]
                img_sel = imgSequence[idx+i*stepSize:idx+i*stepSize+self.inputLen] # tensor


                sequence_img.append(img_sel)
                sequence.append(np.expand_dims(data_sel, 0))
                vis.append(np.expand_dims(vis_sel, 0))
                bbx.append(np.expand_dims(bbx_sel, 0))

            sequence_to_use = np.concatenate(sequence, axis=0).astype(float)
            vis_to_use = np.concatenate(vis, axis=0).astype(float)
            bbox_to_use = np.concatenate(bbx, axis=0).astype(float)
            imgSequence_to_use = torch.cat((sequence_img), 0).type(torch.FloatTensor)

            mask_idx = np.ones((sequence_to_use.shape))

        elif  useLen <=  nframes <= useLen + 10*shiftTimes:
            idx = random.randint(0, nframes-useLen)

            data_sel = np.expand_dims(skeleton[idx : idx+useLen], 0)
            vis_sel = np.expand_dims(visibility[idx : idx+useLen], 0)
            bbx_sel = np.expand_dims(bbox[idx : idx+useLen], 0)
            img_sel = imgSequence[idx : idx+useLen]

            sequence_to_use = data_sel
            vis_to_use = vis_sel
            bbox_to_use = bbx_sel
            imgSequence_to_use = img_sel

            mask_idx = np.ones((sequence_to_use.shape))

        else:
            seqLeft = useLen - nframes
            sequence = []
            vis = []
            bbx = []
            sequence_img = []
            m_idx = []
            idx = 0
            for i in xrange(seqLeft):

                'padding 0'

                mask_sel = np.zeros((self.numJoint, 2))
                data_sel = np.zeros((self.numJoint, 2))
                vis_sel = np.zeros((self.numJoint))
                bbx_sel = np.zeros((4))
                img_sel = torch.zeros(3, 224, 224)

                sequence_img.append(img_sel.unsqueeze(0))
                sequence.append(np.expand_dims(data_sel, 0))
                vis.append(np.expand_dims(vis_sel, 0))
                bbx.append(np.expand_dims(bbx_sel, 0))
                m_idx.append(np.expand_dims(mask_sel, 0))

            sequence = np.concatenate(sequence, axis=0)
            vis = np.concatenate(vis, axis=0)
            bbx = np.concatenate(bbx, axis=0)
            sequence_img = torch.cat((sequence_img),0)
            ma_idx = np.concatenate(m_idx, axis=0)

            sequence_to_use = np.concatenate((skeleton, sequence), axis=0).astype(float)
            vis_to_use = np.concatenate((visibility, vis), axis=0).astype(float)
            bbox_to_use = np.concatenate((bbox, bbx), axis=0).astype(float)
            imgSequence_to_use = torch.cat((imgSequence, sequence_img), 0).type(torch.FloatTensor) # T x 3 x 224 x 224


            mask_part1 = np.ones((skeleton.shape))
            mask_idx = np.concatenate((mask_part1, ma_idx), axis=0).astype(float)


        return sequence_to_use, vis_to_use, bbox_to_use, imgSequence_to_use, mask_idx, idx



    def __getitem__(self, idx):
        if self.split == 'train':
            annotSet = self.trainSet[idx]
        else:
            annotSet = self.testSet[idx]

        imgFolderPath, skeleton, skeleton_norm, visibility, nframes, bbox = self.read_annot(annotSet)
        imgSequence = self.preProcessImage(imgFolderPath)

        skeleton_to_use, vis_to_use, bbox_to_use, imgSequence_to_use, mask_idx, idx = self.data_to_use(nframes, skeleton_norm,
                                                                                           visibility, bbox,
                                                                                           imgSequence, shiftTimes=1)



        dict = {'skeleton_to_use': skeleton_to_use, 'vis_to_use': vis_to_use, 'bbox_to_use': bbox_to_use,
                'imgSequence_to_use': imgSequence_to_use, 'mask_idx': mask_idx, 'nframes':nframes, 'randIdx': idx,
                'imgPath': imgFolderPath}
        return dict




if __name__ == '__main__':
    data_root = '/data/Yuexi/Penn_Action'

    trainAnnot, testAnnot = get_train_test_annot(data_root)

    dataset = pennDataset(trainAnnot, testAnnot, T=60, split='train')
    dloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    NFRA = np.zeros((1258, 1))  # test: 1068/ train: 1258
    for i, sample in enumerate(dloader):


        skeleton_to_use = sample['skeleton_to_use']
        vis_to_use = sample['vis_to_use']
        bbox_to_use = sample['bbox_to_use']
        imgSequence_to_use = sample['imgSequence_to_use']
        mask_idx = sample['mask_idx']

        # print(i, imgSequence_to_use.shape[1])
        print(i, mask_idx.shape)
        # print('done')
        # print(imgFolderPath)

    print('done')
