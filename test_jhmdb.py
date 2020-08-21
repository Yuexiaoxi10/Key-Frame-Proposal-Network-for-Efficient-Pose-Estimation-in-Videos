import time
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler

import torch.nn as nn
from modelZoo.DyanOF import creatRealDictionary
from modelZoo.networks import keyframeProposalNet, load_preTrained_model
from utils import *

from JHMDB_dloader import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
from ptflops import get_model_complexity_info
torch.manual_seed(0)
np.random.seed(0)

def test_val(net, testloader, alpha, thresh, Dictionary_pose, dataset_test, gpu_id):
    with torch.no_grad():
        T = Dictionary_pose.shape[0]
        keyFrames = []
        numKey = 0
        numJoint = 15

        sample_num = testloader.__len__()
        gtData = torch.zeros(sample_num, T, numJoint, 2)
        testData = torch.zeros(sample_num, T, numJoint, 2)


        imPath = []

        Time =[]
        BBOX = torch.zeros(sample_num, T, 4)
        nFrames = torch.zeros(sample_num)
        # t0 = time.time()
        for i, sample in enumerate(testloader):
            print('testing sample:', i)

            sequence_to_use = sample['sequence_to_use']  # already normalized
            img_data = sample['imgSequence_to_use']
            bbox = sample['Bbox_to_use']
            nframes = sample['nframes']
            # baseline_to_use = sample['baseline_to_use']

            inputData = img_data[0].cuda(gpu_id)
            imagePath = sample['imgPath']
            imPath.append(imagePath)

            if len(inputData.shape) == 5:
                inputData = inputData.squeeze(0)
            else:
                inputData = inputData

            t0 = time.time()

            # print('input shape:',inputData.shape)
            feature , Dictionary,_ = net.forward(inputData)
            out = net.forward2(feature, alpha)

            s = out[0, :]
            key_ind = (s > thresh).nonzero().squeeze(1)
            key_list = list(key_ind.cpu().numpy())
            # print('imgpath:', imagePath, 'keyframes:', len(key_list))
            keyFrames.append(len(key_list))
            numKey = numKey + len(key_list)

            skeletonData = sequence_to_use[0].type(torch.FloatTensor).cuda(gpu_id)
            # baselineData = baseline_to_use[0].type(torch.FloatTensor).cuda(gpu_id)

            dim = numJoint*2
            GT = skeletonData.reshape(1, T, dim)  # Tx30
            # baseline = baselineData.reshape(1, T, dim)

            if key_list == []:
                y_hat_gt = torch.zeros(GT.shape)
                # y_hat_gt = torch.zeros(baseline.shape)

            else:
                y_hat_gt = get_recover_fista(Dictionary_pose.cuda(gpu_id), GT, key_list, 0.1, gpu_id)  # for validation
                # y_hat_gt = get_recover_fista(Dictionary_pose.cuda(gpu_id), baseline, key_list, gpu_id) # for testing

            endtime = time.time() - t0
            Time.append(endtime)
            # print('time:', endtime)
            # endtime = time.time() - t0
            # print('time:', endtime)
            # get mpjpe
            test_gt = GT.squeeze(0).reshape(T, -1, 2).cpu()  # T x 15 x 2
            test_yhat_gt = y_hat_gt.squeeze(0).reshape(T, -1, 2).cpu()  # T x 15 x 2

            test_out_unnorm = dataset_test.get_unNormalized_data(test_yhat_gt)
            test_gt_unnorm = dataset_test.get_unNormalized_data(test_gt)


            gtData[i] = test_gt_unnorm
            testData[i] = test_out_unnorm
            BBOX[i] = bbox
            nFrames[i] = nframes
        # endtime = time.time() - t0
        # print('time:',endtime)

        totalTime = numKey * (0.4 / 40) + statistics.mean(Time) * sample_num  # for GeForce 1080i, roughly, Time(baseline) = 0.4 for each video;'
        print('time/fr ms:', 1000 * (totalTime / (T * sample_num)))

        meanNumKey = numKey / sample_num

        get_PCKh_jhmdb(gtData, testData, BBOX, nFrames, imPath, normTorso=False)

        print('mean_keyframe:', meanNumKey, 'max_keyframe:', np.max(keyFrames), 'min_keyframe:',
              np.min(keyFrames), 'std_keyframe:', np.std(keyFrames),'median_keyframe:', np.median(keyFrames))
        # with torch.cuda.device(gpu_id):
        #     flops, params = get_model_complexity_info(net, (3, 244, 244), as_strings=True, print_per_layer_stat=True)
        #     print('Flops:' + flops)
        #     # print('Params:' + params)
        # print('time:', endtime)
def random_select(keyframes, testSkeleton, Dictionary_pose, gpu_id):
    'testSkeleton is baseline skeleton'
    maxtIter = 100
    Y_hat = torch.zeros(maxtIter, T, 15, 2)
    L = len(testSkeleton)
    k = len(keyframes)

    for iter in range(0, maxtIter):
        keys = np.random.choice(L, k)

        y_hat = get_recover_fista(Dictionary_pose.cuda(gpu_id), testSkeleton, keys, gpu_id)
        Y_hat[iter] = y_hat

    return Y_hat

if __name__ == '__main__':
    data_root = '/data/Yuexi/JHMDB'
    Data_to_use = scipy.io.loadmat('./testData/JHMDB_2DGauNorm_train_T40_DYAN.mat')
    T = 40
    Dictionary_pose = torch.Tensor(Data_to_use['Dictionary'])
    Dict_to_use = Dictionary_pose[0:T,:]

    numJoint = 15

    gpu_id = 3
    trainAnnot, testAnnot = get_train_test_annotation(data_root)
    # dataset_test = jhmdbDataset(trainAnnot, testAnnot, T, split='test', if_occ=False)
    # testloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    dataset_test = jhmdbDataset(trainAnnot, testAnnot, T, split='val', if_occ=False)
    testloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    modelFolder = './models'
    'Resent-18'

    modelFile = os.path.join(modelFolder, 'kfpn_jhmdb_resnet18.pth') # ResNet-18
    'Resent-34'
    # modelFile = os.path.join(modelFolder, 'kfpn_jhmdb_resnet34.pth') # ResNet-34

    'Resent-50'

    # modelFile = os.path.join(modelFolder, 'kfpn_jhmdb_resnet50.pth') # Resnet-50

    state_dict = torch.load(modelFile)['state_dict']

    Drr = state_dict['Drr']
    Dtheta = state_dict['Dtheta']

    N = 40*4

    net = keyframeProposalNet(numFrame=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id, backbone='Resnet18',config='jhmdb')
    newDict = net.state_dict()

    pre_dict = {k: v for k, v in state_dict.items() if k in newDict}

    newDict.update(pre_dict)

    net.load_state_dict(newDict)
    # net.load_state_dict(state_dict)
    net.cuda(gpu_id)
    net.eval()
    alpha = 3  # alpha is linearly increasing with epoch while training

    thresh = 0.992 # tunning threshhold to control number of keyframes you want to keep
    test_val(net, testloader, alpha, thresh, Dictionary_pose.cuda(gpu_id), dataset_test, gpu_id)



    print('done')