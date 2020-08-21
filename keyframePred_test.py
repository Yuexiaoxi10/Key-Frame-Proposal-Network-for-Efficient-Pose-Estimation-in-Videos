from modelZoo.networks import *
from utils import *

from JHMDB_dloader import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
import statistics
from torch.optim import lr_scheduler
import torch

def test_val_online(net, testloader,alpha,FRA, PRE ,Dict_pose_use, dataset_test, gpu_id):
    with torch.no_grad():
        T = Dict_pose_use.shape[0]
        # predLen = T - FRA
        numJoint = 15
        dim = numJoint*2

        imPath = []

        gtData = []
        testBase = []
        testBase_wo = []

        BBOX = []
        nFrames = []


        numKey_in = 0
        numKey_pred = 0
        sampleNum = 0
        keyFrame_in = []
        keyFrame_up = []
        imgPath = []
        for i, sample in enumerate(testloader):

            sequence_to_use = sample['sequence_to_use']  # already normalized
            img_data = sample['imgSequence_to_use']
            bbox = sample['Bbox_to_use']
            nframes = sample['nframes']
            inputData = img_data[0].cuda(gpu_id)
            imagePath = sample['imgPath'][0]
            imgPath.append(imagePath)
            # baseline = sample['baseline']
            if nframes >= 0:
                print('testing sample:', i, 'impath:', imagePath)
                sampleNum += 1
                imPath.append(imagePath)
                if len(inputData.shape) == 5:
                    inputData = inputData.squeeze(0)
                else:
                    inputData = inputData

                sparseCode_key, Dictionary, keylist_to_pred, keylist_FRA, key_list, imgFeature = net.get_keylist(inputData, alpha)
                predKeylist = []



                skeletonData = sequence_to_use[0].type(torch.FloatTensor).cuda(gpu_id)
                GT = skeletonData[0:FRA+PRE].reshape(1, FRA+PRE, dim)


                if predKeylist == []:
                    keyList_full = keylist_FRA
                else:
                    keyList_full = keylist_FRA + predKeylist

                # print('gt key:',key_list ,'pred key:', keyList_full)

                if keyList_full == []:
                    # y_hat_gt = torch.zeros(GT.shape)
                    y_hat_base = torch.zeros(GT.shape).cuda(gpu_id)
                    y_hat_base_wo = torch.zeros(GT.shape).cuda(gpu_id)
                else:

                    y_hat_base = get_recover_fista(Dict_pose_use[0:FRA+PRE].cuda(gpu_id), GT, keyList_full,0.1, gpu_id)
                    y_hat_base_wo = get_recover_fista(Dict_pose_use[0:FRA+PRE].cuda(gpu_id), GT, key_list,0.1, gpu_id)


                numKey_pred = numKey_pred + len(keyList_full)
                numKey_in = numKey_in + len(key_list)

                keyFrame_in.append(key_list)
                # keyFrame_up.append(keyList_full)
                keyFrame_up.append(len(keyList_full))


                test_gt = GT.squeeze(0).reshape(FRA+PRE, -1, 2).cpu()  # T x 15 x 2
                test_yhat_base = y_hat_base.squeeze(0).reshape(FRA+PRE, -1, 2).cpu()
                test_yhat_base_wo = y_hat_base_wo.squeeze(0).reshape(FRA+PRE, -1, 2).cpu()

                test_gt_unnorm = dataset_test.get_unNormalized_data(test_gt)
                test_base_unnorm = dataset_test.get_unNormalized_data(test_yhat_base)
                test_base_unnorm_wo = dataset_test.get_unNormalized_data(test_yhat_base_wo)


                'only consider prediction '
                # gtData[i] = test_gt_unnorm[FRA:T]
                # testBase[i] = test_base_unnorm[FRA:T]
                # BBOX[i] = bbox

                # gtData.append(test_gt_unnorm[FRA:T].unsqueeze(0))
                # testBase.append(test_base_unnorm[FRA:T].unsqueeze(0))
                # testBase_wo.append(test_base_unnorm_wo[FRA:T].unsqueeze(0))
                # BBOX.append(bbox[:, FRA:T].type(torch.FloatTensor))
                # nFrames.append(nframes - FRA)

                print('update list:', len(keyList_full), 'all keylist:', len(key_list))
                gtData.append(test_gt_unnorm.unsqueeze(0))
                testBase.append(test_base_unnorm.unsqueeze(0))
                testBase_wo.append(test_base_unnorm_wo.unsqueeze(0))

                BBOX.append(bbox.type(torch.FloatTensor))
                nFrames.append(nframes)
            else:
                continue


        meanNumKey_in = numKey_in / sampleNum
        meanNumKey_pred = numKey_pred / sampleNum

        gtData = torch.cat((gtData))
        testBase = torch.cat((testBase))
        BBOX = torch.cat((BBOX))
        testBase_wo = torch.cat((testBase_wo))

        print('update with keyframes')
        get_PCKh_jhmdb(gtData, testBase, BBOX, nFrames, imPath, normTorso=False)

        print('non-updating keyframes')
        get_PCKh_jhmdb(gtData, testBase_wo, BBOX, nFrames, imPath, normTorso=False)

        print('mean_keyframe pred:', meanNumKey_pred, 'mean_keyframe in', meanNumKey_in)

        keyFrame_up = np.asarray(keyFrame_up)
        print('for online mode:','max_keyframe:', np.max(keyFrame_up), 'min_keyframe:',
              np.min(keyFrame_up), 'std_keyframe:', np.std(keyFrame_up), 'median_keyframe:', np.median(keyFrame_up))


if __name__ == '__main__':
    'parameters'
    FRA = 30
    T = 40
    alpha = 3
    PRE = 10
    dim = 15*2
    gpu_id = 2

    dataRoot = '/data/Yuexi/JHMDB'
    trainAnnot, testAnnot = get_train_test_annotation(dataRoot)
    testSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='val',if_occ=False)
    testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=4)

    Data_to_use = scipy.io.loadmat('./testData/JHMDB_2DGauNorm_train_T40_DYAN.mat')
    Dictionary_pose = torch.Tensor(Data_to_use['Dictionary'])
    Dict_pose_use = Dictionary_pose[0:T, :]

    modelPath = '/home/yuexi/Documents/keyframeModel/JHMDB/keyframePred'
    modelFile = os.path.join(modelPath, '8.pth')
    state_dict = torch.load(modelFile)['state_dict']
    Drr = state_dict['K_FPN.Drr']
    Dtheta = state_dict['K_FPN.Dtheta']
    net = onlineUpdate(FRA=FRA, PRE=PRE,T=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)

    net.load_state_dict(state_dict)
    net.eval()
    net.cuda(gpu_id)
    test_val_online(net, testloader, alpha, FRA,PRE, Dict_pose_use.cuda(gpu_id), testSet, gpu_id)

    print('done')
    # net, testloader, alpha, FRA, PRE, Dict_pose_use, dataset_test, gpu_id