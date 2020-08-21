import torch
from PENN_dloader import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
from modelZoo.networks import *
from utils import get_recover_fista
from ptflops import get_model_complexity_info
import time
# from print_feature_map import *
import pickle


def test_val(net, testloader, alpha, thresh, Dict_pose_use, dataset_test, gpu_id):
    with torch.no_grad():
        T = Dict_pose_use.shape[0]
        keyFrames = []
        imgPath =[]
        numKey = 0
        Time = []
        # sample_num = len(testAnnot)
        sample_num = testloader.__len__()
        gtData = torch.zeros(sample_num, T, 13, 2)
        testData = torch.zeros(sample_num, T, 13, 2)
        testUniform = torch.zeros(sample_num, T, 13, 2)


        Y_rs_min = torch.zeros(sample_num, T, 13, 2)
        # Y_rs_mean = torch.zeros(sample_num, T, 13, 2)



        VIS = torch.zeros(sample_num, T, 13)
        BBOX = torch.zeros(sample_num, T, 4)
        nFrames = torch.zeros(sample_num)
        randIdx = []
        Error_all = []
        # finalMean_mean = []
        for i, sample in enumerate(testloader):

            print('starting to test sample:', i)
            skeleton_to_use = sample['skeleton_to_use']
            vis_to_use = sample['vis_to_use']
            bbox_to_use = sample['bbox_to_use']
            imgSequence_to_use = sample['imgSequence_to_use']
            nframes = sample['nframes']
            idx = sample['randIdx']
            imgFolderPath = sample['imgPath']
            imgPath.append(imgFolderPath[0])
            randIdx.append(idx)

            img_data = imgSequence_to_use.squeeze(0)
            inputData = img_data.cuda(gpu_id)

            t0 = time.time()
            feature, Dictionary,_ = net.forward(inputData)
            out = net.forward2(feature, alpha)
            # endtime = time.time() - t0
            # print('time:', endtime)

            s = out[0, :]
            key_ind = (s >= thresh).nonzero().squeeze(1)
            key_list = list(key_ind.cpu().numpy())
            # print('sample:', i, 'keyframes:', key_list)
            keyFrames.append(len(key_list))
            numKey = numKey + len(key_list)

            skeletonData = skeleton_to_use[0].type(torch.FloatTensor).cuda(gpu_id)

            input = skeletonData.reshape(1, T, 13 * 2)  # Tx26

            if key_list == []:
                y_hat = torch.zeros(input.shape)
            else:
                y_hat = get_recover_fista(Dict_pose_use.cuda(gpu_id), input, key_list,0.01, gpu_id)
            endtime = time.time() - t0
            # print('time:', endtime)
            Time.append(endtime)



            test_gt = input.reshape(T, -1, 2).cpu()  # T x 13 x 2
            test_yhat = y_hat.reshape(T, -1, 2).cpu()  # T x 13 x 2
            test_gt_unnorm = dataset_test.unnormData(test_gt.unsqueeze(0))
            test_out_unnorm = dataset_test.unnormData(test_yhat.unsqueeze(0))


            """""
            'uniform sampling'
            idx = np.round(T / len(key_list))
            n = int(T / idx)
            key_uniform = idx * np.linspace(0, n-1, n, dtype=np.int16)
            # print('uni keylist:', key_uniform)
            y_hat_uniform = get_recover_fista(Dict_pose_use.cuda(gpu_id), input, list(key_uniform), gpu_id)

            test_uniform = dataset_test.unnormData(y_hat_uniform.reshape(T, -1, 2).cpu().unsqueeze(0))
            testUniform[i] = test_uniform


            'Random selection'

            Y_hat_rs, Error, _,_ = random_select(key_list, input, Dict_pose_use, gpu_id)

            Error_all.append(np.min(np.asarray(Error)))
            minIndex = np.where(np.min(np.asarray(Error)))[0]
            Y_hat_min = Y_hat_rs[minIndex].cpu().reshape(T, -1, 2)
            Y_hat_min_unnorm = dataset_test.unnormData(Y_hat_min.unsqueeze(0)) # recovery corresponding with the minimum error
            Y_rs_min[i] = Y_hat_min_unnorm

            """""
            gtData[i] = test_gt_unnorm
            testData[i] = test_out_unnorm

            VIS[i] = vis_to_use
            BBOX[i] = bbox_to_use
            nFrames[i] = nframes
        #
        # with torch.cuda.device(gpu_id):
        #     flops, params = get_model_complexity_info(net, (3, 244, 244), as_strings=True, print_per_layer_stat=True)
        #     print('Flops:' + flops)
        #     print('Params:' + params)
        # print('time:', endtime)

        'min error of random selection'
        # minError = np.mean(np.asarray(Error_all))
        # print(minError)

        meanNumKey = numKey / sample_num

        'totalTime = numKey * (baseline time per video / T) + statistics.mean(Time) * sample_num '
        # print('time/fr ms:', 1000 * (totalTime / (T * sample_num)))


        keyFrames = np.asarray(keyFrames)

        finalMean_kfpn, _ = get_PCKh_penn(gtData, testData, VIS, BBOX, nFrames, normTorso=False)

        # finalMean_uniform , _ = get_PCKh_penn(gtData, testUniform, VIS, BBOX, nFrames, normTorso=False)

        # finalMean_min, _ = get_PCKh_penn(gtData, Y_rs_min, VIS, BBOX, nFrames, normTorso=False)

        print('mean_keyframe:', meanNumKey, 'max_keyframe:', np.max(keyFrames), 'min_keyframe:', np.min(keyFrames), 'std_keyframe:', np.std(keyFrames),
              'median_keyframe:', np.median(keyFrames))
        # print('mean kfpn:', finalMean_kfpn,'mean min', finalMean_min, 'mean uniform:', finalMean_uniform)


        print('done')


def random_select(keyframes, testSkeleton,Dictionary_pose, gpu_id):
    'testSkeleton is baseline skeleton'
    maxtIter = 100
    # Y_hat = torch.zeros(maxtIter, T, 13, 2)
    Y_hat = torch.zeros(maxtIter, T, 26)
    # L = len(testSkeleton)
    L = testSkeleton.shape[1] # T = 40
    k = len(keyframes)
    Error = []
    PCK = []
    KEYS = []
    for iter in range(0, maxtIter):
        keys = np.random.choice(L, k)
        keys.sort()


        y_hat = get_recover_fista(Dictionary_pose.cuda(gpu_id), testSkeleton, keys, gpu_id)

        Y_hat[iter] = y_hat.squeeze(0)
        error = torch.norm(testSkeleton-y_hat)
        Error.append(error.data.item())

    return Y_hat, Error, KEYS, PCK

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    data_root = '/data/Yuexi/Penn_Action'
    T = 40
    gpu_id = 3
    trainAnnot, testAnnot = get_train_test_annot(data_root)
    dataset_test = pennDataset(trainAnnot, testAnnot, T, split='val')
    testloader = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=8)
    Data_to_use = scipy.io.loadmat('./testData/PENN_2DGauNorm_train_T60_DYAN_mask.mat')
    Dictionary_pose = torch.Tensor(Data_to_use['Dictionary'])
    Dict_pose = Dictionary_pose[0:T,:]

    modelPath = './models'
    'Resnet-101'
    # modelFile = os.path.join(modelPath, 'kfpn_penn_resent101.pth')

    'Resnet-50'

    # modelFile = os.path.join(modelPath, 'kfpn_penn_resnet50.pth')

    'Resnet-34'

    modelFile = os.path.join(modelPath, 'kfpn_penn_resnet34.pth')

    state_dict = torch.load(modelFile)['state_dict']
    Drr = state_dict['Drr']
    Dtheta = state_dict['Dtheta']

    # net = keyframeProposalNet(numFrame=T, gpu_id=gpu_id, if_bn=True, if_init=True)
    net = keyframeProposalNet(numFrame=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id, backbone='Resnet34', config='Penn')
    net.load_state_dict(state_dict)
    net.eval()

    net.cuda(gpu_id)

    alpha = 4
    thresh = 0.99

    test_val(net, testloader, alpha, thresh, Dict_pose, dataset_test, gpu_id)
    print('done')
