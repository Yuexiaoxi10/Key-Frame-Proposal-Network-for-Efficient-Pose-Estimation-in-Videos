#from https://github.com/bearpaw/pytorch-pose/blob/master/evaluation/eval_PCKh.py
import sys
from scipy.io import loadmat
from numpy import transpose
# import skimage.io as sio
import numpy as np
import os
from h5py import File
import torch


def get_PCKh_penn(Test_gt, Test_out, Visbility, Bbox, nFrames, normTorso):
    # adopted code from : https://github.com/lawy623/LSTM_Pose_Machines/blob/master/testing/src/run_benchmark_GPU_PENN.m
    # Penn Action Official Joints Info, Menglong
    # 0.  head
    # 1.  left_shoulder  2.  right_shoulder
    # 3.  left_elbow     4.  right_elbow
    # 5.  left_wrist     6.  right_wrist
    # 7.  left_hip       8.  right_hip
    # 9.  left_knee      10. right_knee
    # 11. left_ankle     12. right_ankle

    # orderToPENN = [0 2 5 4 7 5 8 9 12 10 13 11 14];
    gtJointOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    thresh = 0.2


    # torso_norm = 1 # 1: Torso / 0:bbx; default as 0 -> 0.2*max(h,w)
    sample_num = Test_gt.shape[0]

    HitPoint = np.zeros((sample_num, len(gtJointOrder)))
    visible_joint = np.zeros((sample_num, len(gtJointOrder)))

    for sample in range(0, sample_num):
        # print('test sample:', sample)
        test_gt = Test_gt[sample]
        test_out = Test_out[sample]
        visibility = Visbility[sample]
        bbox = Bbox[sample]
        nframes = nFrames[sample].int()

        if nframes >= test_gt.shape[0]:
            nfr = test_gt.shape[0]
        else:
            nfr = nframes

        # num_frame = test_gt.shape[0]
        seqError = torch.zeros(nfr, len(gtJointOrder))

        seqThresh = torch.zeros(nfr, len(gtJointOrder))
        for frame in range(0, nfr):
            gt = test_gt[frame] # 13x2
            pred = test_out[frame] # 13x2
            # vis = visibility[frame] # 1x13

            if normTorso:
                bodysize = torch.norm(gt[2] - gt[7])
                if bodysize < 1:
                    bodysize = torch.norm(pred[2] - pred[7])
            else:
                bodysize = torch.max(bbox[frame,2]-bbox[frame, 0], bbox[frame, 3] - bbox[frame, 1])


            error_dis = torch.norm(gt-pred, p=2, dim=1, keepdim=False)

            seqError[frame] = error_dis
            seqThresh[frame] = (bodysize*thresh) * torch.ones(len(gtJointOrder))

        vis = visibility[0:nfr]
        visible_joint[sample] = np.sum(vis.numpy(), axis=0)
        less_than_thresh = np.multiply(seqError.numpy()<=seqThresh.numpy(), vis.numpy())
        # visibleJoint = np.sum(visibility.numpy(), axis=0)
        HitPoint[sample] = np.sum(less_than_thresh, axis=0)

    # finalPCK = np.divide(np.sum(HitPoint, axis=0), np.sum(np.sum(Visbility.numpy(), axis=1), axis=0))
    finalPCK = np.divide(np.sum(HitPoint, axis=0), np.sum(visible_joint, axis=0))
    finalMean = np.mean(finalPCK)
    print('normTorso,    Head,      Shoulder,   Elbow,    Wrist,     Hip,     Knee,    Ankle,  Mean')
    print('{:5s}        {:.4f}      {:.4f}     {:.4f}     {:.4f}      {:.4f}    {:.4f}    {:.4f}   {:.4f}'.format(str(normTorso),
          finalPCK[0], 0.5*(finalPCK[1]+finalPCK[2]), 0.5*(finalPCK[3]+finalPCK[4]), 0.5*(finalPCK[5]+finalPCK[6]),
          0.5*(finalPCK[7]+finalPCK[8]), 0.5*(finalPCK[9]+finalPCK[10]),  0.5*(finalPCK[11]+finalPCK[12]), finalMean))

    return finalMean, finalPCK


def get_PCKh_jhmdb(Test_gt, Test_out, Bbox, nFrames,imgPath ,normTorso):

    # 0: neck    1:belly   2: face
    # 3: right shoulder  4: left shoulder
    # 5: right hip       6: left hip
    # 7: right elbow     8: left elbow
    # 9: right knee      10: left knee
    # 11: right wrist    12: left wrist
    # 13: right ankle    14: left ankle

    orderJHMDB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    thresh = 0.2
    N = Test_out.shape[1]
    if normTorso:
        torso_norm = 1 # 1: Torso / 0:bbx; default as 0 -> 0.2*max(h,w)
    else:
        torso_norm = 0
    sample_num = Test_gt.shape[0]

    HitPoint = np.zeros((sample_num, len(orderJHMDB)))
    allPoint = np.ones((sample_num,  N, len(orderJHMDB)))
    Point_to_use = np.ones((sample_num, len(orderJHMDB)))


    for sample in range(0, sample_num):
        # print('test sample:', sample)
        test_gt = Test_gt[sample]
        test_out = Test_out[sample]
        nframes = nFrames[sample]
        img_path = imgPath[sample]
        bbox = Bbox[sample]

        # num_frame = test_gt.shape[0]
        if nframes >= test_gt.shape[0]:
            nfr = test_gt.shape[0]
        else:
            nfr = nframes.int()

        seqError = torch.zeros(nfr, len(orderJHMDB))
        seqThresh = torch.zeros(nfr,  len(orderJHMDB))

        for frame in range(0, nfr):
            gt = test_gt[frame] # 13x2
            pred = test_out[frame] # 13x2
            # vis = visibility[frame] # 1x13

            if torso_norm == 1:
                bodysize = torch.norm(gt[4] - gt[5])
                if bodysize < 1:
                    bodysize = torch.norm(pred[4] - pred[5])
            else:
                bodysize = torch.max(bbox[frame, 2]-bbox[frame, 0], bbox[frame, 3] - bbox[frame, 1])

            error_dis = torch.norm(gt-pred, p=2, dim=1, keepdim=False)

            seqError[frame] = torch.FloatTensor(error_dis)
            # seqThresh[frame] = (bodysize * thresh) * torch.ones(partJHMDB)
            seqThresh[frame] = (bodysize*thresh) * torch.ones(len(orderJHMDB))

        pts = allPoint[sample, 0:nfr]
        Point_to_use[sample] = np.sum(pts, axis=0)

        less_than_thresh = seqError.numpy()<=seqThresh.numpy()
        HitPoint[sample] = np.sum(less_than_thresh, axis=0)
        eachPCK = np.divide(np.sum(HitPoint[sample], axis=0), np.sum(Point_to_use[sample], axis=0))
        eachMean = np.mean(eachPCK)

        # print('sample num:', sample, 'imgpath:', img_path, 'eachMean:', eachMean)


    finalPCK = np.divide(np.sum(HitPoint, axis=0), np.sum(Point_to_use, axis=0))

    finalMean = np.mean(finalPCK)
    print('normTorso,     Head,     Shoulder,     Elbow,     Wrist,     Hip,      Knee,     Ankle,    Mean')

    print('{:5s}          {:.4f}      {:.4f}      {:.4f}     {:.4f}     {:.4f}    {:.4f}    {:.4f}   {:.4f}'.format(str(normTorso), finalPCK[2],
          0.5*(finalPCK[3]+finalPCK[4]), 0.5*(finalPCK[7]+finalPCK[8]), 0.5*(finalPCK[11]+finalPCK[12]),
          0.5*(finalPCK[5]+finalPCK[6]), 0.5*(finalPCK[9]+finalPCK[10]),  0.5*(finalPCK[13]+finalPCK[14]), finalMean))

    return finalMean, finalPCK

def eval_PCKh(dict, preds, model_name, idx):
    threshold = 0.5
    SC_BIAS = 0.6
    pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]

    dataset_joints = dict['dataset_joints']
    jnt_missing = dict['jnt_missing'][:, idx]
    # pos_pred_src = dict['pos_pred_src']
    pos_gt_src = dict['pos_gt_src'][:,:,idx]
    headboxes_src = dict['headboxes_src'][:,:,idx]



    #predictions
    # model_name = 'hg'
    # predfile = sys.argv[1]
    # preds = loadmat(predfile)['preds']

    pos_pred_src = transpose(preds, [1, 2, 0])

    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)


    # save
    rng = np.arange(0, 0.5, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err < threshold, jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    # name = predfile.split(os.sep)[-1]
    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True
    print("Model,  Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean")
    print('{:5s}   {:.2f}   {:.2f}     {:.2f}   {:.2f}    {:.2f}    {:.2f}    {:.2f}   {:.2f}'.format(model_name, PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho])\
            , 0.5 * (PCKh[lelb] + PCKh[relb]),0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
            , 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)))


if __name__ == '__main__':

    """""
    file_path = '/home/yuexi/Documents/python-outRemove/src/tools/data/'

    gt_file = 'detections_our_format.mat'
    dict = loadmat(os.path.join(file_path, gt_file))

    # pred_file = 'valid-ours.h5'
    pred_file_py = 'hg_pytorch_preds.mat'
    preds_PY = loadmat(os.path.join(file_path, pred_file_py))['preds']


    pred_file_DL = 'pred_multiscale_1.h5'
    file_DL = File(os.path.join(file_path, pred_file_DL), 'r')
    preds_DL = np.asarray(file_DL['preds'])
    preds_DL = preds_DL[:,:,0:2]
    # model_name =

    pred_file_lua = 'hg_lua.h5'
    file_lua = File(os.path.join(file_path, pred_file_lua), 'r')
    preds_lua = np.asarray(file_lua['preds'])


    idx = [0, 1, 4, 6, 7, 8, 9]

    eval_PCKh(dict, preds_DL, 'DLCM', idx)
    eval_PCKh(dict, preds_PY, 'hg_py', idx)
    eval_PCKh(dict, preds_lua, 'hg_lua', idx)
    """""
    test_gt = torch.randn(10, 50, 13, 2)
    test_out = torch.randn(10, 50, 13, 2)
    visibility = torch.randint(0,2,(10, 50, 13))
    bbox = torch.randn(10, 50, 4)
    nFrames = torch.randint(10, 70, (10,))
    get_PCKh_penn(test_gt, test_out, visibility, bbox, nFrames, normTorso=False)

    print('ok')