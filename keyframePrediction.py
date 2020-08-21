from modelZoo.networks import *
from utils import *

from JHMDB_dloader import *
from keyframePred_test import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
from torch.optim import lr_scheduler
import torch

dataRoot = 'your dataroot'

Data_to_use = scipy.io.loadmat('./testData/JHMDB_2DGauNorm_train_T40_DYAN.mat')
saveModel = '/your path/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)

'parameters'
T = 40
FRA = 30 # input length
PRE = 10 # pred length
dim = 15*2
gpu_id = 3
alpha = 3
EPOCH = 200


Dictionary_pose = torch.Tensor(Data_to_use['Dictionary'])
Dict_to_use = Dictionary_pose[0:T,:].cuda(gpu_id)

trainAnnot, testAnnot,_ = get_train_test_annotation(dataRoot)
trainSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='train', if_occ=False)
trainloader = DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=4)

testSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='test', if_occ=False)
testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=4)


modelFolder = '/models'
# numJoint = 15

modelFile = os.path.join(modelFolder, 'kfpn_jhmdb_resnet18.pth')
state_dict = torch.load(modelFile)['state_dict']

Drr = state_dict['Drr']
Dtheta = state_dict['Dtheta']


net = onlineUpdate(FRA=FRA, PRE=PRE, T=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)

        nn.init.constant_(m.bias, 0)


newDict = net.K_FPN.state_dict()
pre_dict = {k: v for k, v in state_dict.items() if k in newDict}
newDict.update(pre_dict)
net.K_FPN.load_state_dict(newDict)
for param in net.K_FPN.parameters():
    param.requires_grad = False

net.cuda(gpu_id)
optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-4, weight_decay=0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# print('check')
print('start training')

for epoch in range(1, EPOCH+1):
    lossVal = []
    # print('start training epoch:', epoch)
    for i, sample in enumerate(trainloader):
        # print('start training sample:', i)
        img_data = sample['imgSequence_to_use']
        inputData = img_data[0].cuda(gpu_id)

        if len(inputData.shape) == 5:
            inputData = inputData.squeeze(0)
        else:
            inputData = inputData
        sparseCode_key, Dictionary, keylist_to_pred, _, _, imgFeature = net.get_keylist(inputData, alpha)
        'keyframe prediction'
        Label = torch.zeros(PRE, dtype=torch.long).cuda(gpu_id)
        Out = torch.zeros(PRE, 2).cuda(gpu_id)
        for fraNum in range(FRA, FRA+PRE):
            # optimizer.zero_grad()
            if fraNum in keylist_to_pred:
                label = torch.ones([1], dtype=torch.long).cuda(gpu_id)
            else:
                label = torch.zeros([1], dtype=torch.long).cuda(gpu_id)
            Label[fraNum-FRA] = label

            out = net.forward(imgFeature, sparseCode_key, Dictionary, fraNum)
            Out[fraNum-FRA] = out

        loss = criterion(Out, Label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lossVal.append(loss.data.item())

    loss_val = np.mean(np.array(lossVal))
    print('Epoch:', epoch, '|loss:', loss_val)

    scheduler.step()
    if epoch % 1 == 0:
        torch.save({'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')


print('done')
