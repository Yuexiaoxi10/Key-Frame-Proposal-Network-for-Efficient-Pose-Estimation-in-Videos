import torch
# torch.manual_seed(0)
import torch.nn as nn
from modelZoo.resNet import ResNet, Bottleneck, BasicBlock
from modelZoo.DyanOF import creatRealDictionary
from utils import generateGridPoles, gridRing,fista
import numpy as np

def load_preTrained_model(pretrained, newModel):
    'load pretrained resnet-X to self defined model '
    'modified resnet has no last two layers, only return feature map'

    pre_dict = pretrained.state_dict()

    new_dict = newModel.state_dict()

    pre_dict = {k: v for k, v in pre_dict.items() if k in new_dict}

    new_dict.update(pre_dict)

    newModel.load_state_dict(new_dict)

    for param in newModel.parameters():
        param.requires_grad = False

    return newModel


class keyframeProposalNet(nn.Module):
    def __init__(self, numFrame, Drr, Dtheta, gpu_id, backbone, config):
        super(keyframeProposalNet, self).__init__()
        self.num_frame = numFrame
        self.gpu_id = gpu_id
        self.backbone = backbone
        self.config = config
        if self.backbone == 'Resnet101':
            self.modifiedResnet = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], zero_init_residual=False,
                                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                         norm_layer=None)  # ResNet-101
            self.Conv2d = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
            self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        elif self.backbone == 'Resnet50':
            self.modifiedResnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=False,
                                    groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                    norm_layer=None)  # ResNet-50
            self.Conv2d = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
            self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        elif self.backbone == 'Resnet34':
            self.modifiedResnet = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], zero_init_residual=False,
                                    groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                    norm_layer=None)  # ResNet-34
            # self.layer2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)

        elif self.backbone == 'Resnet18':
            self.modifiedResnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                         norm_layer=None)  # Resent-18

        self.relu = nn.LeakyReLU(inplace=True)

        'downsample feature map'
        self.layer2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn_l4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        self.Drr = nn.Parameter(Drr, requires_grad=True)
        self.Dtheta = nn.Parameter(Dtheta, requires_grad=True)

        'embeded infomation along time space'
        if self.config == 'Penn':
            self.fcn1 = nn.Conv2d(self.num_frame, 25, kernel_size=1, stride=2, padding=0, groups=1, bias=False, dilation=1)
            self.fc = nn.Linear(2560, self.num_frame)

        else:
            self.fcn1 = nn.Conv2d(self.num_frame, 25, kernel_size=1, stride=1, padding=0, groups=1, bias=False,
                                  dilation=1)
            self.fc = nn.Linear(5760, self.num_frame)

        self.bn2 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fcn2 = nn.Conv2d(25, 10, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))


        self.sig = nn.Sigmoid()


    def forward(self, x):
        Dictionary = creatRealDictionary(self.num_frame, self.Drr, self.Dtheta, self.gpu_id)
        imageFeature = self.modifiedResnet(x)  # T X 512 X 7 X 7

        if self.backbone == 'Resnet34' or 'Resnet18':
            convx = imageFeature

        else:
            convx = self.Conv2d(imageFeature)
            convx = self.bn1(convx)
            convx = self.relu(convx)

        x2 = self.layer2(convx)
        x2 = self.bn_l2(x2)
        x2 = self.relu(x2)

        x3 = self.layer3(x2)
        x3 = self.bn_l3(x3)

        x3 = self.relu(x3)

        x4 = self.layer4(x3)
        x4 = self.bn_l4(x4)
        feature = self.relu(x4)


        return feature, Dictionary, imageFeature


    def forward2(self, feature, alpha):

        x = feature.permute(1, 0, 2, 3)
        x = self.fcn1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fcn2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.view(1, -1)
        x = self.fc(x)
        out = self.sig(alpha*x)
        return out


class onlineUpdate(nn.Module):
    def __init__(self, FRA, PRE, T, Drr, Dtheta, gpu_id):
        super(onlineUpdate, self).__init__()
        self.gpu_id = gpu_id
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.numFrame = T
        self.K_FPN = keyframeProposalNet(numFrame=self.numFrame, Drr=self.Drr, Dtheta=self.Dtheta, gpu_id=gpu_id,
                                         backbone='Resnet18', config='jhmdb')
        self.FRA = FRA
        self.PRE = PRE


        self.relu = nn.LeakyReLU(inplace=True)

        self.layer0 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l0 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc = nn.Linear(1*64*3*3, 2)
    def get_keylist(self, x, alpha):
        feature, Dictionary, imgFeature = self.K_FPN.forward(x)
        indicator = self.K_FPN.forward2(feature, alpha)
        s = indicator[0, :]
        key_ind = (s > 0.995).nonzero().squeeze(1)
        key_list_tot = key_ind.cpu().numpy()

        key_list_FRA = list(key_list_tot[np.where(key_list_tot < self.FRA)[0]])  # input key list
        key_list = list(key_list_tot[np.where(key_list_tot < self.PRE+ self.FRA)[0]])
        keylist_to_pred = list(set(key_list) - set(key_list_FRA))


        Dict_key = Dictionary[key_list_FRA, :]
        feat_key = imgFeature[key_list_FRA, :]


        t, c, w, h = feat_key.shape
        feat_key = feat_key.reshape(1, t, c * w * h)
        sparseCode_key = fista(Dict_key, feat_key, 0.01, 100, self.gpu_id)

        return sparseCode_key, Dictionary, keylist_to_pred, key_list_FRA, key_list,imgFeature

    def forward(self, imgFeature, sparseCode_key, Dictionary, fraNum):
        gtImgFeature = imgFeature[fraNum]
        c, w, h = gtImgFeature.shape
        newDictionary = torch.cat((Dictionary[0:self.FRA], Dictionary[fraNum].unsqueeze(0)))
        newImgFeature = torch.matmul(newDictionary, sparseCode_key).reshape(newDictionary.shape[0], c, w, h)

        predImgFeature = newImgFeature[-1]
        combineFeature = torch.cat((gtImgFeature, predImgFeature)).unsqueeze(0)
        x = self.layer0(combineFeature)
        x = self.bn_l0(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.bn_l1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.bn_l2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.bn_l3(x)
        x = self.relu(x)

        x = x.view(1, -1)
        out = self.fc(x)
        return out

if __name__ == "__main__":

    gpu_id = 2
    alpha = 4 # step size for sigmoid
    N = 4 * 40
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    net = keyframeProposalNet(numFrame=40,Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id, backbone='Resnet34', config='Penn')
    net.cuda(gpu_id)

    X = torch.randn(1, 40, 3, 224, 224).cuda(gpu_id)

    for i in range(0, X.shape[0]):
        x = X[i]
        feature,dictionary,_ = net.forward(x)
        out = net.forward2(feature, alpha)
        print('check')


    print('done')
