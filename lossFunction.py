import torch
import numpy as np


def minInverse_loss(m, Dictionary, input, T, lamd, gpu_id, config):
    ddt = torch.matmul(Dictionary, Dictionary.t())
    if len(m.shape) == 3 and len(input.shape) == 3:
        input = input.permute(2, 0, 1).squeeze(2)
        m = m[0, 0, :]
    elif len(m.shape) == 3 and len(input.shape) == 2:
        input = input.t()
        # m = m.squeeze(0)
        m = m[0,0,:]
    elif len(m.shape) == 2 and len(input.shape) ==3:
        # input = input.permute(2, 0, 1).squeeze(2)
        dim = input.shape[1] * input.shape[2]
        input = input.reshape(T, dim)
        m = m.squeeze(0)

    elif len(m.shape) == 2 and len(input.shape) == 4:
        # dim = input.shape[1] * input.shape[3]
        # input = input.squeeze(0).permute(1, 2, 0).reshape(T, dim)
        input = input.view(T, -1)
        m = m.squeeze(0)
    elif len(m.shape) == 2 and len(input.shape) == 5:
        dim = input.shape[2] * input.shape[3] * input.shape[4]
        input = input.squeeze(0).reshape(T, dim)
        m = m.squeeze(0)


    else:
        input = input.t()
        m = m.squeeze(0)

    sigma = 0.1 #0.01
    if config == 'penn':
        # lam1 = 1
        # lam2 = 5
        lam1 = lamd[0]
        lam2 = lamd[1]

    elif config == 'jhmdb':
        # lam1 = 2
        # lam2 = 4
        lam1 = lamd[0]
        lam2 = lamd[1]
    else:
        print('error model')



    lam3 = 0
    M = torch.diag(m)
    # M = m.t() * m
    I = torch.eye(T).cuda(gpu_id)
    A = torch.inverse(I+(1/sigma)*(torch.matmul(ddt, M)))

    l1 = torch.sum(m)
    l2 = torch.norm(torch.matmul(A, input), p='fro')
    l3 = torch.sum(torch.pow(torch.pow(m, 2)-m, 2))

    total_loss = lam1 * l1 + lam2 * l2 + lam3 * l3

    return total_loss, l1, l2, l3


