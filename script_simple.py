import os
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
import networks as nets
from data import generate_recog_data, generate_recog_data_batch
from plotting import plot_generalization, get_recog_positive_rates

parser= argparse.ArgumentParser()

parser.add_argument('--model_name', default='HebbNet', choices=['HebbNet', 'nnLSTM', 'HebbLSTM', 'DoubleHebb'])
parser.add_argument('--input_dim', default=100, type=int)
parser.add_argument('--hidden_dim', default=100, type=int)
parser.add_argument('--train_mode', default='dat', choices=['dat', 'inf', 'curr', 'multiR'])
parser.add_argument('--R', default=3, type=int)
parser.add_argument('--T', default=500, type=int)
parser.add_argument('--threshold', default=4.9, type=float)

args = parser.parse_args()

#choose parameters
netType = args.model_name      # HebbFF or LSTM
d = args.input_dim             # input dim
N = args.hidden_dim            # hidden dim
force = 'Anti'                 # ensure either Hebbian or anti-Hebbian plasticity
trainMode = args.train_mode    # train on single dataset or infinite data
R = args.R                     # delay interval
T = args.T                     # length of dataset
save = True
os.environ['EARLY_STOP'] = str(args.threshold)

#initialize net
if netType == 'nnLSTM':
    net = nets.VanillaRNN([d,N,1])
elif netType == 'HebbLSTM':
    net = nets.nnLSTM([d, N, 1])
elif netType == 'HebbNet':
    net = nets.HebbNet([d,N,1])
    if force == 'Hebb':
        net.forceHebb = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item()) #need to re-init for this to work
    elif force == 'Anti':
        net.forceAnti = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item())
    elif force is not None:
        raise ValueError
elif netType == 'DoubleHebb':
    net = nets.DoubleHebb([d,N,1])
    if force == 'Hebb':
        net.forceHebb = torch.tensor(True)
        net.init_hebb(eta=net.heb1.eta.item(), lam=net.heb1.lam.item()) #need to re-init for this to work
    elif force == 'Anti':
        net.forceAnti = torch.tensor(True)
        net.init_hebb(eta=net.heb1.eta.item(), lam=net.heb1.lam.item())
    elif force is not None:
        raise ValueError
else:
    raise ValueError

#train
if trainMode == 'dat':
    trainData = generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
    validBatch = generate_recog_data(T=T, d=d, R=R, P=0.5, multiRep=False).tensors
    net.fit('dataset', epochs=float('inf'), trainData=trainData,
            validBatch=validBatch, earlyStop=False)
elif trainMode == 'inf':
    gen_data = lambda: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
    net.fit('infinite', gen_data)
elif trainMode == 'curr':
    gen_data = lambda R: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
    net.fit('curriculum', gen_data, iters=10000)
else:
    gen_data = lambda R: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
    net.fit('multiR', gen_data, itersToQuit=2000, increment=lambda Rlo, Rhi: (Rlo, int(np.ceil(Rhi*1.3))))

    

#optional save
if save:
    fname = '{}[{},{},1]_{}train={}{}_{}_{}.pkl'.format(
                netType, d, N, 'force{}_'.format(force) if force else '',
                trainMode, R, 'T={}'.format(T) if trainMode != 'cur' else '', th
                )
    net.save(fname)

#plot generalization
gen_data = lambda R: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_data)
ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate)
figname = '{}[{},{},1]_{}train={}{}_{}_{}.png'.format(
        netType, d, N, 'force{}_'.format(force) if force else '',
        trainMode, R, 'T={}'.format(T) if trainMode != 'cur' else '', th
        )
plt.savefig(figname)