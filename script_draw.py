import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
import torch
import networks as nets
from data import generate_recog_data, generate_recog_data_batch
from plotting import plot_generalization, get_recog_positive_rates

d = 100             # input dim
T = 500             # length of dataset
N = 100

def t_test_data(model1, model2, num):
    r1, r2, acc1, acc2, tpr1, tpr2, fpr1, fpr2 = [], [], [], [], [], [], [], []
    gen_data = lambda R: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
    for i in range(num):
        r, acc, tpr, fpr = get_recog_positive_rates(model1, gen_data)
        r1.append(r)
        acc1.append(acc)
        tpr1.append(tpr)
        fpr1.append(fpr)
        r, acc, tpr, fpr = get_recog_positive_rates(model2, gen_data)
        r2.append(r)
        acc2.append(acc)
        tpr2.append(tpr)
        fpr2.append(fpr)
    return r1, r2, acc1, acc2, tpr1, tpr2, fpr1, fpr2

def test_plot(r1, r2, v1, v2):
    length = min(min(map(lambda x: len(x), r1)), min(map(lambda x: len(x), r2)))
    stat = []
    pvalue = []
    for i in range(length):
        v11 = [v[i] for v in v1]
        v22 = [v[i] for v in v2]
        s, p = ttest_rel(tpr11, tpr22)
        stat.append(s)
        pvalue.append(p)
    plt.plot(r1[0][:length], stat, marker='*', markevery=[pv<0.05 for pv in pvalue])
    plt.xlabel('TestR')
    plt.ylabel('T-statistic')
    plt.savefig(f'T-Test_{num}.png')

# hebb = nets.HebbNet([d,N,1])
heblstm = nets.nnLSTM([d,N,1])
# lstm = nets.rawLSTM([d, N, 1])
dhebb = nets.DoubleHebb([d, N, 1])
# hebb.load('/Users/chenjianhui/Code/Courses/Neural Computation/HebbNet[100,100,1]_forceAnti_train=inf6_T=500_4.95.pkl')
heblstm.load('/Users/chenjianhui/Code/Courses/Neural Computation/nnLSTM[100,100,1]_forceAnti_train=multiR6_T=500_4.9.pkl')
# lstm.load('/Users/chenjianhui/Code/Courses/Neural Computation/nnLSTM[100,100,1]_forceAnti_train=inf6_T=500_4.9_(2).pkl')
dhebb.load('/Users/chenjianhui/Code/Courses/Neural Computation/DoubleHebb[100,100,1]_forceAnti_train=multiR6_T=500_4.9.pkl')
# gen_data = lambda R: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
# testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(hebb, gen_data)
# ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate, label='HebbNet')
# testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(heblstm, gen_data)
# ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate, label='HebbLSTM')
# testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(lstm, gen_data, upToR=50)
# while len(testR) <= 3:
#     testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(lstm, gen_data)
# ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate, label='nnLSTM', ax=ax)
# testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(dhebb, gen_data)
# ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate, label='DoubleHebb', ax=ax)
num = 20
r1, r2, acc1, acc2, tpr1, tpr2, fpr1, fpr2 = t_test_data(heblstm, dhebb, num)
test_plot(r1, r2, tpr1, tpr2)
# plt.savefig('Hebbnet vs HebbLSTM.png')

fig= plt.figure()
ax1 = fig.add_subplot(111)
ax1 = plt.gca()
for net, name in zip([heblstm, dhebb], ['HebbLSTM', 'DoubleHebb']):

# path = '/Users/chenjianhui/Code/Courses/Neural Computation/final/HEBBFF_pro/models'
# filenames=os.listdir(path)
# for filename in filenames:
#     net = nets.nnLSTM([d,N,1])
#     net.load(path+'/'+filename)
#     #plot generalization
#     gen_data = lambda R: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
#     testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_data)
#     ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate)
#     plt.savefig(filename+'_test.png')
    
    
#     # train loss acc
#     state = torch.load(path+'/'+filename)
    loss = net.hist['train_loss']
    acc = net.hist['train_acc']
    window_size = 10
    loss = np.convolve(loss, np.ones(window_size) / window_size, mode='valid')
    x = np.arange(len(loss))
    plot1 = ax1.plot(range(0, len(x)), loss, label=f'loss-{name}',linewidth=1)
    # acc = np.convolve(acc, np.ones(window_size) / window_size, mode='valid')
    # x = np.arange(len(acc))
    # plot1 = ax1.plot(range(0, len(x)), acc, label=f'acc-{name}',linewidth=1)
    
fig.legend()
fig.savefig(filename+'_train.png')
    