import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import torch
from torch import nn
from model import VGG11, CGCVGG11, Learn_CGCVGG11
from torch.optim import lr_scheduler
import argparse
from dataset import cifar10_DVS, DVS_Gesture, Ncaltech101
import torch.nn.functional as F
from spikingjelly.activation_based import encoding
from spikingjelly.clock_driven import functional
from tqdm import tqdm
from pathlib import Path
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch import amp
from neuron import Neuron
import random


seed_value = 2002      # 设定随机数种子
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
# torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
torch.backends.cudnn.deterministic = True



def parseargs():
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--seed_value', type=int, default=seed_value)
    parser.add_argument('--data_dir', type=str, default='../CIFAR10_DVS', help='dataset path')
    parser.add_argument('-encode_method', type=str, default='method3', help='encoding method')
    parser.add_argument('-model', default='Learn_CGCVGG11', type=str, help='use which SNN model')
    parser.add_argument('-segments', type=int, default=4, help='number of segments')
    parser.add_argument('-device', default=torch.device('cuda:0'), type=str, help='device')
    parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('-T_data', default=8, type=int, help='Timestep of dataset')
    parser.add_argument('-tau', default=2.0, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-resume', default=False, type=bool,  help='resume from the checkpoint path')
    parser.add_argument('-neuron', default=Neuron, help='type of neuron')
    parser.add_argument('-outdir', type=str, default='./logs/train', help='root dir for saving logs and checkpoint')
    parser.add_argument('-amp', default=True, type=bool, help='automatic mixed precision training')
    parser.add_argument('-opt', default='SGD', type=str, help='use which optimizer. SGD or AdamW')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=10, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.75, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-cutupmix_auto', default=False, help='cutupmix autoaugmentation for cifar and tinyimagenet')
    return parser


def increment_path(path, exist_ok=False, sep='', mkdir=True):
    # 创建增量目录
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir_ = path if path.suffix == '' else path.parent  # directory
    if not dir_.exists() and mkdir:
        dir_.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def procession_DVS(model, input, T, T_data):
    output = []
    A = np.random.choice(T_data, T, replace=False)
    A.sort()
    out_put = 0
    for t in range(T):
        img = input[A[t]].float().to('cuda')
        out = model(img)
        output.append(out)
        out_put += out
    output = torch.cat(output, dim=0)
    out_put /= T
    return output, out_put


def procession_DVS_test(model, input, T_data):
    output = []
    out_put = 0
    for t in range(T_data):
        img = input[t].float().to('cuda')
        out = model(img)
        output.append(out)
        out_put += out
    output = torch.cat(output, dim=0)
    out_put /= T_data
    return output, out_put


def main():
    args = parseargs().parse_args()

    #####################################
    # data and model
    # method0:分极性积分事件数量，channel=2
    # method1:时间表面法和事件数量积分，归一化后拼接，channel=4
    # method2:分段时间表面法，PCA降维；并对事件数量积分，归一化后拼接，channel=4
    #####################################
    data_dir = args.data_dir
    _, _, train_data_loader, test_data_loader = cifar10_DVS(b=args.b, num_workers=4, data_dir=data_dir, T=args.T_data, method=args.encode_method, segments=args.segments)
    if args.encode_method == 'method0':
        if args.model == 'VGG11':
            model = VGG11(args.neuron, c_in=2, numclass=10)
    else:
        if args.model == 'VGG11':
            model = VGG11(args.neuron, c_in=4, numclass=10)
        elif args.model == 'CGCVGG11':
            model = CGCVGG11(args.neuron, c_in=4, numclass=10)
        elif args.model == 'Learn_CGCVGG11':
            model = Learn_CGCVGG11(args.neuron, c_in=10, numclass=10)
    model.to(args.device)


    #####################################
    # optimizer and loss function
    #####################################
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    loss_fn = nn.CrossEntropyLoss()

    ##########################################################
    # resume
    ##########################################################
    start_epoch = 0
    max_acc = 0
    if args.resume:
        print('resuming...')
        checkpoint = torch.load('logs_0/train9/checkpoint_latest.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_acc = checkpoint['max_acc']
        print('start epoch:', start_epoch, ', max test acc:', max_acc)

    #####################################
    # output
    #####################################
    out_path = increment_path(args.outdir)
    argsdict = args.__dict__
    with open(out_path / "args.txt", 'w') as fire:
        fire.writelines('-----------start-----------' + '\n')
        for eacharg, value in argsdict.items():
            fire.writelines(eacharg + ':' + str(value) + '\n')
        fire.writelines('------------end------------')
        fire.writelines('\n' + str(model))

    #####################################
    # train and test
    #####################################
    trainloss = np.zeros(args.epochs)
    trainacc = np.zeros(args.epochs)
    testloss = np.zeros(args.epochs)
    testacc = np.zeros(args.epochs)
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        trl, tra = train(args, model, train_data_loader, loss_fn, optimizer, scheduler, scaler, out_path, epoch+1)
        trainloss[epoch] = trl
        trainacc[epoch] = tra
        tel, tea = test(args, model, test_data_loader, loss_fn, out_path)
        testloss[epoch] = tel
        testacc[epoch] = tea
        if tea > max_acc:
            max_acc = tea
            torch.save(model.state_dict(), out_path / 'best_model.pth')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'max_acc': max_acc
        }
        torch.save(checkpoint, os.path.join(out_path, 'checkpoint_latest.pth'))
        end_time = time.time()
        time_cost = end_time - start_time
        with open(out_path / "process.txt", 'a') as fire:
            fire.write('time_cost:' + str(time_cost) + '\n')
            fire.writelines('---------------------------------' + '\n')
        print('max_test_acc:' + str(max_acc))
        print('-----------------------------------')

    with open(out_path / "process.txt", 'a') as fire:
        fire.write('Max_test_acc: ' + str(max_acc) + '\n')

    #####################################
    # 画图
    #####################################
    plt.figure(1)
    plt.plot(trainloss)
    # plt.title('Training Loss')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.savefig(out_path / 'Training Loss.png')
    plt.figure(2)
    plt.plot(trainacc)
    # plt.title('Training Accuracy')
    plt.ylabel('train_accuracy')
    plt.xlabel('epochs')
    plt.savefig(out_path / 'Training Accuracy.png')
    plt.figure(3)
    plt.plot(testacc)
    # plt.title('Test Accuracy')
    plt.ylabel('test_accuracy')
    plt.xlabel('epochs')
    plt.savefig(out_path / 'Test Accuracy.png')
    plt.figure(4)
    plt.plot(testloss)
    # plt.title('Test Loss')
    plt.ylabel('test_loss')
    plt.xlabel('epochs')
    plt.savefig(out_path / 'Test Loss.png')
    print('Done!')


def train(args, model, train_data_loader, loss_fn, optimizer, scheduler, scaler, outpath, ep):
    model.train()
    loop = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for step,(img, label) in loop:
        optimizer.zero_grad()
        label = label.to(args.device)
        label_cpt = torch.cat([label for _ in range(args.T)], 0)
        if args.amp:
            with amp.autocast('cuda'):
                output, out_put = procession_DVS(model, img, args.T, args.T_data)
                loss = loss_fn(output, label_cpt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            raise NotImplementedError(
                'We do not implement mixed precision training for BPTT. Please implement it by yourself.')

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_put.argmax(1) == label).float().sum().item()
        functional.reset_net(model)

        loop.set_description(f'Epoch [{ep}/{args.epochs}]')
        loop.set_postfix(loss=train_loss / (train_samples), acc=float(train_acc) / float(train_samples))

    train_loss /= train_samples
    train_acc /= train_samples

    with open(outpath / "process.txt", 'a') as fire:
        fire.write('Epoch: ' + str(ep) + '\n')
        fire.write('train_loss: ' + str(train_loss) + ' // ')
        fire.write('train_acc: ' + str(train_acc) + '\n')

    scheduler.step()
    return train_loss, train_acc


def test(args, model, test_data_loader, loss_fn, outpath):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            label = label.to(args.device)

            output, out_put = procession_DVS_test(model, img, args.T_data)
            loss = loss_fn(out_put, label)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_put.argmax(1) == label).float().sum().item()
            functional.reset_net(model)

        test_loss /= test_samples
        test_acc /= test_samples
        with open(outpath / "process.txt", 'a') as fire:
            fire.write('test_loss: ' + str(test_loss) + ' // ')
            fire.write('test_acc: ' + str(test_acc) + '\n')

    print('test_loss:' + str(test_loss) + ' test_acc:' + str(test_acc))
    return test_loss, test_acc


if __name__ == '__main__':
    main()





