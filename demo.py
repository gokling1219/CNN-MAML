import torch
import numpy as np
from HSI_dataset import HSI_dataset
import scipy.stats
from torch.utils.data import DataLoader
import argparse
from meta import Meta
import time

torch.backends.cudnn.benchmark = True


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    # 5 layers
    num_filter = 128
    config = [

        ('conv2d', [num_filter, 103, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [num_filter]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [num_filter, num_filter, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [num_filter]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [num_filter, num_filter, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [num_filter]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [num_filter, num_filter, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [num_filter]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [num_filter, num_filter, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [num_filter]),


        ('flatten', []),
        ('linear', [args.n_way, num_filter])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    import h5py
    ########################## 训练数据集 ##########################
    f=h5py.File('./data/PU-16-16-103-train0-4-1000perclass.h5','r')
    data=f['data'][:]
    f.close()
    data=data.reshape(-1,16,16,103)
    data=data.transpose((0,3,1,2))
    dataset_train=HSI_dataset(data, 10000, args.n_way, args.k_spt, args.k_qry, 5, 1000)

    model_loss = []

    for epoch in range(1):
        db = DataLoader(dataset_train, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
        S = time.time()
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            #print(y_qry.shape) # torch.Size([4, 60])
            accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

            if (step+1) % 5 == 0:
                print('epoch: ', epoch + 1, '\tstep: ', step + 1, '\ttraining acc:', accs, 'model loss:', loss)
                model_loss.append(loss)
        f = open('./results/PU_train_time_update10.txt', 'w')
        f.write('train_time: ' + str(time.time() - S))
        f.close()

        '''
            if (step) % 50 == 0:
                print('epoch: ', epoch+1, '\tstep: ', step+1, '\ttraining acc:', accs, 'model loss:', loss)

            if (step) % 50 == 0:  # evaluation
                ########################## 微调数据集 ##########################
                f = h5py.File('./data/PU-16-16-103-test5-8-support25.h5', 'r')  # 48*200 = 9600, 28900
                x_spt = f['data'][:].reshape(-1, 16, 16, 103)
                x_spt = x_spt.transpose((0, 3, 1, 2))
                y_spt = f['label'][:] - 5
                # print(y_spt)
                f.close()

                class_corrects = np.asarray([0, 0, 0, 0])  # 记录用于测试的四个类别的预测正确数量
                preds = []
                for i in [1, 2, 3, 4, 5, 6]:
                    ########################## 测试数据集 ##########################
                    f = h5py.File('./data/PU-16-16-103-test5-8-file' + str(i) + '.h5', 'r')
                    x_qry = f['data'][:].reshape(-1, 16, 16, 103)
                    x_qry = x_qry.transpose((0, 3, 1, 2))
                    y_qry = f['label'][:] - 5
                    # print(np.unique(y_qry))
                    f.close()

                    x_spt_, y_spt_, x_qry_, y_qry_ = torch.from_numpy(x_spt).to(device), \
                                                     torch.LongTensor(y_spt).to(device), \
                                                     torch.from_numpy(x_qry).to(device), \
                                                     torch.LongTensor(y_qry).to(device)
                    pred = maml.finetunning(x_spt_, y_spt_, x_qry_, y_qry_)
                    #print(i)

                    # y_qry = np.asarray(y_qry.cpu())
                    if i == 1:
                        class_corrects[0] = sum(pred[0:1000] == y_qry[0:1000])
                        preds.append(pred[0:1000])  # 0
                        class_corrects[1] = sum(pred[1000:2000] == y_qry[1000:2000])
                        preds.append(pred[1000:2000])  # 1
                        class_corrects[2] = sum(pred[2000:3000] == y_qry[2000:3000])
                        preds.append(pred[2000:3000])  # 2
                        class_corrects[3] = sum(pred[3000:3947] == y_qry[3000:3947])
                        preds.append(pred[3000:3947])  # 3
                    elif i == 2:
                        class_corrects[0] += sum(pred[0:1000] == y_qry[0:1000])
                        preds.append(pred[0:1000])  # 4
                        class_corrects[1] += sum(pred[1000:1330] == y_qry[1000:1330])
                        preds.append(pred[1000:1330])  # 5
                        class_corrects[2] += sum(pred[2000:3000] == y_qry[2000:3000])
                        preds.append(pred[2000:3000])  # 6
                    elif i == 3:
                        class_corrects[0] += sum(pred[0:1000] == y_qry[0:1000])
                        preds.append(pred[0:1000])  # 7
                        class_corrects[2] += sum(pred[2000:3000] == y_qry[2000:3000])
                        preds.append(pred[2000:3000])  # 8
                    elif i == 4:
                        class_corrects[0] += sum(pred[0:1000] == y_qry[0:1000])
                        preds.append(pred[0:1000])  # 9
                        class_corrects[2] += sum(pred[2000:2682] == y_qry[2000:2682])
                        preds.append(pred[2000:2682])  # 10
                    elif i == 5:
                        class_corrects[0] += sum(pred[0:1000] == y_qry[0:1000])
                        preds.append(pred[0:1000])  # 11
                    elif i == 6:
                        class_corrects[0] += sum(pred[0:29] == y_qry[0:29])
                        preds.append(pred[0:29])  # 12


                # print(np.asarray(preds).shape) #7ge
                preds_flatten = []
                for i in [0, 4, 7, 9,11,12, 1, 5, 2, 6, 8,10, 3]:
                    preds_flatten.extend(preds[i])
                print(np.asarray(preds_flatten).shape)
                # preds = np.asarray(preds).flatten() # 展平
                test_acc = class_corrects.sum() / 10988.0
                print('Test acc:', test_acc)
                f = open('./results/PU_pred_epoch' + str(epoch + 1) + '_step' + str(step + 1) + '_acc' + str(
                    format(test_acc, '.4f')) + '.txt', 'w')
                for j in range(len(preds_flatten)):
                    f.write(str(preds_flatten[j]) + '\n')
                f.close()
        '''



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)

    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=19)

    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=100)

    args = argparser.parse_args()

    main()
