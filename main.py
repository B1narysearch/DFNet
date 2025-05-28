from utils.data import *
from utils.metric import *
from utils.save_imgs import *
from argparse import ArgumentParser
import torch.utils.data as Data
from utils.loss import *
from utils.BDLoss import *
from model.UNet.UNet import *
#from .model.LRNet.LRNet import *
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time
import torch


os.environ['CUDA_VISIBLE_DEVICES']="0"



def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset', type=str, default='NUDT')
    parser.add_argument('--dataset-dir', type=str, default='./dataset/NUDT')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default='./weight/NUDT-84.59/weight.pkl')
    parser.add_argument('--model', type=str, default='UNet')
    parser.add_argument('--log', type=str, default='./log/')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

    
        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        if args.dataset == "IRSTD-1k":
            trainset = IRSTD_Dataset(args, mode='train')
            valset = IRSTD_Dataset(args, mode='val')
        elif args.dataset == "NUDT":
            trainset = NUDT_Dataset(args, mode='train')
            valset = NUDT_Dataset(args, mode='val')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False)

        device = torch.device('cuda')
        self.device = device

        model = UNet(3)
        model.to(device)
        self.model = model
        self.loss_fun = SoftIoULoss()
        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        # if args.multi_gpus:
        #     if torch.cuda.device_count() > 1:
        #         print('use '+str(torch.cuda.device_count())+' gpus')
        #         model = nn.DataParallel(model, device_ids=[0, 1])
        # model.to(device)
        # self.model = model


        self.down = nn.MaxPool2d(2, 2)
        self.PD_FA = PD_FA(1, 10, args.base_size)
        # self.PD_FA = PD_FA()
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = './weight/DFNet-2024-08-27-08-37-39'
                checkpoint = torch.load(check_folder+'/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']+1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                self.save_folder = './weight/DFNet-%s'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                if not osp.exists(self.save_folder):
                    os.mkdir(self.save_folder)
        if args.mode=='test':
          
            weight = torch.load(args.weight_path)
            self.model.load_state_dict(weight)
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1
        

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        for i, (data, mask) in enumerate(tbar):
            data = data.to(self.device)
            labels = mask.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(data)
            loss = self.loss_fun(pred, labels)

            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):


                data = data.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(data)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()
            # print("----------------" + str(PD[0]) + "------------------")
            
            if self.mode == 'train':
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                                epoch, self.best_iou, PD[0], FA[0] * 1000000))

                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                torch.save(all_states, self.save_folder+'/checkpoint.pkl')
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


def FeatureMap2Heatmap(x, feature1, feature2, feature3, map_x, i):
    ## initial images
    feature_first_frame = x[0, :, :, :].cpu()  ## the middle frame

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log + '/' + str(i) +'_x_visual.jpg' )
    plt.close()

    ## first feature
    feature_first_frame = feature1[0, :, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log + '/' + '_x_Block1_visual.jpg')
    plt.close()

    ## second feature
    feature_first_frame = feature2[0, :, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log + '/' + '_x_Block2_visual.jpg')
    plt.close()

    ## third feature
    feature_first_frame = feature3[0, :, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log + '/' + '_x_Block3_visual.jpg')
    plt.close()

    ## third feature
    heatmap2 = torch.pow(map_x[0, :, :], 2)  ## the middle frame

    heatmap2 = heatmap2.data.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap2)
    plt.colorbar()
    plt.savefig(args.log + '/' + '_x_DepthMap_visual.jpg')
    plt.close()

         
if __name__ == '__main__':
    args = parse_args()
    random.seed(42)
    trainer = Trainer(args)

    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
 