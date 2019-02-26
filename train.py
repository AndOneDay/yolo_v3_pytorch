from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from dataset.cocodataset import *

import argparse
import yaml
import random

import torch
from torch.autograd import Variable
import torch.optim as optim
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3-tiny.yaml',#default='config/yolov3_default.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    parser.add_argument('--snap_name', type=str,
                        required=True, help='snapshot name')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    parser.add_argument('--eval_interval', type=int,
                            default=1000, help='interval between evaluations')
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/workspace/mnt/bucket/pulp-det/yolo-output',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard', help='tensorboard path for logging', type=str, default=None)
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    args = parse_args()
    print("Setting Arguments.. : ", args)
    cuda = torch.cuda.is_available() and args.use_cuda

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    print("successfully loaded config file: ", cfg)

    lr = cfg['TRAIN']['LR']
    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Learning rate setup
    base_lr = lr

    # Initiate model
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)
    if args.pretrained:
        if cfg['MODEL']['TYPE'] == 'YOLOv3':
            # self.module_list = create_yolov3_modules(config_model, ignore_thre)
            print('load yolov3 pretrained model')
            model.load_pretrained_weights('weights/darknet53.conv.74', cutoff=75)
        elif cfg['MODEL']['TYPE'].lower() == 'yolov3-tiny':
            print('load yolov3 pretrained model')
            model.load_pretrained_weights('weights/yolov3-tiny.conv.15', cutoff=16)
        else:
            print('no this type pretrained model')

    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        #parse_yolo_weights(model, args.weights_path)
        model.load_weights(args.weights_path)
    elif args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        print("using cuda") 
        model = model.cuda()
    #model.save_weights('../darknet/torch-yolo-tiny.weights')
    if args.tfboard:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard)
    #model = torch.nn.DataParallel(model)

    model.train()

    imgsize = cfg['TRAIN']['IMGSIZE']
    dataset = COCODataset(model_type=cfg['MODEL']['TYPE'],
                  data_dir='COCO/',
                  img_size=imgsize,
                  augmentation=cfg['AUGMENTATION'],
                  debug=args.debug)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
    dataiterator = iter(dataloader)

    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir='COCO/',
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'])

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # optimizer setup
    # set weight decay only on conv.weight
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay * batch_size * subdivision}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]
    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size * subdivision)

    iter_state = 0

    if args.checkpoint:
        if 'optimizer_state_dict' in state.keys():
            optimizer.load_state_dict(state['optimizer_state_dict'])
            iter_state = state['iter'] + 1

    # TODO: replace the following scheduler with the PyTorch's official one

    tmp_lr = base_lr

    def set_lr(tmp_lr):
        #print('set lr:', tmp_lr / batch_size / subdivision)
        for param_group in optimizer.param_groups:
            param_group['lr'] = tmp_lr / batch_size / subdivision
            #param_group['lr'] = tmp_lr / subdivision

    # start training loop
    for iter_i in range(iter_state, iter_size + 1):
        # COCO evaluation
        if iter_i % args.eval_interval == 0 and iter_i > 0:
            ap50_95, ap50 = evaluator.evaluate(model)
            model.train()
            if args.tfboard:
                tblogger.add_scalar('val/COCOAP50', ap50, iter_i)
                tblogger.add_scalar('val/COCOAP50_95', ap50_95, iter_i)

        # learning rate scheduling
        if iter_i < burn_in:
            tmp_lr = base_lr * pow(iter_i / burn_in, 4)
            set_lr(tmp_lr)
        elif iter_i == burn_in:
            tmp_lr = base_lr
            set_lr(tmp_lr)
        elif iter_i in steps:
            tmp_lr = tmp_lr * 0.1
            set_lr(tmp_lr)

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            loss = model(imgs, targets)
            loss.backward()

        optimizer.step()

        if iter_i % 10 == 0:
            # logging
            total_loss = model.loss_dict['xy'] + model.loss_dict['wh'] + model.loss_dict['conf'] + model.loss_dict['cls']
            print('[Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, l2 %f, total %f, imgsize %d]'
                  % (iter_i, iter_size, tmp_lr,
                     model.loss_dict['xy'], model.loss_dict['wh'],
                     model.loss_dict['conf'], model.loss_dict['cls'], 
                     model.loss_dict['l2'], total_loss , imgsize),
                  flush=True)

            if args.tfboard:
                tblogger.add_scalar('train/total_loss', model.loss_dict['l2'], iter_i)

            # random resizing
            if random_resize:
                imgsize = (random.randint(0, 9) % 10 + 10) * 32
                dataset.img_shape = (imgsize, imgsize)
                dataset.img_size = imgsize
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
                dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            try:
                args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.snap_name + '_' + date)
                #print(args.checkpoint_dir)
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                time.sleep(5)
                torch.save({'iter': iter_i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            },
                            os.path.join(args.checkpoint_dir, "snapshot"+str(iter_i)+".pth"))
                time.sleep(5)
                model.save_weights(os.path.join(args.checkpoint_dir, "snapshot"+str(iter_i)+".weights"))
                time.sleep(5)
            except Exception as e:
                print(e)

    if args.tfboard:
        tblogger.close()


if __name__ == '__main__':
    main()
