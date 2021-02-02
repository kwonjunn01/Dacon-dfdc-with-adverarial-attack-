import os
import glob
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="PyTorch base for AI Hub Deepfake Detection Challenge")
parser.add_argument('--root_test', type=str,
                    default="/home/diml/ddrive/dataset/test/leaderboard")
parser.add_argument('--source_file', type=str,
                    default='sample_submission.csv')
parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names,
                    default='resnet18')
parser.add_argument('--check1', type=str, metavar='PATH',
                    default='19fgsm_b7_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--check2', type=str, metavar='PATH',
                    default='24fgsm_b7_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--check3', type=str, metavar='PATH',
                    default='25fgsm_b7_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--check4', type=str, metavar='PATH',
                    default='33fgsm_b7_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--check5', type=str, metavar='PATH',
                    default='40fgsm_b7_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--check6', type=str, metavar='PATH',
                    default='41fgsm_b7_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--check7', type=str, metavar='PATH',
                    default='46fgsm_b7_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use')


def loadmodel(model, check):
    if check:
        if os.path.isfile(check):
            print("=> loading checkpoint '{}'".format(check))
            if args.gpu is None:
                checkpoint = torch.load(check)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(check, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(check, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(check))

    cudnn.benchmark = True
    
    model = model.cuda().eval()
    return model
def average_weights(md1, md2, md3):
    pr_A = md1.state_dict()
    pr_B = md2.state_dict()
    pr_C = md3.state_dict()
    
    for key in pr_A:
        pr_A[key] = (pr_A[key] + pr_B[key] + pr_C[key])/3
    model = EfficientNet.from_name('efficientnet-b6').cuda()
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(pr_A)
    return model
def main(mode = 'sv'):
    global args, best_acc1
    args = parser.parse_args()

    if args.gpu is not None:
        print("Use GPU: {} for inference".format(args.gpu))

    # create submission csv file paralle with sample_submission.csv
    dir = os.path.dirname(args.source_file)
    save_csv = os.path.join(dir, 'ensemble_test_submission.csv')

    sc = open(save_csv, 'w')

    sc.write('path,y')
    sc.write('\n')

    # load model
    print("=> creating model '{}'".format(args.arch))
    model1 = EfficientNet.from_name('efficientnet-b6')
    model1.fc = nn.Linear(512, 2)
    model1 = loadmodel(model1, args.check1)
    
    model2 = EfficientNet.from_name('efficientnet-b6')
    model2.fc = nn.Linear(512, 2)
    model2 = loadmodel(model2, args.check2)

    model3 = EfficientNet.from_name('efficientnet-b6')
    model3.fc = nn.Linear(512, 2)
    model3 = loadmodel(model3, args.check3)

    model4 = EfficientNet.from_name('efficientnet-b6')
    model4.fc = nn.Linear(512, 2)
    model4 = loadmodel(model4, args.check4)

    model5 = EfficientNet.from_name('efficientnet-b6')
    model5.fc = nn.Linear(512, 2)
    model5 = loadmodel(model5, args.check5)
    
    model6 = EfficientNet.from_name('efficientnet-b6')
    model6.fc = nn.Linear(512, 2)
    model6 = loadmodel(model6, args.check6)

    model7 = EfficientNet.from_name('efficientnet-b6')
    model7.fc = nn.Linear(512, 2)
    model7 = loadmodel(model7, args.check7)


    
    # load pretrained weight
    if mode == 'swa':
       model_ensemble = average_weights(model1, model2, model3)
      


    # data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        normalize,
    ])

    # gather all images
    images = glob.glob(os.path.join(args.root_test, '*.jpg'))
    images.sort()

    # predict label
    m = nn.Softmax()
    if mode == 'sv':
        with torch.no_grad():
            for i, image_path in tqdm(enumerate(images)):
                image = Image.open(image_path)
                image = transform(image)
                image = torch.unsqueeze(image, dim=0)
    
                image = image.cuda(args.gpu, non_blocking=True)
    
                output1 = model1(image)
                output2 = model2(image)
                output3 = model3(image)
                output4 = model4(image)
                output5 = model5(image)
                output6 = model6(image)
                output7 = model7(image)
                output = m(output1)[0] + m(output2)[0] + m(output3)[0] + m(output4)[0] + m(output5)[0] + m(output6)[0] +m(output7)[0] # apply softmax
                
                image_tmpl = os.path.join('leaderboard', os.path.basename(image_path))
    
                # write to submission file
                if output[0] > output[1]:
                    saveline = image_tmpl + ',0'
                    sc.write(saveline)
                    sc.write('\n')
                else:
                    saveline = image_tmpl + ',1'
                    sc.write(saveline)
                    sc.write('\n')
    elif mode == 'swa':
        with torch.no_grad():
            for i, image_path in tqdm(enumerate(images)):
                image = Image.open(image_path)
                image = transform(image)
                image = torch.unsqueeze(image, dim=0)
    
                image = image.cuda(args.gpu, non_blocking=True)
    
                output1 = model_ensemble(image)
                output = m(output1)[0] # apply softmax
                
                image_tmpl = os.path.join('leaderboard', os.path.basename(image_path))
    
                # write to submission file
                if output[0] > output[1]:
                    saveline = image_tmpl + ',0'
                    sc.write(saveline)
                    sc.write('\n')
                else:
                    saveline = image_tmpl + ',1'
                    sc.write(saveline)
                    sc.write('\n')
    

    sc.close()


if __name__ == '__main__':
    main('sv')
