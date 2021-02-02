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
                    default='efficientnet-b6')
parser.add_argument('--resume', type=str, metavar='PATH',
                    default='/home/diml/ddrive/weights/20fgsm_b6_checkpoint.pth.tar',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use')


def main():
    global args, best_acc1
    args = parser.parse_args()

    if args.gpu is not None:
        print("Use GPU: {} for inference".format(args.gpu))

    # create submission csv file paralle with sample_submission.csv
    dir = os.path.dirname(args.source_file)
    save_csv = os.path.join(dir, '20test_submission.csv')

    sc = open(save_csv, 'w')

    sc.write('path,y')
    sc.write('\n')

    # load model
    print("=> creating model '{}'".format(args.arch))
    model = EfficientNet.from_name(args.arch)
    # modify the number of output nodes



    
    # load pretrained weight
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    model = model.eval()

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
    model.cuda()
    with torch.no_grad():
        for i, image_path in tqdm(enumerate(images)):
            image = Image.open(image_path)
            image = transform(image)
            image = torch.unsqueeze(image, dim=0)

            image = image.cuda(args.gpu, non_blocking=True)

            output = model(image)
            output = m(output)[0]  # apply softmax

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
    main()
