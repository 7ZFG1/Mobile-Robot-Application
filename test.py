import os
import torch
import sys
import argparse
import numpy as np
import imageio
import cv2
import timeit
from model import td4_psp18, td2_psp50, pspnet
from dataloader_edited import cityscapesLoader

torch.backends.cudnn.benchmark = True
torch.cuda.cudnn_enabled = True

class Segmenter:
    def __init__(self,args):
        self.model = None
        self.device = None
        self.args = args
        self.i = 0
        self.initialize()
        self.data_load = cityscapesLoader()
    def initialize(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.model=='td4-psp18':
            self.path_num = 4
            self.model = td4_psp18.td4_psp18(nclass=19,path_num=self.path_num,model_path=self.args._td4_psp18_path)

        elif self.args.model=='td2-psp50':
            self.path_num = 2
            self.model = td2_psp50.td2_psp50(nclass=19,path_num=self.path_num,model_path=self.args._td2_psp50_path)

        elif self.args.model=='psp101':
            self.path_num = 1
            self.model = pspnet.pspnet(nclass=19,model_path=self.args._psp101_path)
        
        self.model.eval()
        self.model.to(self.device)

    def tester(self,image):
        image = self.data_load.load_frames(img=image)
        image = image.to(self.device)

        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        output = self.model(image, pos_id=0%self.path_num)

        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time

        pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)

        pred = pred.astype(np.int8)
        pred = cv2.resize(pred, (self.ori_size[0]//4,self.ori_size[1]//4), interpolation=cv2.INTER_NEAREST)
        decoded = self.data_load.decode_segmap(pred)

        save_dir = os.path.join(self.args.output_path,"vid1")
        res_path = os.path.join(save_dir,str(self.i) + ".png")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        imageio.imwrite(res_path, decoded.astype(np.uint8))
        cv2.namedWindow("Image")
        cv2.imshow("Image", decoded.astype(np.uint8))
        cv2.waitKey(1)

        print(" Frame {0:2d}   RunningTime/Latency={1:3.5f} s".format(self.i + 1,  elapsed_time))


def test(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model=='td4-psp18':
        path_num = 4
        vid_seq = cityscapesLoader(img_path=args.img_path,in_size=(769,1537))
        vid_seq.load_frames()
        model = td4_psp18.td4_psp18(nclass=19,path_num=path_num,model_path=args._td4_psp18_path)

    elif args.model=='td2-psp50':
        path_num = 2
        vid_seq = cityscapesLoader(img_path=args.img_path,in_size=(769,1537))
        vid_seq.load_frames()
        model = td2_psp50.td2_psp50(nclass=19,path_num=path_num,model_path=args._td2_psp50_path)

    elif args.model=='psp101':
        path_num = 1
        vid_seq = cityscapesLoader(img_path=args.img_path,in_size=(769,1537))
        vid_seq.load_frames()
        model = pspnet.pspnet(nclass=19,model_path=args._psp101_path)
    
    model.eval()
    model.to(device)

    timer = 0.0
    print("a1")
    with torch.no_grad():
        print(len(vid_seq.data))
        for i, (image, img_name, folder, ori_size) in enumerate(vid_seq.data):
            print("a3")

            image = image.to(device)

            torch.cuda.synchronize()
            start_time = timeit.default_timer()

            output = model(image, pos_id=i%path_num)

            torch.cuda.synchronize()
            elapsed_time = timeit.default_timer() - start_time

            if i > 5:
                timer += elapsed_time

            pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)

            pred = pred.astype(np.int8)
            pred = cv2.resize(pred, (ori_size[0]//4,ori_size[1]//4), interpolation=cv2.INTER_NEAREST)
            decoded = vid_seq.decode_segmap(pred)

            save_dir = os.path.join(args.output_path,folder)
            res_path = os.path.join(save_dir,img_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            imageio.imwrite(res_path, decoded.astype(np.uint8))
            cv2.namedWindow("Image")
            cv2.imshow("Image", decoded.astype(np.uint8))
            cv2.waitKey(1)

            print(" Frame {0:2d}   RunningTime/Latency={1:3.5f} s".format(i + 1,  elapsed_time))

    print("---------------------")
    print(" Model: {0:s}".format(args.model))
    #print(" Average  RunningTime/Latency={0:3.5f} s".format(timer/(i-5)))
    print("---------------------")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument("--img_path",nargs="?",type=str,
            default="./data/vid1",help="Path_to_Frame",)

    parser.add_argument("--output_path",nargs="?",type=str,
            default="./output/",help="Path_to_Save",)

    parser.add_argument("--_td4_psp18_path",nargs="?",type=str,
            default="./checkpoint/td4-psp18.pkl",help="Path_to_PSP_Model",)

    parser.add_argument("--_td2_psp50_path",nargs="?",type=str,
            default="./checkpoint/td2-psp50.pkl",help="Path_to_PSP_Model",)

    parser.add_argument("--_psp101_path",nargs="?",type=str,
            default="./checkpoint/psp101.pkl",help="Path_to_PSP_Model",)

    parser.add_argument("--gpu",nargs="?",type=str,default='0',help="gpu_id",)

    parser.add_argument("--model",nargs="?",type=str,default='td4-psp18',
            help="model in [td4-psp18, td2_psp50, psp101]",)

    args = parser.parse_args()


    test(args)
