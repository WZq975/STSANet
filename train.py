import torch
from torch.utils import data
import os
from STSA_model import STSANet
from dataloader import DHF1KDataset, UCFDataset, Hollywood, DIEM
from tqdm import tqdm
from loss import cc, similarity, nss, kldiv, auc_judd
import torch.nn as nn
import cv2
import argparse


def loss_function(prediction, label):
    loss = kldiv(prediction, label) - cc(prediction, label)

    return loss


def train(model, optimzer, dataloader, device, epoch):
    model.train()

    loss_tra = 0.0
    kld_tra = 0.0
    pro_bar = tqdm(dataloader)
    for img_clip, label, fixation in pro_bar:
        flag = 0
        for i in range(fixation.shape[0]):
            if fixation[i, :, :].max() == torch.tensor([0]):
                flag = 1
                break
        if flag == 1:
            continue
        img_clip = img_clip.to(device)
        label = label.to(device)
        # fixation = fixation.to(device)

        optimzer.zero_grad()

        prediction = model(img_clip)

        assert prediction.size() == label.size()
        # assert prediction.size() == fixation.size()
        loss = loss_function(prediction, label)
        kld = kldiv(prediction, label)

        loss.backward()
        optimzer.step()
        loss_tra += loss.item()
        kld_tra += kld.item()

    len_dataloader = len(dataloader)
    tqdm.write('Epoch: {:d} | loss_tra_avg: {:.5f} KLD: {:.5f} '.format(epoch, loss_tra / len_dataloader, kld_tra / len_dataloader))


def validate(model, dataloader, device, epoch):
    model.eval()

    loss_val = 0.0
    SIM = 0.0
    CC = 0.0
    NSS = 0.0
    KLD = 0.0
    AUC = 0.0
    pro_bar = tqdm(dataloader)
    cnt = 0.0
    for img_clip, label, fixation in pro_bar:
        if fixation.max() == torch.tensor([0]):
            continue
        img_clip = img_clip.to(device)
        label = label.to(device)
        fixation = fixation.to(device)

        prediction = model(img_clip)

        prediction = prediction.cpu().squeeze(0).clone().numpy()
        prediction = cv2.resize(prediction, (label.shape[2], label.shape[1]))
        prediction = torch.FloatTensor(prediction).unsqueeze(0).to(device)

        assert prediction.size() == label.size()

        # print(fixation.shape)
        # print(label.shape)
        # print(prediction.shape)
        loss = loss_function(prediction, label)
        score_cc = cc(prediction, label)
        # print(video_name, start, prediction.shape, label.shape, score_cc)
        score_sim = similarity(prediction, label)
        score_nss = nss(prediction, fixation)
        score_kld = kldiv(prediction, label)
        score_auc = auc_judd(prediction, fixation)

        loss_val += loss.item()
        CC += score_cc.item()
        # print(CC)
        SIM += score_sim.item()
        NSS += score_nss.item()
        KLD += score_kld.item()
        AUC += score_auc.item()

        cnt += 1
        # if cnt % 500 == 0:
        #     print(CC / cnt)
    len_dataloader = cnt

    tqdm.write('Epoch: {:d} length of dataloader: {:d} | loss_val_avg:{:.5f} SIM:{:.4f} CC:{:.4f}'
               'NSS: {:.4f} KLD: {:.4f} AUC: {:.4f}'.format(epoch, int(len_dataloader), loss_val / len_dataloader,
                                                SIM / len_dataloader, CC / len_dataloader,
                                                NSS / len_dataloader, KLD / len_dataloader, AUC / len_dataloader))
    return loss_val / len(dataloader), NSS / len_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, help='initial learning rate')
    parser.add_argument('--ds', default='DHF1K', help='dataset (DHF1K, Hollywood-2, UCF, DIEM)')
    parser.add_argument('--pd', default='./DHF1K', help='path of dataset')
    args = parser.parse_args()

    if args.ds == 'DHF1K':
        DHF1K_frame_path = os.path.join(args.pd, 'frames')
        DHF1K_label_path = os.path.join(args.pd, 'annotation')
        dataset_tra = DHF1KDataset(DHF1K_frame_path, DHF1K_label_path, len_clip=32, loader_mode='tra')
        dataset_val = DHF1KDataset(DHF1K_frame_path, DHF1K_label_path, len_clip=32, loader_mode='val')
        dataloader_tra = data.DataLoader(dataset_tra, batch_size=3, shuffle=True)
        dataloader_val = data.DataLoader(dataset_val, batch_size=1, shuffle=False)
    elif args.ds == 'UCF':
        ucf_frame_path = args.pd
        dataset_tra = UCFDataset(ucf_frame_path, len_clip=32, loader_mode='tra')
        dataset_val = UCFDataset(ucf_frame_path, len_clip=32, loader_mode='test')
        dataloader_tra = data.DataLoader(dataset_tra, batch_size=3, shuffle=True)
        dataloader_val = data.DataLoader(dataset_val, batch_size=1, shuffle=False)
    elif args.ds == 'Hollywood-2':
        Hollywood_frame_path1 = os.path.join(args.pd, 'training')
        Hollywood_frame_path2 = os.path.join(args.pd, 'testing')
        dataset_tra = Hollywood(Hollywood_frame_path1, len_clip=32, loader_mode='tra')
        dataset_val = Hollywood(Hollywood_frame_path2, len_clip=32, loader_mode='test')
        dataloader_tra = data.DataLoader(dataset_tra, batch_size=3, shuffle=True)
        dataloader_val = data.DataLoader(dataset_val, batch_size=1, shuffle=False)
    elif args.ds == 'DIEM':
        DIEM_frame_path = args.pd
        dataset_tra = DIEM(DIEM_frame_path, len_clip=32, loader_mode='tra')
        dataset_val = DIEM(DIEM_frame_path, len_clip=32, loader_mode='val')
        dataloader_tra = data.DataLoader(dataset_tra, batch_size=3, shuffle=True)
        dataloader_val = data.DataLoader(dataset_val, batch_size=1, shuffle=False)
    else:
        raise Exception('Reenter dataset name.')

    model = STSANet()

    # load S3D backbone weights
    model_dict = model.state_dict()
    pretrained_dict = torch.load('./S3D_backbone.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("On", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    LR = args.lr
    epoch_size = 12
    for epoch in range(80):
        print('current learning rate: ', LR)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        if epoch % epoch_size == 0 and epoch != 0:
            LR = LR * 0.1
            print('current learning rate: ', LR)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
            if epoch != 0:
                model_dict = model.state_dict()
                pretrained_dict = torch.load(pth_dir)
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            if LR != 0.0001:
                epoch_size = 9

        train(model, optimizer, dataloader_tra, device, epoch)
        with torch.no_grad():
            loss_val, NSS = validate(model, dataloader_val, device, epoch)
            if epoch == 0:
                best_val = loss_val

            if best_val >= loss_val:
                best_val = loss_val
                print('SAVED weights in Epoch {:d}'.format(epoch))
                if not os.path.exists('./weights'):
                    os.mkdir('./weights')
                torch.save(model.state_dict(),
                           './weights/' + '{}.pth'.format(LR))
                pth_dir = './weights/' + '{}.pth'.format(LR)


if __name__ == '__main__':
    main()




