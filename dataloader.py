import os
import torch
from torch.utils import data
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from scipy import io


def resize_fixation(image, row, col):
    resized_fixation = np.zeros((row, col))
    ratio_row = row / image.shape[0]
    ratio_col = col / image.shape[1]

    coords = np.argwhere(image)
    for coord in coords:
        coord_r = int(np.round(coord[0] * ratio_row))
        coord_c = int(np.round(coord[1] * ratio_col))
        if coord_r == row:
            coord_r -= 1
        if coord_c == col:
            coord_c -= 1
        resized_fixation[coord_r, coord_c] = 1

    return resized_fixation


class DHF1KDataset(data.Dataset):
    def __init__(self, frame_path, label_path,  len_clip,  loader_mode):
        super(DHF1KDataset, self).__init__()
        self.frame_path = frame_path
        self.label_path = label_path
        self.len_clip = len_clip
        self.loader_mode = loader_mode

        if self.loader_mode == 'tra':
            self.video_names = sorted(os.listdir(frame_path))[0:600]
            self.list_frame_num = [len(os.listdir(os.path.join(frame_path, video_name, 'images'))) for video_name in self.video_names]
        elif self.loader_mode == 'val':
            self.list_frame_num = []
            for video_name in sorted(os.listdir(frame_path))[600:700]:
                for start_frame in range(0, len(os.listdir(os.path.join(frame_path, video_name, 'images'))) - self.len_clip + 1, 32):
                    self.list_frame_num.append((video_name, start_frame))

        self.img_trans = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        if self.loader_mode == 'tra':
            video_name = self.video_names[index]
            start_frame = np.random.randint(0, self.list_frame_num[index] - self.len_clip)  # [0, num-len)
        elif self.loader_mode == 'val':
            (video_name, start_frame) = self.list_frame_num[index]

        clip_path = os.path.join(self.frame_path, video_name, 'images')
        label_path = os.path.join(self.label_path, video_name, 'maps')
        fixation_path = os.path.join(self.label_path, video_name, 'fixation')

        img_clip = []
        for i in range(self.len_clip):
            img = Image.open(os.path.join(clip_path, '%04d.png' % (start_frame + i + 1))).convert('RGB')
            img = self.img_trans(img)
            img_clip.append(img)
        img_clip = torch.stack(img_clip, dim=0)
        img_clip = img_clip.permute((1, 0, 2, 3))

        # label = Image.open(os.path.join(label_path, '%04d.png' % (start_frame + self.len_clip - int(self.len_clip / 2)))).convert('L')
        label = Image.open(
            os.path.join(label_path, '%04d.png' % (start_frame + self.len_clip- int(self.len_clip / 2)))).convert('L')
        label = np.array(label)
        label = label.astype('float')

        fixation = Image.open(os.path.join(fixation_path, '%04d.png' % (start_frame + self.len_clip- int(self.len_clip / 2)))).convert('L')
        fixation = np.array(fixation)

        if self.loader_mode == 'tra':
            label = cv2.resize(label, (384, 224))
            label = label / 255.0
            fixation = resize_fixation(fixation, 224, 384)
        else:
            label = label / 255.0
            fixation = fixation / 255.0

        return img_clip, torch.FloatTensor(label), torch.FloatTensor(fixation)

    def __len__(self):
        return len(self.list_frame_num)


class UCFDataset(data.Dataset):
    def __init__(self, frame_path, len_clip,  loader_mode):
        super(UCFDataset, self).__init__()
        self.len_clip = len_clip
        self.loader_mode = loader_mode

        if self.loader_mode == 'tra':
            self.frame_path = os.path.join(frame_path, 'training')
            self.video_names = os.listdir(self.frame_path)
            self.list_frame_num = [len(os.listdir(os.path.join(self.frame_path, video_name, 'images'))) for video_name in self.video_names]

        elif self.loader_mode == 'test':
            self.frame_path = os.path.join(frame_path, 'testing')
            self.list_frame_num = []
            for video_name in os.listdir(self.frame_path):
                for start_frame in range(0, len(os.listdir(os.path.join(self.frame_path, video_name, 'images'))) - self.len_clip, 4):
                    self.list_frame_num.append((video_name, start_frame))

        self.img_trans = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        if self.loader_mode == 'tra':
            video_name = self.video_names[index]
            start_frame = np.random.randint(0, self.list_frame_num[index] - self.len_clip)
        elif self.loader_mode == 'val' or self.loader_mode == 'test':
            (video_name, start_frame) = self.list_frame_num[index]

        clip_path = os.path.join(self.frame_path, video_name, 'images')
        label_path = os.path.join(self.frame_path, video_name, 'maps')
        fixation_path = os.path.join(self.frame_path, video_name, 'fixation')

        img_clip = []
        for i in range(self.len_clip):
            img = Image.open(os.path.join(clip_path, video_name[:video_name.rfind('-')] + '_' + video_name[video_name.rfind('-') + 1:] + '_%03d.png' % (start_frame + i + 1))).convert('RGB')
            img = self.img_trans(img)
            img_clip.append(img)
        img_clip = torch.stack(img_clip, dim=0)
        img_clip = img_clip.permute((1, 0, 2, 3))

        label = Image.open(os.path.join(label_path, video_name[:video_name.rfind('-')] + '_' + video_name[video_name.rfind('-') + 1:] + '_%03d.png' %
                                        (start_frame + self.len_clip - int(self.len_clip / 2)))).convert('L')
        label = np.array(label)
        label = label.astype('float')

        fixation = Image.open(os.path.join(fixation_path, video_name[:video_name.rfind('-')] + '_' + video_name[video_name.rfind('-') + 1:] + '_%03d.png' %
                                           (start_frame + self.len_clip - int(self.len_clip / 2)))).convert('L')
        fixation = np.array(fixation)

        if self.loader_mode == 'tra':
            label = cv2.resize(label, (384, 224))
            label = label / 255.0
            fixation = resize_fixation(fixation, 224, 384)
        else:
            label = label / 255.0
            fixation = fixation / 255.0

        return img_clip, torch.FloatTensor(label), torch.FloatTensor(fixation)

    def __len__(self):
        return len(self.list_frame_num)


class Hollywood(data.Dataset):
    def __init__(self, frame_path, len_clip,  loader_mode):
        super(Hollywood, self).__init__()
        self.len_clip = len_clip
        self.loader_mode = loader_mode

        if self.loader_mode == 'tra':
            # self.frame_path = os.path.join(frame_path, 'training')
            self.frame_path = frame_path
            self.video_names = os.listdir(self.frame_path)
            self.list_frame_num = [len(os.listdir(os.path.join(self.frame_path, video_name, 'images'))) for video_name in self.video_names]
            # print(self.list_frame_num)
        elif self.loader_mode == 'test':
            # self.frame_path = os.path.join(frame_path, 'testing')
            self.frame_path = frame_path
            self.list_frame_num = []
            for video_name in os.listdir(self.frame_path):
                for start_frame in range(0, len(os.listdir(os.path.join(self.frame_path, video_name, 'images'))) - self.len_clip+1, 64):
                    self.list_frame_num.append((video_name, start_frame))

        self.img_trans = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        if self.loader_mode == 'tra':
            video_name = self.video_names[index]
            start_frame = np.random.randint(0, self.list_frame_num[index] - self.len_clip + 1)
        elif self.loader_mode == 'val' or self.loader_mode == 'test':
            (video_name, start_frame) = self.list_frame_num[index]

        clip_path = os.path.join(self.frame_path, video_name, 'images')
        label_path = os.path.join(self.frame_path, video_name, 'maps')
        fixation_path = os.path.join(self.frame_path, video_name, 'fixation')

        img_clip = []
        for i in range(self.len_clip):
            img = Image.open(os.path.join(clip_path, video_name.split('_')[0] + '_%05d.png' % (start_frame + i + 1))).convert('RGB')
            img = self.img_trans(img)
            img_clip.append(img)
        img_clip = torch.stack(img_clip, dim=0)
        img_clip = img_clip.permute((1, 0, 2, 3))

        label = Image.open(os.path.join(label_path, video_name.split('_')[0] + '_%05d.png' % (start_frame + self.len_clip - int(self.len_clip / 2)))).convert('L')
        label = np.array(label)
        label = label.astype('float')

        fixation = Image.open(os.path.join(fixation_path, video_name.split('_')[0] + '_%05d.png' % (start_frame + self.len_clip - int(self.len_clip / 2)))).convert('L')
        fixation = np.array(fixation)

        if self.loader_mode == 'tra':
            label = cv2.resize(label, (384, 224))
            label = label / 255.0
            fixation = resize_fixation(fixation, 224, 384)
        else:
            label = label / 255.0
            fixation = fixation / 255.0

        return img_clip, torch.FloatTensor(label), torch.FloatTensor(fixation)

    def __len__(self):
        return len(self.list_frame_num)


class DIEM(data.Dataset):
    def __init__(self, frame_path, len_clip,  loader_mode):
        super(DIEM, self).__init__()
        self.frame_path = frame_path
        self.len_clip = len_clip
        self.loader_mode = loader_mode

        if self.loader_mode == 'tra':
            self.video_names = os.listdir(os.path.join(frame_path, 'training'))
            self.list_frame_num = [len(os.listdir(os.path.join(frame_path, 'training', video_name))) for video_name in self.video_names]
        elif self.loader_mode == 'val':
            self.list_frame_num = []
            for video_name in os.listdir(os.path.join(frame_path, 'testing')):
                for start_frame in range(0, len(os.listdir(os.path.join(frame_path, 'testing', video_name))) - self.len_clip + 1, 32):
                # for start_frame in range(0, 300, 5):
                    self.list_frame_num.append((video_name, start_frame))

        self.img_trans = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        if self.loader_mode == 'tra':
            video_name = self.video_names[index]
            start_frame = np.random.randint(0, self.list_frame_num[index] - self.len_clip)  # [0, num-len)
            clip_path = os.path.join(self.frame_path, 'training', video_name)
        elif self.loader_mode == 'val':
            (video_name, start_frame) = self.list_frame_num[index]
            clip_path = os.path.join(self.frame_path, 'testing', video_name)

        label_path = os.path.join(self.frame_path, 'ann', video_name)
        fixation_path = os.path.join(self.frame_path, 'ann', video_name)

        img_clip = []
        for i in range(self.len_clip):
            img = Image.open(os.path.join(clip_path, 'img_%05d.jpg' % (start_frame + i + 1))).convert('RGB')
            img = self.img_trans(img)
            img_clip.append(img)
        img_clip = torch.stack(img_clip, dim=0)
        img_clip = img_clip.permute((1, 0, 2, 3))

        # label = Image.open(os.path.join(label_path, '%04d.png' % (start_frame + self.len_clip - int(self.len_clip / 2)))).convert('L')
        label = Image.open(
            os.path.join(label_path, 'maps',  'eyeMap_%05d.jpg' % (start_frame + self.len_clip- int(self.len_clip / 2)-15))).convert('L')
        label = np.array(label)
        label = label.astype('float')

        # fixation = io.loadmat('/home/wzq/data/DIEM/ann/university_forum_construction_ionic_1280x720/fixMap_00001.mat')['eyeMap']
        fixation = io.loadmat(os.path.join(fixation_path, 'fixMap_%05d.mat' % (start_frame + self.len_clip- int(self.len_clip / 2)-15)))['eyeMap']
        # if fixation.max() == 0:
        #     print(video_name, start_frame)
        fixation = np.array(fixation)

        if self.loader_mode == 'tra':
            label = cv2.resize(label, (384, 224))
            label = label / 255.0
            fixation = resize_fixation(fixation, 224, 384)
        else:
            label = label / 255.0
            fixation = fixation / 1.0

        return img_clip, torch.FloatTensor(label), torch.FloatTensor(fixation)

    def __len__(self):
        return len(self.list_frame_num)