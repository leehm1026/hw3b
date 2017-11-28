import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from skimage import io, transform
import torch.optim as optim
import random
import cv2


class SiamesNetwork(nn.Module):
         
    def __init__(self):
        super(SiamesNetwork, self).__init__()
        self.C1 = nn.Sequential(
            # Convolution
            nn.Conv2d(3, 64, 5, 1, 2),
            # ReLU (in-place)
            nn.ReLU(inplace=True),
            # Barch Normalization 
            nn.BatchNorm2d(64),
            # Max pooling
            nn.MaxPool2d(2,(2,2))
            )
        self.C2 = nn.Sequential(
            # Convolution
            nn.Conv2d(64, 128, 5, 1, 2),
            # ReLU (in-place)
            nn.ReLU(inplace=True),
            # Barch Normalization 
            nn.BatchNorm2d(128),
            # Max pooling
            nn.MaxPool2d(2,(2,2))
            )
        self.C3 = nn.Sequential(
            # Convolution
            nn.Conv2d(128, 256, 3, 1, 1),
            # ReLU (in-place)
            nn.ReLU(inplace=True),
            # Barch Normalization 
            nn.BatchNorm2d(256),
            # Max pooling
            nn.MaxPool2d(2,(2,2))  
            )
        self.C4 = nn.Sequential(
            # Convolution
            nn.Conv2d(256, 512, 3, 1, 1),
            # ReLU (in-place)
            nn.ReLU(inplace=True),
            # Barch Normalization 
            nn.BatchNorm2d(512)
            )
        self.Flat = nn.Sequential(
            # Aka Linear Layer
            nn.Linear(16*16*512, 1024), 
            # ReLU (in-place)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
            )
        self.lastLinear = nn.Linear(1024*2, 1)
        
        self.sigmoid = nn.Sigmoid()


    def forward_once(self, x):
        output = self.C1(x)
        output = self.C2(output)
        output = self.C3(output)
        output = self.C4(outout)
        output = output.view(output.size()[0], -1)
        output = self.Flat(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

net = SiamesNetwork().cuda()

# Data Loader

class Faceloader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(txt_file, sep=" " ,header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        img2_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 1])
        label = self.landmarks_frame.ix[idx, 2]
        image1 = io.imread(img1_name)
        image2 = io.imread(img2_name)
        
        sample = {'image1': image1, 'image2': image2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
# Image modification Process  
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image1, image2, label = sample['image1'], sample['image2'], sample['label']

        h, w = image1.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img1 = transform.resize(image1, (new_h, new_w))
        img2 = transform.resize(image2, (new_h, new_w))
        
        return {'image1': img1, 'image2': img2, 'label': label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1, image2, label = sample['image1'], sample['image2'], sample['label']

        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        return {'image1': torch.from_numpy(image1),
                'image2': torch.from_numpy(image2),
                'label': label}
    
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image1, image2, label = sample['image1'], sample['image2'], sample['label']
        
        if random.random() < 0.7:
            image1 = cv2.flip(image1,1)
            image2 = cv2.flip(image2,1)
            
            return {'image1': image1, 'image2': image2, 'label': label}
        return   {'image1': image1, 'image2': image2, 'label': label}  
    


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image1, image2, label = sample['image1'], sample['image2'], sample['label']

        h, w = image1.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image1 = image1[top: top + new_h,
                      left: left + new_w]
        
        image2 = image2[top: top + new_h,
                      left: left + new_w]



        return   {'image1': image1, 'image2': image2, 'label': label}   
    
    

class Rotation(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1, image2, label = sample['image1'], sample['image2'], sample['label']
        
        if random.random() < 0.7:
            angle = random.randrange(-30 ,30 ,1)
            image1 = transform.rotate(image1, angle)
            image2 = transform.rotate(image2, angle)
            return {'image1': image1, 'image2': image2, 'label': label}
        
        return {'image1': image1, 'image2': image2, 'label': label} 


class Translation(object):
    """Convert ndarrays in sample to Tensors."""


    def __call__(self, sample):
        image1, image2, label = sample['image1'], sample['image2'], sample['label']

        h, w = image1.shape[:2]
        
        drift = random.randrange(-10, 10, 1)
        
        M = np.float32([[1,0,drift],[0,1,drift]])

        if random.random() < 0.7:

            img1 = cv2.warpAffine(image1, M, (h, w))
            img2 = cv2.warpAffine(image2, M, (h, w))
            return {'image1': img1, 'image2': img2, 'label': label}
        
        return {'image1': image1, 'image2': image2, 'label': label}

    
    
    
# Data Loading


transformed_dataset = Faceloader(txt_file='train.txt',root_dir='lfw',transform=transforms.Compose([Rescale(128),ToTensor()]))

testset = Faceloader(txt_file='test.txt', root_dir='lfw', transform=transforms.Compose([Rescale(128),ToTensor()]))  

    
# Train Loader

train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=8,shuffle=True, num_workers=2)
test_train_loader = torch.utils.data.DataLoader(testset, batch_size=8 ,shuffle=True, num_workers=2)



# loss criterion
criterion = BCELoss()
# Optimization
optimizer = optim.Adam(net.parameters(), betas = (0.9,0.9), lr = 1e-3)


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i_batch, sample_batched in enumerate(train_loader):
        # get the inputs
        image1, image2, label = sample_batched['image1'], sample_batched['image2'], sample_batched['label'] 
        image1 ,image2 = image1.type(torch.cuda.FloatTensor),image2.type(torch.cuda.FloatTensor)
        image1, image2 = Variable(image1), Variable(image2)
        label = label.type(torch.cuda.FloatTensor)

        
        # forward + backward + optimize
        outputs1, outputs2 = net(image1, image2)


        loss = criterion(outputs1, outputs2, Variable(label))
        loss.backward()
        optimizer.step()
   
#         print statistics
        running_loss += loss.data[0]
        if i_batch % 2 == 1:    # print every 2 mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i_batch + 1, running_loss/2))
            running_loss = 0.0

print('Finished Training')





# Load and Save model
torch.save(net.state_dict(),'model_2')


net.load_state_dict(torch.load('model_2'))


accuracy = 0.0
# Test the network
for i_batch, sample_batched in enumerate(testloader):
        # get the inputs
        image1, image2, label = sample_batched['image1'], sample_batched['image2'], sample_batched['label'] 
        image1 ,image2 = image1.type(torch.cuda.FloatTensor),image2.type(torch.cuda.FloatTensor)
        image1, image2 = Variable(image1), Variable(image2)
        label = label.type(torch.cuda.FloatTensor)
        label = label.numpy()
        # forward + backward + optimize
        outputs1, outputs2 = net(image1, image2)
        euclidean_distance = F.pairwise_distance(outputs1, outputs2)
        
#        print outputs[i],label[i]
#        print outputs[i] == label[i

        a = torch.ones(8,1)
        a = Variable(a)
        for i in range(len(euclidean_distance)):
            a[i] = (euclidean_distance[i] <= 1)
        
        a = a.data.numpy()
        for i in range(len(a)):
            if a[i][0] == label[i]:
                accuracy += 1
        
        

#       print statistics
        print("train accuracy is ",(i_batch + 1, accuracy/(i_batch+1)/8))

                  
