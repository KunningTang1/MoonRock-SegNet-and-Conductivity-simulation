
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import sys

def unet_conv(in_planes, out_planes):
    conv = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        # nn.Dropout2d(0.2),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        # nn.Dropout2d(0.2)
    )
    return conv


class Uresnet(nn.Module):
    def __init__(self, input_nbr = 3,label_nbr = 6):
        super(Uresnet, self).__init__()
        
        # forwarf
        self.downconv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )      # No.1 long skip 
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.ReLU(True),
        )      # No1 resudual block
        
        self.downconv3 = unet_conv(128, 128) # No2 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv4 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.ReLU(True),
        )      # No2 resudual block
        
        self.downconv5 = unet_conv(256, 256) # No3 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv6 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.ReLU(True),
        )      # No3 resudual block
        
        self.downconv7 = unet_conv(512, 512) # No4 long skip

        self.updeconv2 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
           
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(True),
        )       # No6 resudual block
        self.upconv4 = unet_conv(256, 256)
        
        self.updeconv3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
        )
           
        self.upconv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(True),
        )       # No6 resudual block
        self.upconv6 = unet_conv(128, 128)
        self.updeconv4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )
        
        self.upconv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(True),
        
        )       # No6 resudual block
        self.upconv8 = unet_conv(64, 64)
        
        self.last = nn.Conv2d(64, label_nbr, 1)  # 6 is number of phases to be segmented
        
        # xavier       
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    
            
        
    def forward(self, x):
        
        # encoding
        x1 = self.downconv1(x) 
 
        x2 = self.maxpool(x1)     
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)      
        x4 = x4 + x3
        x5 = self.maxpool(x4)
        
        x6 = self.downconv4(x5)
        x7 = self.downconv5(x6)
        x7 = x7+x6
        x8 = self.maxpool(x7)
        
        x9 = self.downconv6(x8)
        x10 = self.downconv7(x9)
        x10 = x9 +x10

        y3 = nn.functional.interpolate(x10, mode='bilinear', scale_factor=2,align_corners=True)
        y4 = self.updeconv2(y3)
        y5 = self.upconv3(torch.cat([y4, x7],1))
        y6 = self.upconv4(y5)
        y6 = y5 + y6
        
        y6 = nn.functional.interpolate(y6, mode='bilinear', scale_factor=2,align_corners=True)
        y7 = self.updeconv3(y6)   
        y8 = self.upconv5(torch.cat([y7, x4],1))
        y9 = self.upconv6(y8)
        y9 = y8 +y9
        
        y9 = nn.functional.interpolate(y9, mode='bilinear', scale_factor=2,align_corners=True)
        y10= self.updeconv4(y9)
        y11 = self.upconv7(torch.cat([y10, x1],1))
        y12 = self.upconv8(y11)
        y12 = y11+y12
     
        out = self.last(y12)
        
        return out

def uresnet():
    net = Uresnet()
    return net

##############################
#        Discriminator
##############################
class Discriminator_n_layers(nn.Module):
    def __init__(self, nlayers = 5):
        super(Discriminator_n_layers, self).__init__()

        n_layers = nlayers
        in_channels = 9
        def discriminator_block(in_filters, out_filters, k=4, s=2, p=1, norm=True, sigmoid=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=k, stride=s, padding=p)]
            if norm:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sigmoid:
                layers.append(nn.Sigmoid())
                print('use sigmoid')
            return layers

        sequence = [*discriminator_block(in_channels, 64, norm=False)]

        assert n_layers<=5

        if (n_layers == 1):
            'when n_layers==1, the patch_size is (16x16)'
            out_filters = 64* 2**(n_layers-1)

        elif (1 < n_layers & n_layers<= 4):
            '''
            when n_layers==2, the patch_size is (34x34)
            when n_layers==3, the patch_size is (70x70), this is the size used in the paper
            when n_layers==4, the patch_size is (142x142)
            '''
            for k in range(1,n_layers): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
            out_filters = 64* 2**(n_layers-1)

        elif (n_layers == 5):
            '''
            when n_layers==5, the patch_size is (286x286), lis larger than the img_size(256),
            so this is the whole img condition
            '''
            for k in range(1,4): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
                # k=4
            sequence += [*discriminator_block(2**9, 2**9)] #
            out_filters = 2**9

        num_of_filter = min(2*out_filters, 2**9)

        sequence += [*discriminator_block(out_filters, num_of_filter, k=4, s=1, p=1)]
        sequence += [*discriminator_block(num_of_filter, 1, k=4, s=1, p=1, norm=False, sigmoid=False)]

        self.model = nn.Sequential(*sequence)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        #print("self.model(img_input):  ",self.model(img_input).size())
        return self.model(img_input)
    
if __name__ == "__main__":  
    net = Uresnet()  
    discriminator = Discriminator_n_layers(4)
    if torch.cuda.is_available():
        net = net.cuda()    
        discriminator = discriminator.cuda() 
        
    colormap = [[255,0,0],[79,255,130],[198,118,255],[255,225,10],[84,226,255],[0,0,0]]
    num_classes = len(colormap)

    cm2lbl = np.zeros(256**3)
    for i,cm in enumerate(colormap):
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
    
    def image2label(im):
        data = np.array(im, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(cm2lbl[idx], dtype='int64')
    num_classes = len(colormap)
    number =  [[],[],[],[]]
    
    
    def label2image(im):
        data = np.array(im, dtype='uint8')
      
        return np.array(number)[data] 

    ROOT = "C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\X036\\3Draw_8bit.nii\\"
    def read_image(mode="train", val=False):
        if(mode=="train"):
            filename = ROOT + "\\train.txt"
        elif(mode == "test"):
            filename = ROOT + "\\test.txt"   
        data = []
        label = []
        with open(filename, "r") as f:
            images = f.read().split()
            for i in range(len(images)):
                if(i%2 == 0):
                    data.append(ROOT+images[i])
                else:
                    label.append(ROOT+images[i])
                    
        print(mode+":contains: "+str(len(data))+" images")
        print(mode+":contains: "+str(len(label))+" labels")
        return data, label
    
    
    data, label = read_image("train")
       
    size = 96
    def crop(data, label, height=size, width=size):
        st_x = 0
        st_y = 0
        box = (st_x, st_y, st_x+width, st_y+height)
        data = data.crop(box)
        label = label.crop(box)
        return data, label
    

    
    def image_transforms(data, label, height=size, width=size):
        data, label = crop(data, label, height, width)
        im_tfs = tfs.Compose([
            tfs.ToTensor(),
            # tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        data = im_tfs(data)
        label = np.array(label)
        label = image2label(label)
        label = torch.from_numpy(label).long()   # CrossEntropyLoss require a long() type
        return data, label
    
    class SegmentDataset(torch.utils.data.Dataset):
        
        # make functions
        def __init__(self, mode="train", height=size, width=size, transforms=image_transforms):
            self.height = height
            self.width = width
            self.transforms = transforms
            data_list, label_list = read_image(mode=mode)
            self.data_list = data_list
            self.label_list = label_list
            
        
        # do literation
        def __getitem__(self, idx):
            img = self.data_list[idx]
            label = self.label_list[idx]
            img = Image.open(img)
            img= img.convert('RGB')
            lb = Image.open(label)
 
            label= lb.convert('RGB')
            img, label = self.transforms(img, label, self.height, self.width)
            return img, label
        
        def __len__(self):
            return len(self.data_list)
    
    
    height = size
    width = size
    Segment_train = SegmentDataset(mode="train")
    Segment_test = SegmentDataset(mode="test")
    
    train_data = DataLoader(Segment_train, batch_size= 8, shuffle=True)
    test_data = DataLoader(Segment_test, batch_size= 1)
    
    # Confusion matrix
    def _fast_hist(label_true, label_pred, n_class):
    
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist
    
    def label_accuracy_score(label_trues, label_preds, n_class):
        hist = np.zeros((n_class, n_class))
    
        for lt, lp in zip(label_trues, label_preds):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc    

    
    LEARNING_RATE = 0.00001
                
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5 , patience=7, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    criterion = nn.CrossEntropyLoss()
    criterion_GAN = nn.MSELoss()
    
    def predict(img, label):
        img = img.unsqueeze(0).cuda()
        out = net(img)
        pred = out.max(1)[1].squeeze().cpu().data.numpy()
        return pred, label
    
    import random as rand
    
    # Show the comparison between original, labeled, segmented images
    def show(size=256, num_image=4, img_size=10, offset=0, shuffle=False):
        _, figs = plt.subplots(num_image, 3, figsize=(img_size, img_size))
        for i in range(num_image):
            if(shuffle==True):
                offset = rand.randint(0, min(len(Segment_train)-i-1, len(Segment_test)-i-1))
            img_data, img_label = Segment_test[i+offset]
            pred, label = predict(img_data, img_label)
            min_val,max_val,min_indx,max_indx = cv2.minMaxLoc(pred) 
            print('pred: ', min_val,max_val,min_indx,max_indx)
            img_data = Image.open(Segment_test.data_list[i+offset])
            img_label = Image.open(Segment_test.label_list[i+offset])
            img_data, img_label = crop(img_data, img_label) 
            img_label = img_label.convert('RGB')
            img_label = image2label(img_label)
            min_val,max_val,min_indx,max_indx = cv2.minMaxLoc(img_label) 
            print('Label: ',min_val,max_val,min_indx,max_indx)
            figs[i, 0].imshow(img_data)  
            figs[i, 0].axes.get_xaxis().set_visible(False)  
            figs[i, 0].axes.get_yaxis().set_visible(False)  
            figs[i, 1].imshow(img_label)                  
            figs[i, 1].axes.get_xaxis().set_visible(False) 
            figs[i, 1].axes.get_yaxis().set_visible(False)  
            figs[i, 2].imshow(pred)                       
            figs[i, 2].axes.get_xaxis().set_visible(False)  
            figs[i, 2].axes.get_yaxis().set_visible(False)  
    
        # titles
        figs[num_image-1, 0].set_title("Image", y=-0.2*(10/img_size))
        figs[num_image-1, 1].set_title("Label", y=-0.2*(10/img_size))
        figs[num_image-1, 2].set_title("U-resnet", y=-0.2*(10/img_size))
        plt.show()
        
    EPOCH = 100
    
    # train data record
    train_loss = []
    train_acc = []
    train_acc_cls = []
    train_mean_iu = []
    train_fwavacc = []
    average_train_acc = []
    # valid data record
    test_loss = []
    test_acc = []
    test_acc_cls = []
    test_mean_iu = []
    test_fwavacc = []
    k = 6
    average_val_acc = []

    num_zero_train_epoch = 0
    num_zero_test_epoch = 0
    num_zero_train = 0
    num_zero_test = 0
    train_zero = []
    test_zero = []
    average_test_acc = []
    lr = []


    for epoch in range(EPOCH):
        Num_0_test= Num_1_test=Num_2_test= Num_3_test=Num_4_test= Num_5_test = len(Segment_test)
        _train_loss = 0
        _train_acc = 0
        _train_acc_cls = 0
        _train_mean_iu = 0
        _train_fwavacc = 0
        _each_acc_train = 0
        prev_time = datetime.now()
        net = net.train()
        for step, (x,label) in enumerate(train_data):
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            # ---------------------
            #  Train Generator
            # ---------------------                        
            out = net(x)
            pred_fake = discriminator(out, x)
            valid = torch.ones((pred_fake.size(0),1,pred_fake.size(2),pred_fake.size(3)), requires_grad=False).cuda()
            fake = torch.zeros((pred_fake.size(0), 1,pred_fake.size(2),pred_fake.size(3)), requires_grad=False).cuda()
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion(out, label)
            
            loss_G = 0.05 * loss_GAN + loss_pixel
            
            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Real loss
            real_B= torch.nn.functional.one_hot(label.to(torch.int64), num_classes=6).cpu()
            real_B = np.transpose(real_B, (0, 3, 2, 1)).cuda()
        
            pred_real = discriminator(real_B, x)
            loss_real = criterion_GAN(pred_real, valid)
    
            # Fake loss
            pred_fake = discriminator(out.detach(), x)
            loss_fake = criterion_GAN(pred_fake, fake)
    
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()   
            
            sys.stdout.write("\r[Epoch%d/%d], [Batch%d/%d], [loss_G:%f], [loss_D:%f]" %
                       (epoch,EPOCH,
                       step, len(train_data),
                       loss_G.data.cpu(),
                       loss_D.cpu(),
                       ))    
        # recold loss and acc in the epoch
        
            _train_loss += loss_G.item()
    
            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                _train_acc += acc
                _train_acc_cls += acc_cls
                _train_mean_iu += mean_iu
                _train_fwavacc += fwavacc
        train_loss.append(_train_loss/len(train_data))
        train_acc.append(_train_acc/len(Segment_train))
        average_train_acc.append(_each_acc_train/len(Segment_train))
        print('')
        epoch_str = ('Epoch: {}, train Loss: {:.5f}, train Weight Acc: {:.5f}, train UNWeight Acc: {:.5f} '.format(
            epoch, _train_loss / len(train_data), _train_acc / len(Segment_train), _train_mean_iu / len(Segment_train)))
        print(epoch_str)

        scheduler.step(_train_loss/len(train_data))

        print('Epoch:', epoch, '| Learning rate_D', optimizer.state_dict()['param_groups'][0]['lr'])
        print('')
        
        if epoch == k:
            PATH = 'C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\X036\\3Draw_8bit.nii\\ck\\%d' % (epoch+1) +'.pt'
        
            torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
            k += 2
                

    # train loss visualization  
    plt.figure()  
    epoch = np.array(range(EPOCH))
    plt.plot(epoch, train_loss, label="train_loss")
    plt.plot(epoch, test_loss, label="test_loss")
    plt.title("loss during training")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure()    
    plt.plot(epoch, train_acc, label="train_acc")
    plt.plot(epoch, test_acc, label="test_acc")
    plt.plot(epoch, average_train_acc, label="average_train_acc")
    plt.plot(epoch, average_test_acc, label="average_test_acc")
    
    plt.title("accuracy during training")
    plt.legend()
    plt.grid()
    plt.show()

        
    show(offset=10)

