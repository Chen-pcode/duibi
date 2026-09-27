import os
import numpy as np
import time
from datetime import datetime
from PIL import Image
import csv
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from ptflops import get_model_complexity_info
from tensorboardX import SummaryWriter

# Evaluation and utilities
from evaluation import *
from misc import *

# UCA-Net model imports
from UCA_Net import UCA_Net



writer = SummaryWriter('mylogdir')


class Solver(object):
    def __init__(self, config, model, train_loader, valid_loader, test_loader):

        # Data loader
        self.mode = config.mode
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Hyper-parameters
        self.lr = config.lr
        self.optimizer_type = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0.01

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.num_epochs_decay = config.num_epochs_decay

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.SR_path = config.SR_path
        self.model_type = model
        self.dataset = config.dataset
        self.loss = config.loss_type

        # Report file
        self.report_file = config.report_file

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch

        self.augmentation_prob = config.augmentation_prob
        
        # Image saving control
        self.save_images = getattr(config, 'save_images', False)  # Default to False if not specified
        
   
        
        # Enhanced device detection for MPS (Apple Silicon) and CUDA
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🍎 Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("🚀 Using NVIDIA CUDA")
        else:
            self.device = torch.device('cpu')
            print("⚠️  Using CPU (no GPU acceleration available)")
            
        print(f"Device: {self.device}")
        
        self.criterion1 = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion2 = mIoULoss(threshold=config.loss_threshold).to(self.device)
        self.criterion3 = DiceLoss(threshold=config.loss_threshold).to(self.device)

        

       


    def build_model(self):
        """Build UCA_Net models and variants."""
        print("Initializing UCA_Net model training...")

        if self.model_type == 'UCA_Net':
            self.unet = UCA_Net(img_in=self.img_ch, segout=self.output_ch)
            print("🔥 UCA_Net: Complete model with all attention mechanisms")
            
       

        else:
            self.unet = U_Net(self.img_ch,self.output_ch)


        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.unet.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=1e-4)
        elif self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr, betas=[self.beta1, self.beta2], 
                                       weight_decay=self.weight_decay if hasattr(self, 'weight_decay') else 0.01)
        else:
            self.optimizer = optim.SGD(self.unet.parameters(), lr=self.lr, momentum=self.beta1, weight_decay=2e-4)

        # Learning rate scheduler
        if hasattr(self, 'weight_decay') and self.optimizer_type == 'Adam':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        else:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=self.num_epochs_decay)

        self.unet.to(self.device)
        self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.report.write('\n'+str(model))
        print(name)
        self.report.write('\n'+str(name))
        print("The number of parameters: {}".format(num_params))
        self.report.write("\n The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()


    def train(self,loss):
        factor = 0.8
        t = time.time()
        self.loss = loss
        isExist = os.path.exists(self.result_path + self.model_type+ '_' + loss)
        if not isExist:
            os.makedirs(self.result_path + self.model_type + '_' + loss)
        self.result_path_loss = os.path.join(self.result_path, self.model_type + '_' + loss) + '/' # Corrected path concatenation
        self.report = open(
            self.result_path_loss+ self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.txt',
            'a+')
        self.report.write('\n' + str(datetime.now()))

        self.f1 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_train.csv'),
                       'a', encoding='utf-8', newline='')
        self.f2 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_val.csv'),
                       'a', encoding='utf-8', newline='')
        self.model_save_path = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl')
        self.model_save_path1 = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss)

        self.build_model()
        wr1 = csv.writer(self.f1)
        wr1.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])
        wr2 = csv.writer(self.f2)
        wr2.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])

        # U-Net Train
        if os.path.isfile(self.model_save_path):
            try:
                # Try loading with weights_only=False for backward compatibility with older PyTorch models
                self.unet = torch.load(self.model_save_path, weights_only=False)
                print('%s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
                self.report.write('\n %s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
            except Exception as e:
                print(f"Warning: Could not load existing model from {self.model_save_path}")
                print(f"Error: {e}")
                print("Starting training from scratch...")
                self.report.write(f'\n Warning: Could not load existing model from {self.model_save_path}')
                self.report.write(f'\n Error: {e}')
                self.report.write('\n Starting training from scratch...')
                # Continue with fresh training
                best_unet_score = 0.
                results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["SP",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["DC",[],[]]]

                for epoch in range(self.num_epochs):
                    self.unet.train(True)
                    train_loss = 0.

                    acc = 0.
                    RC = 0.
                    SP = 0.
                    PC = 0.
                    F1 = 0.
                    IoU = 0
                    mIoU = 0.
                    DC = 0.
                    length = 0
                    buff = []

                    for i, (image, GT, name) in enumerate(self.train_loader):
                        # print('image')
                        # print(i)
                        # SR : Segmentation Result
                        # GT : Ground Truth
                        image = image.to(self.device)
                        GT = GT.to(self.device)
    # ----------------------------------UNet--------------------------------------------------------------

                        SR = self.unet(image)
                        
                        # Handle different model output formats
                        if self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4']:
                            # MobileNet models return output directly
                            SR = SR.contiguous().view(-1)
                        elif self.model_type in ['VMUNet', 'VMUNetV2', 'LightMUNet', 'EADSNet', 'FinalEnhancedEADSNet', 'FinalEnhancedEADSNet-v2', 'FinalEnhancedEADSNet-v3', 'ULS_MSA']:
                            # These models return single tensor output directly
                            SR = SR.contiguous().view(-1)
                        else:
                            # Other models (V4, V4-Lite, V3-Enhanced) return output as tuple/list
                            SR = SR[0]
                            SR = SR.contiguous().view(-1)
                    
                        GT = GT.contiguous().view(-1)

                        loss1 = self.criterion1(SR, GT)
                        loss2 = self.criterion2(SR, GT)
                        loss3 = self.criterion3(SR,GT)

                        #total_loss = loss1 + loss2 + loss3
                        total_loss = loss1 + (factor*(loss2+loss3))

                        self.reset_grad()
                        total_loss.backward()
                        self.optimizer.step()

                        SR = SR.detach()
                        GT = GT.detach()

                        train_loss += total_loss.detach().item()
                        acc += get_accuracy(SR,GT)
                        RC += get_Recall(SR,GT)
                        SP += get_specificity(SR,GT)
                        PC += get_Precision(SR,GT)
                        F1 += get_F1(SR,GT)
                        buff = get_mIoU(SR,GT)
                        IoU += buff[0]
                        mIoU += buff[1]
                        DC += get_DC(SR,GT)
                        length += 1

                    train_loss = train_loss/length
                    acc = acc/length
                    RC = RC/length
                    SP = SP/length
                    PC = PC/length
                    F1 = F1/length
                    IoU = IoU/length
                    mIoU = mIoU/length
                    DC = DC/length

                    results[0][1].append((train_loss))
                    results[1][1].append((acc*100))
                    results[2][1].append((RC*100))
                    results[3][1].append((SP*100))
                    results[4][1].append((PC*100))
                    results[5][1].append((F1*100))
                    results[6][1].append((IoU*100))
                    results[7][1].append((mIoU*100))
                    results[8][1].append((DC*100))

                    print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                        epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                        epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    wr1.writerow(
                        [epoch + 1, acc, RC, SP, PC, F1, IoU, mIoU, DC, self.lr, train_loss])
                    writer.add_scalar("Loss/train", train_loss, epoch+1)
                    writer.add_scalar("Precision/train", PC, epoch + 1)
                    writer.add_scalar("Recall/train", RC, epoch + 1)
                    writer.add_scalar("F1 Score/train", F1, epoch + 1)
                    writer.add_scalar("mIoU/train", mIoU, epoch + 1)

                    # Memory cleanup for both CUDA and MPS
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

    #===================================== Validation ====================================#
                    self.unet.train(False)
                    self.unet.eval()
                    valid_loss = 0.

                    acc = 0.
                    RC = 0.
                    SP = 0.
                    PC = 0.
                    F1 = 0.
                    IoU = 0
                    mIoU = 0.
                    DC = 0.
                    length=0
                    buff = []

                    for i, (image, GT, name) in enumerate(self.valid_loader):
                        
                        # SR : Segmentation Result
                        # GT : Ground Truth
                        image = image.to(self.device)
                        GT = GT.to(self.device)
                        GT_original = GT  # Keep original for image saving
                        GT_f = GT

    #-------------------------------------UNet-------------------------------------------------------
                        SR = self.unet(image)
                        
                        # Handle different model output formats
                        if self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2', 'LightMUNet', 'EADSNet', 'FinalEnhancedEADSNet', 'FinalEnhancedEADSNet-v2', 'FinalEnhancedEADSNet-v3', 'ULS_MSA']:
                            # MobileNet, VMUNet, LightMUNet, EADSNet, FinalEnhancedEADSNet, FinalEnhancedEADSNet-v2, FinalEnhancedEADSNet-v3, and ULS_MSA models return output directly
                            SR_original = SR  # Keep original for image saving
                            SR_f = SR.contiguous().view(-1)
                        else:
                            # Other models (V4, V4-Lite, V3-Enhanced) return output as tuple/list
                            SR_original = SR[0]  # Keep original for image saving
                            SR = SR[0]
                            SR_f = SR.contiguous().view(-1)
                      
                        GT_f = GT.contiguous().view(-1)
                        loss_val_1 = self.criterion1(SR_f, GT_f)
                        loss_val_2 = self.criterion2(SR_f, GT_f)
                        loss_val_3 = self.criterion3(SR_f,GT_f)

                        #total_loss = loss_val_1 
                        total_loss = loss_val_1 + (factor*(loss_val_2+loss_val_3))

                        # Apply sigmoid to convert logits to probabilities for metric computation
                        SR_f_prob = torch.sigmoid(SR_f.detach())
                        GT_f = GT_f.detach()

                        valid_loss += total_loss.detach().item()
                        acc += get_accuracy(SR_f_prob,GT_f)
                        RC += get_Recall(SR_f_prob,GT_f)
                        SP += get_specificity(SR_f_prob,GT_f)
                        PC += get_Precision(SR_f_prob,GT_f)
                        F1 += get_F1(SR_f_prob,GT_f)
                        buff = get_mIoU(SR_f_prob,GT_f)
                        IoU += buff[0]
                        mIoU += buff[1]
                        DC += get_DC(SR_f_prob,GT_f)
                        length += 1

                    valid_loss = valid_loss/length
                    acc = acc/length
                    RC = RC/length
                    SP = SP/length
                    PC = PC/length
                    F1 = F1/length
                    IoU = IoU/length
                    mIoU = mIoU/length
                    DC = DC/length
                    unet_score = mIoU

                    results[0][2].append((valid_loss))
                    results[1][2].append((acc*100))
                    results[2][2].append((RC*100))
                    results[3][2].append((SP*100))
                    results[4][2].append((PC*100))
                    results[5][2].append((F1*100))
                    results[6][2].append((IoU*100))
                    results[7][2].append((mIoU*100))
                    results[8][2].append((DC*100))

                    print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                        valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                        valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))

                    wr2.writerow([epoch+1 ,acc,RC,SP,PC,F1,IoU,mIoU,DC,self.lr,valid_loss])
                    writer.add_scalar("Loss/val", valid_loss, epoch + 1)
                    writer.add_scalar("Precision/val", PC, epoch + 1)
                    writer.add_scalar("Recall/val", RC, epoch + 1)
                    writer.add_scalar("F1 Score/val", F1, epoch + 1)
                    writer.add_scalar("mIoU/val", mIoU, epoch + 1)


                    # Handle different types of schedulers
                    if hasattr(self, 'weight_decay') and self.optimizer_type == 'AdamW':
                        self.lr_scheduler.step()  # CosineAnnealingWarmRestarts doesn't need metric
                    else:
                        self.lr_scheduler.step(valid_loss)  # ReduceLROnPlateau needs metric

                    if unet_score > best_unet_score:
                        best_unet_score = unet_score
                        print('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                        self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                        torch.save(self.unet,self.model_save_path)
                    epoch_custom = epoch + 1
                    if epoch_custom % 10 ==0:
                        torch.save(self.unet, self.model_save_path1+'_'+str(epoch_custom)+'.pkl')


                    if unet_score > 0.9:
                        torchvision.utils.save_image(image.data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                        torchvision.utils.save_image(torch.sigmoid(SR_original).data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                        torchvision.utils.save_image(GT_original.data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

                    # Memory cleanup for both CUDA and MPS
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                displayfigures(results, self.result_path_loss, self.report_file, self.dataset, self.model_type)
        else:
            best_unet_score = 0.
            results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["SP",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["DC",[],[]]]

            for epoch in range(self.num_epochs):
                self.unet.train(True)
                train_loss = 0.

                acc = 0.
                RC = 0.
                SP = 0.
                PC = 0.
                F1 = 0.
                IoU = 0
                mIoU = 0.
                DC = 0.
                length = 0
                buff = []

                for i, (image, GT, name) in enumerate(self.train_loader):
                    # print('image')
                    # print(i)
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
# ----------------------------------UCA_Net Models--------------------------------------------------------------

                    # Forward pass - UCA_Net models return single output
                    SR = self.unet(image)
                    
                    # Flatten predictions and ground truth
                    SR_flat = SR.contiguous().view(-1)
                    GT_flat = GT.contiguous().view(-1)
                    
                    # Calculate combined loss
                    loss1 = self.criterion1(SR_flat, GT_flat)
                    loss2 = self.criterion2(SR_flat, GT_flat)
                    loss3 = self.criterion3(SR_flat, GT_flat)
                    total_loss = loss1 + (factor * (loss2 + loss3))

                    self.reset_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    # Apply sigmoid to convert logits to probabilities for metric computation
                    with torch.no_grad():
                        SR_prob_flat = torch.sigmoid(SR_flat.detach())
                        GT_metric = GT_flat.detach()

                    train_loss += total_loss.detach().item()
                    acc += get_accuracy(SR_prob_flat, GT_metric)
                    RC += get_Recall(SR_prob_flat, GT_metric)
                    SP += get_specificity(SR_prob_flat, GT_metric)
                    PC += get_Precision(SR_prob_flat, GT_metric)
                    F1 += get_F1(SR_prob_flat, GT_metric)
                    buff = get_mIoU(SR_prob_flat, GT_metric)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR_prob_flat, GT_metric)
                    length += 1

                train_loss = train_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length

                results[0][1].append((train_loss))
                results[1][1].append((acc*100))
                results[2][1].append((RC*100))
                results[3][1].append((SP*100))
                results[4][1].append((PC*100))
                results[5][1].append((F1*100))
                results[6][1].append((IoU*100))
                results[7][1].append((mIoU*100))
                results[8][1].append((DC*100))

                print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                wr1.writerow(
                    [epoch + 1, acc, RC, SP, PC, F1, IoU, mIoU, DC, self.lr, train_loss])
                writer.add_scalar("Loss/train", train_loss, epoch+1)
                writer.add_scalar("Precision/train", PC, epoch + 1)
                writer.add_scalar("Recall/train", RC, epoch + 1)
                writer.add_scalar("F1 Score/train", F1, epoch + 1)
                writer.add_scalar("mIoU/train", mIoU, epoch + 1)

                # Memory cleanup for both CUDA and MPS
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

#===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()
                valid_loss = 0.

                acc = 0.
                RC = 0.
                SP = 0.
                PC = 0.
                F1 = 0.
                IoU = 0
                mIoU = 0.
                DC = 0.
                length=0
                buff = []

                for i, (image, GT, name) in enumerate(self.valid_loader):
                    
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
                    GT_original = GT  # Keep original for image saving

#-------------------------------------UCA_Net Models-------------------------------------------------------
                    # Forward pass - UCA_Net models return single output
                    SR = self.unet(image)
                    SR_original = SR  # Keep original for image saving
                    
                    # Flatten predictions and ground truth
                    SR_f = SR.contiguous().view(-1)
                    GT_f = GT.contiguous().view(-1)
                   
                    # Calculate combined loss
                    loss_val_1 = self.criterion1(SR_f, GT_f)
                    loss_val_2 = self.criterion2(SR_f, GT_f)
                    loss_val_3 = self.criterion3(SR_f, GT_f)
                    total_loss = loss_val_1 + (factor * (loss_val_2 + loss_val_3))

                    # Apply sigmoid to convert logits to probabilities for metric computation
                    with torch.no_grad():
                        SR_prob_flat = torch.sigmoid(SR_f.detach())
                        GT_metric = GT_f.detach()

                    valid_loss += total_loss.detach().item()
                    acc += get_accuracy(SR_prob_flat, GT_metric)
                    RC += get_Recall(SR_prob_flat, GT_metric)
                    SP += get_specificity(SR_prob_flat, GT_metric)
                    PC += get_Precision(SR_prob_flat, GT_metric)
                    F1 += get_F1(SR_prob_flat, GT_metric)
                    buff = get_mIoU(SR_prob_flat, GT_metric)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR_prob_flat, GT_metric)
                    length += 1

                valid_loss = valid_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length
                unet_score = mIoU

                results[0][2].append((valid_loss))
                results[1][2].append((acc*100))
                results[2][2].append((RC*100))
                results[3][2].append((SP*100))
                results[4][2].append((PC*100))
                results[5][2].append((F1*100))
                results[6][2].append((IoU*100))
                results[7][2].append((mIoU*100))
                results[8][2].append((DC*100))

                print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))

                wr2.writerow([epoch+1 ,acc,RC,SP,PC,F1,IoU,mIoU,DC,self.lr,valid_loss])
                writer.add_scalar("Loss/val", valid_loss, epoch + 1)
                writer.add_scalar("Precision/val", PC, epoch + 1)
                writer.add_scalar("Recall/val", RC, epoch + 1)
                writer.add_scalar("F1 Score/val", F1, epoch + 1)
                writer.add_scalar("mIoU/val", mIoU, epoch + 1)


                # Handle different types of schedulers
                if hasattr(self, 'weight_decay') and self.optimizer_type == 'AdamW':
                    self.lr_scheduler.step()  # CosineAnnealingWarmRestarts doesn't need metric
                else:
                    self.lr_scheduler.step(valid_loss)  # ReduceLROnPlateau needs metric

                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    print('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(self.unet,self.model_save_path)
                epoch_custom = epoch + 1
                if epoch_custom % 10 ==0:
                    torch.save(self.unet, self.model_save_path1+'_'+str(epoch_custom)+'.pkl')


                if unet_score > 0.9:
                    torchvision.utils.save_image(image.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                
                    torchvision.utils.save_image(torch.sigmoid(SR_original).data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                    
                    torchvision.utils.save_image(GT_original.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

              

                # Memory cleanup for both CUDA and MPS
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            displayfigures(results, self.result_path_loss, self.report_file, self.dataset, self.model_type)

        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.close()
        self.f1.close()
        self.f2.close()
        writer.close()

    def get_gradUCA_Net(self,image,SR, GT,size):
        total_loss = self.criterion1(SR, GT)
        total_loss.backward()
        gradients = self.unet.get_activation_gradients()
        pooled_gradients = torch.mean(gradients, dim=[0,2,3])
        activations = self.unet.get_activations(image).detach()
        for i in range(activations.shape[1]):
            activations[:,i,:,:] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
        heatmap = nn.ReLU()(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        image = image.squeeze(0)
        image = image.permute(1,2,0)
        image = image.cpu().numpy()
        image = np.uint8(image * 255)
        heatmap = cv2.resize(heatmap, (320, 320))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        return overlay, heatmap

    def test(self, loss, data, model): 
        # Construct model path with proper path joining
        model_file_path = os.path.join(self.model_path, self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl')
        
        if os.path.isfile(model_file_path):
            try:
                # Try loading with weights_only=False for backward compatibility with older PyTorch models
                self.unet = torch.load(model_file_path, weights_only=False)
                print('%s is Successfully Loaded from %s' % (self.model_type, model_file_path))
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Trained model NOT found or could not be loaded for {self.model_type} with loss {loss}, Please train a model first")
                return
        else:
            print(f"Trained model NOT found for {self.model_type} with loss {loss}")
            print(f"Expected path: {model_file_path}")
            print(f"Please train a model first")
            return

        # Create SR directory structure as in original design
        isExist = os.path.exists(self.SR_path + self.model_type + '_' + loss)
        if not isExist:
            os.makedirs(self.SR_path + self.model_type + '_' + loss)
        self.model_path_loss = self.SR_path + self.model_type + '_' + loss + '/'
        self.test_acc = open(self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_test.csv', 'a+')

        wr_test = csv.writer(self.test_acc)
        if os.path.getsize(self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_test.csv') == 0:
            wr_test.writerow(['Model', 'SE (%)', 'SP (%)', 'ACC (%)', 'IoU (%)', 'Dice (%)', 'Params(M)', 'FLOPs', 'Avg Inference Time'])


        self.unet.train(False)
        self.unet.eval()

        input_size_flops = (self.img_ch, 224, 224)
        try:
            macs, params = get_model_complexity_info(self.unet, input_size_flops, as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
            flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
            flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
        except Exception as e:
            print(f"Could not calculate FLOPs/Params for {self.model_type}: {e}")
            macs, params, flops, flops_unit = "N/A", "N/A", "N/A", ""

        print(f'Computational complexity: {macs}')
        print(f'Computational complexity: {flops} {flops_unit}Flops')
        print(f'Number of parameters: {params}')

        acc = 0.
        RC = 0.
        SP = 0.
        PC = 0.
        F1 = 0.
        IoU = 0
        mIoU = 0.
        DC = 0.
        total_elapsed = 0.
        length = 0

        with torch.no_grad():
            for i, (image, GT, name) in enumerate(self.test_loader):
                image = image.to(self.device)
                GT = GT.to(self.device)

                start_time = time.time()
                SR = self.unet(image)
                elapsed = time.time() - start_time
                total_elapsed += elapsed
                
                # UCA_Net models return single tensor output
                SR_f = SR.contiguous().view(-1)
                SR_f_sigmoid = torch.sigmoid(SR_f)
                SR_sigmoid = torch.sigmoid(SR)
                GT_f = GT.contiguous().view(-1)

                acc += get_accuracy(SR_f_sigmoid, GT_f)
                RC += get_Recall(SR_f_sigmoid, GT_f)
                SP += get_specificity(SR_f_sigmoid, GT_f)
                PC += get_Precision(SR_f_sigmoid, GT_f)
                F1 += get_F1(SR_f_sigmoid, GT_f)
                buff = get_mIoU(SR_f_sigmoid, GT_f)
                IoU += buff[0]
                mIoU += buff[1]
                DC += get_DC(SR_f_sigmoid, GT_f)
                length += 1

                threshold = 0.5
                
                # Process predictions for saving
                SR_processed = SR_sigmoid.squeeze(1)
                SR_processed[SR_processed < threshold] = 0
                SR_processed[SR_processed >= threshold] = 1

                # Only save images if save_images flag is True
                if self.save_images:
                    for j in range(SR_processed.shape[0]):
                        im = Image.fromarray(SR_processed[j].cpu().numpy() * 255).convert('L')
                        imo = im.resize((256, 256), resample=Image.BILINEAR)
                        imo.save(self.model_path_loss + name[j])

        acc /= length
        RC /= length
        SP /= length
        PC /= length
        F1 /= length
        IoU /= length
        mIoU /= length
        DC /= length

        # Convert to percentages and format properly
        SE_percent = RC * 100  # Sensitivity (Recall)
        SP_percent = SP * 100  # Specificity
        ACC_percent = acc * 100  # Accuracy
        IoU_percent = IoU * 100  # IoU
        Dice_percent = DC * 100  # Dice
        
        # Extract numeric values for params and FLOPs
        try:
            params_numeric = float(re.findall(r'([\d.]+)', params)[0])
        except:
            params_numeric = 0.0
            
        try:
            flops_numeric = flops
            flops_unit_str = flops_unit + "Flops"
        except:
            flops_numeric = 0.0
            flops_unit_str = "GFlops"

        total_images_processed = length * self.test_loader.batch_size
        avg_inference_time = total_elapsed / total_images_processed if total_images_processed > 0 else 0

        wr_test.writerow([self.model_type, f"{SE_percent:.2f}", f"{SP_percent:.2f}", f"{ACC_percent:.2f}", 
                         f"{IoU_percent:.2f}", f"{Dice_percent:.2f}", f"{params_numeric:.2f}", 
                         f"{flops_numeric:.2f}", f"{avg_inference_time:.6f}"])
        print('Results have been Saved')
        print(f'Average Inference Time per Image: {avg_inference_time:.6f} seconds')

        self.test_acc.close()