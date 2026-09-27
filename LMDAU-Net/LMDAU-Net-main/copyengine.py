import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch.nn.functional as F

def structure_loss(pred, mask):
    # 计算结构化损失，包括加权二值交叉熵损失和加权IoU损失
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # 切换到训练模式
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                p1, p2, p3, p4 = model(images)
                # 分别计算多尺度输出的结构化损失
                loss_p1 = structure_loss(p1, targets)
                loss_p2 = structure_loss(p2, targets)
                loss_p3 = structure_loss(p3, targets)
                loss_p4 = structure_loss(p4, targets)
                # 综合损失
                loss = loss_p1 + loss_p2 + loss_p3 + loss_p4
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            p1, p2, p3, p4 = model(images)
            loss_p1 = structure_loss(p1, targets)
            loss_p2 = structure_loss(p2, targets)
            loss_p3 = structure_loss(p3, targets)
            loss_p4 = structure_loss(p4, targets)
            loss = loss_p1 + loss_p2 + loss_p3 + loss_p4
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # 切换到评估模式
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            p1, p2, p3, p4 = model(img)
            loss_p1 = structure_loss(p1, msk)
            loss_p2 = structure_loss(p2, msk)
            loss_p3 = structure_loss(p3, msk)
            loss_p4 = structure_loss(p4, msk)
            loss = loss_p1 + loss_p2 + loss_p3 + loss_p4
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            out = p1 + p2 + p3 + p4  # 加和多尺度输出
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # 切换到评估模式
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            p1, p2, p3, p4 = model(img)
            loss_p1 = structure_loss(p1, msk)
            loss_p2 = structure_loss(p2, msk)
            loss_p3 = structure_loss(p3, msk)
            loss_p4 = structure_loss(p4, msk)
            loss = loss_p1 + loss_p2 + loss_p3 + loss_p4
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            out = p1 + p2 + p3 + p4  # 加和多尺度输出
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
