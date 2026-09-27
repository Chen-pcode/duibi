import argparse
import os
from solver import Solver  
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True

    # Create model and result directories if they don't exist
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.result_path, exist_ok=True)

    print(config)

    # Load data
    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)

    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0)

    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0)

    # UCA-Net Ablation Study - Available models
    all_ablation_models = [
        'UCA_Net'
    ]
    
    # Filter models based on command line argument
    if config.models:
        ablation_models = [model for model in config.models.split(',') if model in all_ablation_models]
        if not ablation_models:
            print(f"Warning: No valid models found. Available models: {all_ablation_models}")
            ablation_models = ['UCA_Net'] 
    else:
        ablation_models = ['UCA_Net']  
        
    print(f"Running models: {ablation_models}")

    # Process each model
    for model in ablation_models:
        print(f"\n{'='*50}")
        print(f"Processing : {model}")
        print(f"{'='*50}")

        solver = Solver(config, model, train_loader, valid_loader, test_loader)

        if config.mode == 'train':
            # Training phase
            for loss in ['BCE_Dice_mIoU']:
                solver.train(loss=loss)

        elif config.mode == 'test':


            # Testing phase
            print(f"\n{'='*50}")
            print(f"Testing : {model}")
            print(f"{'='*50}")
            for loss in ['BCE_Dice_mIoU']:
                solver.test(loss=loss, data='PH2', model=model)

        else:
            print(f"Invalid mode: {config.mode}. Please use 'train' or 'test'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=320)   
    parser.add_argument('--num_workers', type=int, default=0)

    # Training hyper-parameters - Research-Proven for Medical Segmentation
    parser.add_argument('--lr', type=float, default=0.001)        
    parser.add_argument('--num_epochs', type=int, default=100)     
    parser.add_argument('--num_epochs_decay', type=int, default=10) 
    parser.add_argument('--batch_size', type=int, default=2)       
    parser.add_argument('--loss_threshold', type=float, default=0.8)
    parser.add_argument('--loss_type', type=str, default='BCE_Dice_mIoU', help='[BCE,BCE_mIoU,BCE_Dice_mIoU]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='[Adam,SGD,AdamW]')
    parser.add_argument('--beta1', type=float, default=0.9)       
    parser.add_argument('--beta2', type=float, default=0.999)     
    parser.add_argument('--weight_decay', type=float, default=0.00005) 
    parser.add_argument('--augmentation_prob', type=float, default=0.9) 



    # Misc  
    parser.add_argument('--report_file', type=str, default='PH2')
    parser.add_argument('--mode', type=str, default='train', help='[train,test]')
    parser.add_argument('--dataset', type=str, default='PH2', help='[PH2,ISIC2017,ISIC2018]')
    parser.add_argument('--use_enhanced_lmnet', action='store_true')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of UCA-Net models to run (e.g., UCA_Net,UCA_Net_Baseline)')
    parser.add_argument('--save_images', action='store_true', help='Save predicted images during testing')

   
    parser.add_argument('--train_path', type=str, default='../train/')
    parser.add_argument('--valid_path', type=str, default='../valid/')
    parser.add_argument('--test_path', type=str, default='../test/')
    parser.add_argument('--model_path', type=str, default='../Results/models/')
    parser.add_argument('--result_path', type=str, default='../Results/results/')
    parser.add_argument('--SR_path', type=str, default='../Results/SR/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    
    # Automatically enable save_images when in test mode
    if config.mode == 'test' and not config.save_images:
        config.save_images = True
        print("🖼️  Auto-enabled save_images for test mode")
    

    main(config)


