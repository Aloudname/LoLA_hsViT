# necessary function repo for train.py.

import os
import cv2
import torch
import numpy as np

from operator import truediv
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score


# ADDING MISSING EVALUATION AND SAVING FUNCTIONS
def AA_andEachClassAccuracy(confusion_matrix):
    """Calculate average accuracy and per-class accuracy"""
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test, dataset_name="LongKou"):
    """Generate comprehensive accuracy reports with target names"""
    dataset_configs = {
        "LongKou": {
            "target_names": ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
                           'Narrow-leaf soybean', 'Rice', 'Water',
                           'Roads and houses', 'Mixed weed'],
            "num_classes": 9
        },
        "IndianPines": {
            "target_names": ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                           'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                           'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                           'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                           'Stone-Steel-Towers'],
            "num_classes": 16
        },
        "PaviaU": {
            "target_names": ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                           'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows'],
            "num_classes": 9
        },
        "PaviaC": {
            "target_names": ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen',
                           'Tiles', 'Shadows', 'Meadows', 'Bare Soil'],
            "num_classes": 9
        },
        "Salinas": {
            "target_names": ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                           'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
                           'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                           'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                           'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'],
            "num_classes": 16
        },
        "HongHu": {
            "target_names": ['Red roof', 'Road', 'Bare soil', 'Red roof 2', 'Red roof 3',
                           'Gray roof', 'Red roof 4', 'White roof', 'Bright roof', 'Trees',
                           'Grass', 'Red roof 5', 'Red roof 6', 'Red roof 7', 'Red roof 8',
                           'Red roof 9', 'Red roof 10', 'Red roof 11', 'Red roof 12', 'Red roof 13',
                           'Red roof 14', 'Red roof 15'],
            "num_classes": 22
        },
        "Qingyun": {
            "target_names": ["Trees", "Concrete building", "Car", "Ironhide building",
                           "Plastic playground", "Asphalt road"],
            "num_classes": 6
            }
    }
    
    if dataset_name in dataset_configs:
        target_names = dataset_configs[dataset_name]["target_names"]
    else:
        # Fallback for unknown datasets
        target_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
    
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names, zero_division=0)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100, target_names

@torch.no_grad()
def test(device, model, test_loader):
    """Test model and get predictions"""
    model.eval()
    count = 0
    y_pred_test = 0
    y_test = 0
    
    for hyperspectral, pretrained, labels in test_loader:
        hyperspectral = hyperspectral.to(device)
        pretrained = pretrained.to(device)
        labels = labels.squeeze().to(device)
        
        # FIXED: Ensure correct tensor format for hyperspectral data
        # Input should be [B, C, H, W] where C is the number of channels (15 for hyperspectral)
        if hyperspectral.dim() == 4 and hyperspectral.shape[-1] == 15:  # [B, H, W, C]
            # [B, H, W, C] -> [B, C, H, W]
            hyperspectral = hyperspectral.permute(0, 3, 1, 2)
        
        # Normalize inputs (FIXED to match working model exactly)
        hyperspectral = (hyperspectral - hyperspectral.mean(dim=(2,3), keepdim=True)) / (hyperspectral.std(dim=(2,3), keepdim=True) + 1e-8)
        
        outputs = model(hyperspectral)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        
        if count == 0:
            y_pred_test = outputs
            y_test = labels.cpu().numpy()
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels.cpu().numpy()))
    
    return y_pred_test, y_test

def plt_curves(train_losses, train_accuracies, eval_accuracies, epoch):
    """Plot enhanced training curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Enhanced Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(eval_accuracies, label='Validation Accuracy')
    plt.title('Enhanced Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('enhanced_plots', exist_ok=True)
    plt.savefig(f'enhanced_plots/enhanced_training_curves_{epoch}.png')
    plt.close()

def save_results(model, config, best_acc, best_epoch, training_time, test_time, 
                         classification, oa, confusion, each_acc, aa, kappa, target_names,
                         pretrained_model_name, efficiency_results):
    """Save comprehensive enhanced results"""
    os.makedirs('enhanced_cls_result', exist_ok=True)
    file_name = f"enhanced_cls_result/classification_report_enhanced_{pretrained_model_name.replace('/', '_')}.txt"
    
    with open(file_name, 'w') as x_file:
        x_file.write(f'Enhanced Model Configuration:\n')
        x_file.write(f'Pretrained Model: {pretrained_model_name}\n')
        x_file.write(f'LoRA rank (r): 16\n')
        x_file.write(f'LoRA alpha: 32\n')
        x_file.write(f'Learning rate: {config["learning_rate"]}\n')
        x_file.write(f'Batch size: {config["batch_size"]}\n')
        x_file.write(f'Parameter reduction: {efficiency_results["parameter_reduction_percent"]:.2f}%\n\n')
        
        x_file.write(f'Training Time (s): {training_time:.2f}\n')
        x_file.write(f'Test Time (s): {test_time:.2f}\n')
        x_file.write(f'Best epoch: {best_epoch}\n\n')
        
        x_file.write(f'Enhanced Performance Metrics:\n')
        x_file.write(f'Overall Accuracy (%): {oa:.2f}\n')
        x_file.write(f'Average Accuracy (%): {aa:.2f}\n')
        x_file.write(f'Kappa Score (%): {kappa:.2f}\n\n')
        
        x_file.write(f'Per-Class Accuracies (%):\n')
        for name, acc in zip(target_names, each_acc):
            x_file.write(f'{name}: {acc:.2f}\n')
        x_file.write(f'\nDetailed Classification Report:\n{classification}\n')
        x_file.write(f'\nConfusion Matrix:\n{confusion}\n')
    
    print(f"\nEnhanced results saved to {file_name}")
    return file_name

def save_model(model, config, best_acc, kappa, training_time, test_time, 
                       best_epoch, each_acc, confusion, pretrained_model_name, efficiency_results):
    """Save enhanced model with comprehensive metadata"""
    os.makedirs('enhanced_peft_checkpoints', exist_ok=True)
    
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'performance': {
            'accuracy': best_acc,
            'kappa': kappa,
            'training_time': training_time,
            'test_time': test_time,
            'best_epoch': best_epoch,
            'per_class_accuracy': each_acc.tolist(),
            'confusion_matrix': confusion.tolist()
        },
        'model_config': {
            'pretrained_model': pretrained_model_name,
            'lora_rank': 16,
            'lora_alpha': 32,
            'dim': 96,
            'depths': [3, 4, 19],
            'num_heads': [4, 8, 16],
            'window_size': [7, 7, 7]
        },
        'efficiency_results': efficiency_results
    }, f'enhanced_peft_checkpoints/enhanced_final_model_{pretrained_model_name.replace("/", "_")}.pth')
    
    print(f"Enhanced model saved with comprehensive metadata")

def generate_cam_plot(model, input_tensor, target_class=None, device='cuda'):
    """
    Generate Class Activation Map (CAM) using the model's built-in generate_cam method.
    This works with EnhancedPEFTHyperspectralGCViT model's existing CAM implementation.
    
    Args:
        model: model with ``generate_cam()`` method.
        input_tensor: Input tensor ``[B, C, H, W]``.
        target_class: Specific class index to visualize (``None`` for all class).
        device: Device to run on (``cuda`` / ``cpu``).
    
    Returns:
        cam_image: CAM visualization array normalized to ``[0, 255]``.
        pred_class: Predicted class index.
        true_cam: Raw CAM tensor from model.
    """

    model.eval()
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Get model predictions AND CAM (using your model's built-in method)
        outputs, cams = model(input_tensor, return_cam=True)  # FIXED: Use your model's forward method with return_cam
        pred_class = torch.argmax(outputs, dim=1).item()
        
        if len(cams.shape) != 4:
            raise ValueError(f"CAM has invalid shape {cams.shape}, expected [B, num_classes, H, W].")

        # Select target class for CAM (use predicted if not specified)
        cam_class = target_class if target_class is not None else pred_class

        num_classes = cams.shape[1]
        if cam_class >= num_classes:
            cam_class = 0    # fallback to 1st class if invalid.

        true_cam = cams[0, cam_class, :, :].cpu().numpy()  # Take first batch item, target class
        
        # Normalize CAM to [0, 255] for visualization
        cam_image = (true_cam - true_cam.min()) / (true_cam.max() - true_cam.min() + 1e-8)
        cam_image = (cam_image * 255).astype(np.uint8)
        
        # Resize to match input spatial dimensions (keep aspect ratio)
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_image = cv2.resize(cam_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return cam_image, pred_class, true_cam
