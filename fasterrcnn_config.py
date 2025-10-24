"""
Faster R-CNN Training with Flexible Dataset Configuration
Supports any COCO-format dataset (CVAT exports, custom annotations)
"""

import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.dataset2 import UniversalCOCODataset, collate_fn
from utils.visualization import plot_detection_history, visualize_predictions_original
from utils.evaluation import evaluate_detection, print_results_table
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import random

# ============================================================================
# DATASET CONFIGURATION - 여기만 수정하면 됩니다!
# ============================================================================

CONFIG = {
    # Dataset paths
    'data_root': './data/coco_val_3',           # 데이터셋 루트 경로
    'annotation_file': 'annotations/instances_merged2.json',  # 어노테이션 파일
    'image_folder': 'images',                # 이미지 폴더
    
    # Class configuration (WITHOUT '__background__')
    'class_names': ['with_helmet', 'without_helmet'],         # 학습할 클래스들
    
    # Training parameters
    'resize_size': 640,                       # 입력 이미지 크기
    'batch_size': 4,                         # 배치 크기
    'num_epochs': 20,                         # 에폭 수
    'learning_rate': 1e-4,                  # 학습률
    'weight_decay': 3e-4,                    # Weight decay
    'lr_step_size': 6,                        # LR scheduler step
    'lr_gamma': 0.3,                          # LR scheduler gamma
    
    # Dataset splits
    'max_train_images': 1000,                 # None = all images, 전체 이미지를 학습하려면
    'max_val_images': 200,
    'max_test_images': 200,
    
    # Model saving
    'model_save_path': 'safety_helmet_model2.pth',
    'best_model_path': 'safety_helmet_best2.pth',
}

# ============================================================================
# EXAMPLE CONFIGURATIONS FOR DIFFERENT DATASETS
# ============================================================================

# Example 1: COCO Person & Car (default)
# CONFIG = {
#     'data_root': './data/coco_val',
#     'annotation_file': 'annotations/instances_val2017.json',
#     'image_folder': 'val2017',
#     'class_names': ['person', 'car'],
#     ...
# }

# Example 2: CVAT Animal Dataset
# CONFIG = {
#     'data_root': './data/animal_dataset',
#     'annotation_file': 'annotations.json',
#     'image_folder': 'images',
#     'class_names': ['cat', 'dog', 'bird'],
#     ...
# }

# Example 3: Custom Object Detection
# CONFIG = {
#     'data_root': './data/my_custom_dataset',
#     'annotation_file': 'instances.json',
#     'image_folder': 'images',
#     'class_names': ['defect', 'crack', 'rust'],
#     ...
# }

# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'xpu')
print(f"Using device: {device}")


def get_fasterrcnn_model(num_classes):
    """Get Faster R-CNN model with custom number of classes"""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # Freeze backbone
    #for param in model.backbone.parameters():
    #    param.requires_grad = False # backbone 그레이드 막았음
    
    # (Optional) Set backbone to eval mode to freeze BN statistics
    #model.backbone.eval()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # num_classes includes background
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    losses = []
    
    for images, targets in tqdm(loader, desc=f"Epoch {epoch}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Filter empty targets
        valid = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
        if not valid:
            continue
        
        images, targets = zip(*valid)
        
        try:
            loss_dict = model(list(images), list(targets))
            total_loss = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(total_loss.item())
        except Exception as e:
            print(f"Training error: {e}")
            continue
    
    return sum(losses) / len(losses) if losses else 0


@torch.no_grad()
def validate(model, loader, calc_loss=False):
    """Validate model - returns all data"""
    model.eval()
    
    predictions_all = []
    targets_all = []
    images_all = []
    losses = []
    times = []
    
    for images, targets in tqdm(loader, desc="Validating"):
        images_gpu = [img.to(device) for img in images]
        targets_gpu = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
        
        # Calculate loss
        if calc_loss:
            valid = [(img, tgt) for img, tgt in zip(images_gpu, targets_gpu) 
                    if len(tgt['boxes']) > 0]
            if valid:
                imgs_loss, tgts_loss = zip(*valid)
                model.train()
                try:
                    loss_dict = model(list(imgs_loss), list(tgts_loss))
                    losses.append(sum(loss for loss in loss_dict.values()).item())
                except:
                    pass
                model.eval()
        
        # Get predictions
        start = time.time()
        preds = model(images_gpu)
        times.append(time.time() - start)
        
        # Store everything
        predictions_all.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
        targets_all.extend(targets)
        images_all.extend([img.cpu() for img in images])
    
    # Calculate metrics
    metrics = evaluate_detection(predictions_all, targets_all, score_threshold=0.5)
    avg_loss = sum(losses) / len(losses) if losses else 0
    avg_time = sum(times) / len(times) if times else 0
    
    return metrics, avg_loss, avg_time, predictions_all, targets_all, images_all


def main():
    print("="*60)
    print("Faster R-CNN Training with Flexible Configuration")
    print("="*60)
    
    # Print configuration
    print("\nDataset Configuration:")
    print(f"  Root: {CONFIG['data_root']}")
    print(f"  Annotations: {CONFIG['annotation_file']}")
    print(f"  Images: {CONFIG['image_folder']}")
    print(f"  Classes: {CONFIG['class_names']}")
    print(f"  Number of classes: {len(CONFIG['class_names'])}")
    
    # Calculate num_classes (including background)
    num_classes = len(CONFIG['class_names']) + 1  # +1 for background
    class_names_with_bg = ['__background__'] + CONFIG['class_names']
    
    print(f"  Total classes (with background): {num_classes}")
    print(f"  Class names: {class_names_with_bg}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = UniversalCOCODataset(
        root=CONFIG['data_root'],
        annotation_file=CONFIG['annotation_file'],
        image_folder=CONFIG['image_folder'],
        class_names=CONFIG['class_names'],
        split='train',
        max_images=CONFIG['max_train_images'],
        resize_size=CONFIG['resize_size']
    )
    
    val_dataset = UniversalCOCODataset(
        root=CONFIG['data_root'],
        annotation_file=CONFIG['annotation_file'],
        image_folder=CONFIG['image_folder'],
        class_names=CONFIG['class_names'],
        split='val',
        max_images=CONFIG['max_val_images'],
        resize_size=CONFIG['resize_size']
    )
    
    test_dataset = UniversalCOCODataset(
        root=CONFIG['data_root'],
        annotation_file=CONFIG['annotation_file'],
        image_folder=CONFIG['image_folder'],
        class_names=CONFIG['class_names'],
        split='test',
        max_images=CONFIG['max_test_images'],
        resize_size=CONFIG['resize_size']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, collate_fn=collate_fn)
    
    # Small subset for quick train metric check
    train_subset = torch.utils.data.Subset(train_dataset, 
                                          range(min(20, len(train_dataset))))
    train_check_loader = DataLoader(train_subset, batch_size=4, 
                                    shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("\nCreating Faster R-CNN model...")
    model = get_fasterrcnn_model(num_classes=num_classes).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), 
                          lr=CONFIG['learning_rate'], 
                          weight_decay=CONFIG['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                             step_size=CONFIG['lr_step_size'], 
                                             gamma=CONFIG['lr_gamma'])
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'val_precision': [],
        'val_recall': []
    }
    
    best_iou = 0
    
    # Training loop
    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    print("-" * 60)
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        history['train_loss'].append(train_loss)
        
        # Quick train check
        train_metrics, _, _, _, _, _ = validate(model, train_check_loader, calc_loss=False)
        history['train_iou'].append(train_metrics['avg_iou'])
        
        # Validation
        val_metrics, val_loss, val_time, _, _, _ = validate(model, val_loader, calc_loss=True)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_metrics['avg_iou'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_metrics['avg_iou']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val IoU:   {val_metrics['avg_iou']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"  Inference: {val_time:.3f}s/batch")
        
        if val_metrics['avg_iou'] > best_iou:
            best_iou = val_metrics['avg_iou']
            torch.save(model.state_dict(), CONFIG['best_model_path'])
            print(f"  ✓ New best IoU: {best_iou:.4f}")
        
        lr_scheduler.step()
    
    # Visualize training history
    print("\nGenerating training history plots...")
    plot_detection_history(history, "Faster R-CNN Training")
    
    # Test evaluation
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(CONFIG['best_model_path']))
    
    print("Evaluating on test set...")
    test_metrics, test_loss, test_time, test_preds, test_targets, test_images = validate(
        model, test_loader, calc_loss=True
    )
    
    # Visualize random samples
    n_samples = min(10, len(test_targets))
    indices = random.sample(range(len(test_targets)), n_samples)
    
    sample_images = [test_images[i] for i in indices]
    sample_targets = [test_targets[i] for i in indices]
    sample_preds = [test_preds[i] for i in indices]
    
    print("\nVisualizing predictions...")
    visualize_predictions_original(
        sample_images, sample_targets, sample_preds,
        test_dataset.img_dir, 
        class_names=class_names_with_bg,
        num_samples=n_samples, 
        score_threshold=0.5
    )
    
    # Print results
    print_results_table(test_metrics, "Faster R-CNN")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best Val IoU: {best_iou:.4f}")
    print(f"Test IoU:     {test_metrics['avg_iou']:.4f}")
    print(f"Test Loss:    {test_loss:.4f}")
    print("="*60)
    
    # Save final model
    torch.save(model, CONFIG['model_save_path'])
    print(f"\n✓ Model saved to {CONFIG['model_save_path']}")


if __name__ == "__main__":
    main()