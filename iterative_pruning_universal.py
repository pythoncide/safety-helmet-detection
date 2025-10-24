"""
Universal Structured Pruning for Faster R-CNN (Iterative Version)
Each pruning step is based on the previous pruned model.
Logs show relative changes to the previous stage.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from utils.dataset2 import UniversalCOCODataset, collate_fn
from utils.evaluation import evaluate_detection
from tqdm import tqdm
import os
import time
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_root': './data/coco_helmet',
    'annotation_file': 'annotations/instances_helmet.json',
    'image_folder': 'images',
    'class_names': ['with_helmet', 'without_helmet'],
    'model_path': 'helmet_model.pth',

    # iterative pruning ratios
    'prune_ratios': [0.2, 0.3, 0.5],

    # fine-tuning hyperparams
    'fine_tune_epochs': 5,
    'fine_tune_lr': 0.0001,
    'fine_tune_weight_decay': 0.0005,

    # dataset options
    'resize_size': 640,
    'batch_size': 8,
    'max_train_images': 1000,
    'max_val_images': 200,

    # output
    'save_prefix': 'helmet_pruned',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'xpu')


# ============================================================================
# UTILITIES
# ============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size(model):
    torch.save(model.state_dict(), 'temp_model.pth')
    size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
    os.remove('temp_model.pth')
    return size_mb


def calculate_filter_importance(conv_layer):
    weight = conv_layer.weight.data
    importance = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
    return importance


def prune_conv_layer(conv_layer, indices_to_keep):
    new_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=len(indices_to_keep),
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=(conv_layer.bias is not None)
    )
    new_conv.weight.data = conv_layer.weight.data[indices_to_keep].clone()
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[indices_to_keep].clone()
    return new_conv


def prune_batchnorm_layer(bn_layer, indices_to_keep):
    new_bn = nn.BatchNorm2d(len(indices_to_keep))
    new_bn.weight.data = bn_layer.weight.data[indices_to_keep].clone()
    new_bn.bias.data = bn_layer.bias.data[indices_to_keep].clone()
    new_bn.running_mean = bn_layer.running_mean[indices_to_keep].clone()
    new_bn.running_var = bn_layer.running_var[indices_to_keep].clone()
    return new_bn


# ============================================================================
# STRUCTURED PRUNING FUNCTION
# ============================================================================

def structured_prune_resnet_backbone(model, prune_ratio=0.3, stage_idx=0):
    """
    Apply structured pruning to ResNet backbone (iterative pruning supported)
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {stage_idx + 1}: STRUCTURED PRUNING ({prune_ratio*100:.0f}%)")
    print(f"{'='*70}")

    pruned_layers = 0
    total_filters_before = 0
    total_filters_after = 0

    backbone = model.backbone.body

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(backbone, layer_name)
        for block_idx, block in enumerate(layer):
            for conv_idx, conv_name in enumerate(['conv1', 'conv2']):
                if not hasattr(block, conv_name):
                    continue

                conv = getattr(block, conv_name)
                bn = getattr(block, f'bn{conv_idx + 1}')
                original_filters = conv.out_channels

                if original_filters <= 64:
                    total_filters_before += original_filters
                    total_filters_after += original_filters
                    continue

                importance = calculate_filter_importance(conv)
                total_filters_before += len(importance)

                num_filters = len(importance)
                num_to_keep = int(num_filters * (1 - prune_ratio))
                num_to_keep = max(num_to_keep, 32)
                _, indices_to_keep = torch.topk(importance, num_to_keep)
                indices_to_keep = sorted(indices_to_keep.tolist())
                total_filters_after += num_to_keep

                new_conv = prune_conv_layer(conv, indices_to_keep)
                setattr(block, conv_name, new_conv)

                new_bn = prune_batchnorm_layer(bn, indices_to_keep)
                setattr(block, f'bn{conv_idx + 1}', new_bn)

                next_conv_name = f'conv{conv_idx + 2}'
                if hasattr(block, next_conv_name):
                    next_conv = getattr(block, next_conv_name)
                    new_next_conv = nn.Conv2d(
                        in_channels=len(indices_to_keep),
                        out_channels=next_conv.out_channels,
                        kernel_size=next_conv.kernel_size,
                        stride=next_conv.stride,
                        padding=next_conv.padding,
                        bias=(next_conv.bias is not None)
                    )
                    new_next_conv.weight.data = next_conv.weight.data[:, indices_to_keep, :, :].clone()
                    if next_conv.bias is not None:
                        new_next_conv.bias.data = next_conv.bias.data.clone()
                    setattr(block, next_conv_name, new_next_conv)

                pruned_layers += 1

            if hasattr(block, 'conv3'):
                total_filters_before += block.conv3.out_channels
                total_filters_after += block.conv3.out_channels

    print(f"\n{'-'*70}")
    print(f"PRUNING SUMMARY (Stage {stage_idx + 1})")
    print(f"{'-'*70}")
    print(f"  Layers pruned: {pruned_layers}")
    print(f"  Filters: {total_filters_before:,} → {total_filters_after:,}")
    print(f"  Filters removed: {total_filters_before - total_filters_after:,}")
    print(f"  Actual pruning ratio (relative): {1 - (total_filters_after / total_filters_before):.2%}")
    print(f"{'='*70}\n")

    return model


# ============================================================================
# EVALUATION & FINE-TUNING
# ============================================================================

@torch.no_grad()
def evaluate_model(model, loader, description="Model", measure_speed=False):
    model.eval()
    predictions_all, targets_all, inference_times = [], [], []

    for images, targets in tqdm(loader, desc=f"Evaluating {description}"):
        images_gpu = [img.to(device) for img in images]
        if measure_speed:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

        preds = model(images_gpu)

        if measure_speed:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_times.append(time.time() - start)

        predictions_all.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
        targets_all.extend(targets)

    metrics = evaluate_detection(predictions_all, targets_all, score_threshold=0.5)
    if measure_speed and inference_times:
        avg_time = np.mean(inference_times)
        metrics['avg_inference_time'] = avg_time
        metrics['fps'] = 1.0 / avg_time
    return metrics


def fine_tune_pruned_model(model, train_loader, val_loader, epochs=5, lr=0.0001, weight_decay=0.0005):
    print(f"\n{'='*70}")
    print(f"FINE-TUNING ({epochs} epochs)")
    print(f"{'='*70}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    best_iou = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            if not any(len(t['boxes']) > 0 for t in targets):
                continue
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(total_loss.item())
        scheduler.step()

        val_metrics = evaluate_model(model, val_loader, f"Fine-tune Epoch {epoch}")
        print(f"Epoch {epoch}: Loss={np.mean(losses):.4f}, mIoU={val_metrics['avg_iou']:.4f}, "
              f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}")
        if val_metrics['avg_iou'] > best_iou:
            best_iou = val_metrics['avg_iou']
    print(f"{'='*70}\n")
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("Universal Structured Pruning for Faster R-CNN (Iterative)")
    print("="*70)

    num_classes = len(CONFIG['class_names']) + 1

    # Datasets
    val_dataset = UniversalCOCODataset(
        root=CONFIG['data_root'], annotation_file=CONFIG['annotation_file'],
        image_folder=CONFIG['image_folder'], class_names=CONFIG['class_names'],
        split='val', max_images=CONFIG['max_val_images'], resize_size=CONFIG['resize_size']
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, collate_fn=collate_fn)
    train_dataset = UniversalCOCODataset(
        root=CONFIG['data_root'], annotation_file=CONFIG['annotation_file'],
        image_folder=CONFIG['image_folder'], class_names=CONFIG['class_names'],
        split='train', max_images=CONFIG['max_train_images'], resize_size=CONFIG['resize_size']
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate_fn)

    # Load model
    model_path = CONFIG['model_path']
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)

    original_metrics = evaluate_model(model, val_loader, "Original", measure_speed=True)
    original_params = count_parameters(model)
    original_size = get_model_size(model)

    print(f"\nOriginal Model: Params={original_params:,}, Size={original_size:.2f}MB, "
          f"mIoU={original_metrics['avg_iou']:.4f}")

    previous_model = model
    previous_params = original_params
    previous_size = original_size

    for stage_idx, ratio in enumerate(CONFIG['prune_ratios']):
        pruned_model = structured_prune_resnet_backbone(previous_model, prune_ratio=ratio, stage_idx=stage_idx)

        before_params = count_parameters(previous_model)
        after_params = count_parameters(pruned_model)
        before_size = get_model_size(previous_model)
        after_size = get_model_size(pruned_model)

        print(f"\n--- Stage {stage_idx + 1} Summary (relative) ---")
        print(f"Params: {before_params:,} → {after_params:,} ({100*(1 - after_params/before_params):.1f}%↓)")
        print(f"Size: {before_size:.2f}MB → {after_size:.2f}MB ({100*(1 - after_size/before_size):.1f}%↓)")

        metrics_before = evaluate_model(pruned_model, val_loader, f"Stage {stage_idx+1} Before FT", measure_speed=True)

        pruned_model = fine_tune_pruned_model(
            pruned_model, train_loader, val_loader,
            epochs=CONFIG['fine_tune_epochs'],
            lr=CONFIG['fine_tune_lr'],
            weight_decay=CONFIG['fine_tune_weight_decay']
        )

        metrics_after = evaluate_model(pruned_model, val_loader, f"Stage {stage_idx+1} After FT", measure_speed=True)

        print(f"\nResults (Stage {stage_idx + 1}): mIoU {metrics_before['avg_iou']:.4f} → {metrics_after['avg_iou']:.4f}")
        torch.save(pruned_model, f"{CONFIG['save_prefix']}_{int(ratio*100)}.pth")

        previous_model = pruned_model
        previous_params = after_params
        previous_size = after_size

    print("\nAll pruning stages complete.\n")


if __name__ == "__main__":
    main()
