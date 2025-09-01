# train.py — (SageMaker-ready, boosted, backbone switch incl. Vision Transformer)
import os, json, argparse, math, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, classification_report
import seaborn as sns

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def detect_channels():
    ch = {}
    env = os.environ
    if "SM_CHANNEL_TRAIN" in env and "SM_CHANNEL_VAL" in env:
        ch["train"] = env["SM_CHANNEL_TRAIN"]
        ch["val"]   = env["SM_CHANNEL_VAL"]
        ch["test"]  = env.get("SM_CHANNEL_TEST", env["SM_CHANNEL_VAL"])
        mode = "multi"
    elif "SM_CHANNEL_DATA" in env:
        root = env["SM_CHANNEL_DATA"]
        ch["train"] = os.path.join(root, "train")
        ch["val"]   = os.path.join(root, "val")
        ch["test"]  = os.path.join(root, "test")
        mode = "root"
    else:
        raise RuntimeError("No SageMaker channels found.")
    print(f"[CHANNEL] mode={mode} → {ch}")
    return ch

def build_transforms(img_size: int):
    # 피부질환: 과한 색 왜곡 X, 약한 기하/밝기/대비 권장
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), value=0)
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    return tfm_train, tfm_eval

def get_data_loaders(batch_size, num_workers, img_size, use_weighted_sampler=True):
    ch = detect_channels()
    tfm_train, tfm_eval = build_transforms(img_size)

    ds_train = datasets.ImageFolder(ch["train"], tfm_train)
    ds_val   = datasets.ImageFolder(ch["val"],   tfm_eval)
    ds_test  = datasets.ImageFolder(ch["test"],  tfm_eval)

    print(f"[DATA] train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")
    print(f"[DATA] classes={ds_train.classes}")

    if use_weighted_sampler:
        labels = [y for _, y in ds_train.imgs]
        counts = Counter(labels)
        n_class = len(counts)
        class_weights = torch.tensor([1.0 / counts[i] for i in range(n_class)], dtype=torch.float)
        sample_weights = [class_weights[y].item() for _, y in ds_train.imgs]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    dl_val  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dataloaders   = {"train": dl_train, "val": dl_val, "test": dl_test}
    dataset_sizes = {"train": len(ds_train), "val": len(ds_val), "test": len(ds_test)}
    class_names   = ds_train.classes
    return dataloaders, dataset_sizes, class_names

# -------- Backbone switcher --------
def get_model(backbone: str, num_classes: int, dropout_p: float = 0.1):
    if backbone == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        m = models.vit_b_16(weights=weights)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes))
        return m

    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        m = models.densenet121(weights=weights)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes))
        return m

    if backbone == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT
        m = models.efficientnet_b3(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[0] = nn.Dropout(dropout_p)
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.DEFAULT
        m = models.efficientnet_b4(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[0] = nn.Dropout(dropout_p)
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        m = models.convnext_tiny(weights=weights)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        m = models.resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unknown backbone: {backbone}")

def atomic_save(model, path):
    tmp = f"{path}.tmp"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    assert size > 1024, f"Saved file too small ({size} bytes)."
    os.replace(tmp, path)
    print(f"[SAVE] -> {path} ({os.path.getsize(path)} bytes)")

# =========================
# Train / Eval
# =========================
@torch.no_grad()
def evaluate_metrics(model, dl, device, class_names, phase="test"):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    
    running_loss, n_samples = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    # Calculate Loss and Accuracy
    avg_loss = running_loss / max(1, n_samples)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # F1 Score
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # AUC
    try:
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        else:
            auc_score = 0.0
            print(f"[WARNING] AUC could not be calculated for {phase} set. Not enough samples per class.")
    except ValueError:
        auc_score = 0.0
        print(f"[WARNING] AUC could not be calculated for {phase} set. Not enough samples per class.")
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {phase}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(os.environ.get("SM_OUTPUT_DATA_DIR", "."), f"confusion_matrix_{phase}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[METRICS] Confusion Matrix saved to {cm_path}")

    # Plot AUC/ROC Curves
    plt.figure(figsize=(10, 8))
    num_classes = len(class_names)
    all_labels_one_hot = np.zeros((len(all_labels), num_classes))
    all_labels_one_hot[np.arange(len(all_labels)), all_labels] = 1
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_one_hot[:, i], np.array(all_probs)[:, i])
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc_score(all_labels_one_hot[:, i], np.array(all_probs)[:, i]):.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {phase}')
    plt.legend(loc="lower right")
    auc_path = os.path.join(os.environ.get("SM_OUTPUT_DATA_DIR", "."), f"roc_curve_{phase}.png")
    plt.savefig(auc_path)
    plt.close()
    print(f"[METRICS] ROC Curve saved to {auc_path}")

    return avg_loss, accuracy, f1_macro, auc_score, class_report

def plot_metrics(history, output_dir, final_test_acc):
    df = pd.DataFrame(history)
    
    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
    plt.axhline(y=final_test_acc, color='r', linestyle='--', label=f'Final Test Accuracy ({final_test_acc:.4f})')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()
    
    print(f"[PLOTS] Loss and Accuracy plots saved to {output_dir}")

def train_and_validate(model, dataloaders, dataset_sizes, criterion, optimizer,
                       num_epochs, device, model_dir, warmup_epochs=2,
                       max_lr=None, grad_clip=1.0, early_stop_patience=7):
    best_acc = -1.0
    best_path = os.path.join(model_dir, "model.pth")
    os.makedirs(model_dir, exist_ok=True)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs_no_improve = 0
    epoch_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}\n----------")
        epoch_metrics = {'epoch': epoch}

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects, n_samples = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    if phase == "train":
                        scaler.scale(loss).backward()
                        if grad_clip is not None:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                n_samples += inputs.size(0)

            epoch_loss = running_loss / max(1, n_samples)
            epoch_acc  = running_corrects / max(1, n_samples)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc

            if phase == "train":
                if epoch < warmup_epochs:
                    for pg in optimizer.param_groups:
                        base_lr = pg.get("initial_lr", pg["lr"])
                        pg["lr"] = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
                else:
                    scheduler.step()

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    atomic_save(model, best_path)
                    print(f"[BEST] val acc={best_acc:.4f}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        epoch_history.append(epoch_metrics)

        if early_stop_patience and epochs_no_improve >= early_stop_patience:
            print(f"[EARLY STOP] no improve for {early_stop_patience} epochs.")
            break
            
    if not os.path.isfile(best_path) or os.path.getsize(best_path) < 1024:
        atomic_save(model, best_path)
        print("[FALLBACK] saved final weights.")

    print("[FINAL] model dir listing")
    for p in Path(model_dir).rglob("*"):
        try:
            sz = p.stat().st_size if p.is_file() else -1
            print(" -", p, sz)
        except Exception as e:
            print(" -", p, "?", e)

    print(f"[RESULT] best_val_acc={best_acc:.6f}")
    return best_path, best_acc, epoch_history

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="densenet121",
                        choices=["densenet121","efficientnet_b3","efficientnet_b4","convnext_tiny","resnet50", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    sm_model_dir   = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    sm_output_data = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    os.makedirs(sm_model_dir, exist_ok=True)
    os.makedirs(sm_output_data, exist_ok=True)

    try:
        num_cpus = int(os.environ.get("SM_NUM_CPUS", "2"))
    except:
        num_cpus = 2

    dataloaders, dataset_sizes, class_names = get_data_loaders(
        batch_size=args.batch_size, num_workers=num_cpus, img_size=args.img_size, use_weighted_sampler=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.backbone, num_classes=len(class_names), dropout_p=0.1).to(device)

    # 1단계: 헤드만 학습 (초기 안정화)
    for name, p in model.named_parameters():
        head_name = (("classifier" in name) or ("fc" in name) or ("heads" in name))
        p.requires_grad = head_name

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 프리즈 단계
    best_path, best_acc, hist_1 = train_and_validate(
        model, dataloaders, dataset_sizes, criterion, optimizer,
        num_epochs=args.freeze_epochs, device=device, model_dir=sm_model_dir,
        warmup_epochs=min(2, args.freeze_epochs), early_stop_patience=None
    )

    # 2단계: 전층 unfreeze 후 미세튜닝
    for p in model.parameters():
        p.requires_grad = True
    
    print("[INFO] Unfreezing all layers and reducing learning rate for fine-tuning.")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr / 20, weight_decay=args.weight_decay)

    best_path, best_acc, hist_2 = train_and_validate(
        model, dataloaders, dataset_sizes, criterion, optimizer,
        num_epochs=args.epochs - args.freeze_epochs, device=device, model_dir=sm_model_dir,
        warmup_epochs=2, early_stop_patience=7
    )
    
    # 총 학습 기록 합치기
    total_history = hist_1 + hist_2
    df_history = pd.DataFrame(total_history)
    df_history.to_csv(os.path.join(sm_output_data, "training_history.csv"), index=False)
    print(f"[HISTORY] Training history saved to {os.path.join(sm_output_data, 'training_history.csv')}")

    # 최종 평가 및 지표 계산
    best_model = get_model(args.backbone, num_classes=len(class_names), dropout_p=0.1).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    
    print("\n[FINAL EVALUATION] Evaluating on test set...")
    test_loss, test_acc, test_f1, test_auc, test_report = evaluate_metrics(best_model, dataloaders["test"], device, class_names, phase="test")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score (macro): {test_f1:.4f}")
    print(f"Test AUC (macro): {test_auc:.4f}")
    print("Classification Report:")
    print(test_report)

    # 메트릭 저장
    metrics = {
        "best_val_acc": float(best_acc),
        "test_acc": float(test_acc),
        "test_f1_score": float(test_f1),
        "test_auc_score": float(test_auc),
        "test_loss": float(test_loss),
        "classes": class_names,
        "img_size": args.img_size,
        "backbone": args.backbone
    }
    with open(os.path.join(sm_output_data, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[METRICS] saved -> {os.path.join(sm_output_data, 'metrics.json')}")

    # 그래프 시각화
    plot_metrics(total_history, sm_output_data, test_acc)

if __name__ == "__main__":
    main()