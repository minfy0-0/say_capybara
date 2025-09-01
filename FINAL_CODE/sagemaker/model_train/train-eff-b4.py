# train.py — SageMaker-ready (0.90 target)
# - Backbone switch: densenet121 / efficientnet_b3 / efficientnet_b4 / convnext_tiny / resnet50
# - Dropout knob per-backbone
# - 2-stage training (freeze head -> unfreeze all)
# - AMP + Cosine LR + Grad Clip
# - Options: BN freeze, Mixup(early), TTA(hflip), EarlyStop by val loss + min_delta
# - Class imbalance: WeightedRandomSampler OR class-weighted loss
# - Metrics per epoch (train/val/test): ACC/F1/AUC/LOSS CSV + curves PNG  # NEW (train/test 추가)
# - Confusion matrix PNG (val/test, Blues colormap)                      # NEW (test 추가 + 파란색)

import os, json, argparse, random, csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, accuracy_score
)

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

def get_data_loaders(batch_size, num_workers, img_size, use_weighted_sampler=False):
    ch = detect_channels()
    tfm_train, tfm_eval = build_transforms(img_size)

    ds_train = datasets.ImageFolder(ch["train"], tfm_train)
    ds_val   = datasets.ImageFolder(ch["val"],   tfm_eval)
    ds_test  = datasets.ImageFolder(ch["test"],  tfm_eval)

    print(f"[DATA] train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")
    print(f"[DATA] classes={ds_train.classes}")

    if use_weighted_sampler:
        # 클래스 불균형 완화 (중복 로딩 없이 ds_train 재사용)
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

# -------- Backbone switcher (Dropout knob supported) --------
def get_model(backbone: str, num_classes: int, dropout_p: float = 0.1):
    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        m = models.densenet121(weights=weights)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes)) if dropout_p>0 else nn.Linear(in_f, num_classes)
        return m

    if backbone == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT
        m = models.efficientnet_b3(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[0] = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Dropout(p=0.0)
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.DEFAULT
        m = models.efficientnet_b4(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[0] = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Dropout(p=0.0)
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        m = models.convnext_tiny(weights=weights)
        in_f = m.classifier[-1].in_features
        if dropout_p > 0:
            m.classifier = nn.Sequential(
                *list(m.classifier[:-1]),  # LayerNorm2d, Flatten 유지
                nn.Dropout(dropout_p),
                nn.Linear(in_f, num_classes)
            )
        else:
            m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        m = models.resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes)) if dropout_p>0 else nn.Linear(in_f, num_classes)
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
# Metrics & Plot helpers
# =========================
def compute_epoch_metrics(y_true, y_pred, y_prob, num_classes, want_auc=True):
    """y_true,y_pred: 1D np.array, y_prob: (N,C) np.array (softmax)"""
    if y_true.size == 0:
        return {"acc": float("nan"), "f1": float("nan"), "auc": float("nan")}
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    auc = float("nan")
    if want_auc and y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
        try:
            y_true_oh = np.zeros((len(y_true), num_classes), dtype=np.float32)
            y_true_oh[np.arange(len(y_true)), y_true] = 1.0
            auc = roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro")
        except Exception:
            pass
    return {"acc": acc, "f1": f1, "auc": auc}

def plot_metrics_curves(history_phase, out_png_path, phase_label):
    """history_phase: [{'epoch', 'loss', 'acc', 'f1', 'auc'}...] for the single phase"""
    epochs = [h["epoch"] for h in history_phase]
    fig = plt.figure(figsize=(12,8))

    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(epochs, [h["loss"] for h in history_phase])
    ax1.set_title(f"{phase_label} Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(epochs, [h["acc"] for h in history_phase])
    ax2.set_title(f"{phase_label} Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylabel("acc")

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(epochs, [h["f1"] for h in history_phase])
    ax3.set_title(f"{phase_label} F1 (macro)"); ax3.set_xlabel("epoch"); ax3.set_ylabel("f1")

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(epochs, [h["auc"] for h in history_phase])
    ax4.set_title(f"{phase_label} AUC (macro-OVR)"); ax4.set_xlabel("epoch"); ax4.set_ylabel("auc")

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)

def plot_metrics_table(best_row, out_png_path):
    fig, ax = plt.subplots(figsize=(6,2.8))
    ax.axis("off")
    data = [
        ["Best Epoch", best_row.get("epoch", "-")],
        ["Val Acc",   f"{best_row['acc']:.4f}" if not np.isnan(best_row['acc']) else "nan"],
        ["Val F1",    f"{best_row['f1']:.4f}" if not np.isnan(best_row['f1']) else "nan"],
        ["Val AUC",   f"{best_row['auc']:.4f}" if not np.isnan(best_row['auc']) else "nan"],
        ["Val Loss",  f"{best_row['loss']:.4f}" if not np.isnan(best_row['loss']) else "nan"],
    ]
    table = ax.table(cellText=data, colLabels=["Metric","Value"], loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)

def plot_confmat(cm, class_names, out_png_path, normalize=True):
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")  # NEW: 파란색
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)

    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)

# =========================
# Helpers: BN freeze, Mixup, TTA
# =========================
def set_bn_eval(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

def do_mixup(x, y, alpha: float):
    if alpha <= 0.0:
        return x, (y, y, 1.0)
    beta = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = beta * x + (1.0 - beta) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed, (y_a, y_b, beta)

@torch.no_grad()
def forward_tta(model, x, use_tta: int, amp_enabled: bool):
    if use_tta >= 2 and x.ndim == 4:
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x)
            logits_flip = model(torch.flip(x, dims=[3]))
        return (logits + logits_flip) / 2
    else:
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            return model(x)

# =========================
# Train / Eval
# =========================
def train_and_validate(model, dataloaders, dataset_sizes, criterion, optimizer,
                       num_epochs, device, model_dir, warmup_epochs=2,
                       grad_clip=1.0, early_stop_patience=7,
                       early_metric="loss", early_min_delta=0.0,
                       bn_freeze=False, tta=0, mixup_alpha=0.0, mixup_epochs=0,
                       epoch_offset=0):
    history = []
    best_acc = -1.0
    monitor_best = float("inf") if early_metric == "loss" else -1.0
    best_path = os.path.join(model_dir, "model.pth")
    os.makedirs(model_dir, exist_ok=True)

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs_no_improve = 0

    num_classes = len(dataloaders["train"].dataset.classes)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}\n----------")
        if bn_freeze:
            model.apply(set_bn_eval)

        # NEW: 매 에폭마다 test까지 평가해서 그래프/CSV에 기록
        for phase in ["train", "val", "test"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects, n_samples = 0.0, 0, 0

            # buffers for epoch metrics
            y_true_buf, y_pred_buf, y_prob_buf = [], [], []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train" and (epoch < mixup_epochs and mixup_alpha > 0):
                        inputs, (ya, yb, lam) = do_mixup(inputs, labels, mixup_alpha)
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            outputs = model(inputs)
                            loss = lam * criterion(outputs, ya) + (1.0 - lam) * criterion(outputs, yb)
                    else:
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            if phase == "train":
                                outputs = model(inputs)
                            else:
                                outputs = forward_tta(model, inputs, tta, torch.cuda.is_available())
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

                # buffers
                y_true_buf.append(labels.detach().cpu().numpy())
                y_pred_buf.append(preds.detach().cpu().numpy())

                # NEW: train/val/test 모두 AUC를 계산할 수 있도록 확률 저장
                prob = torch.softmax(outputs.detach(), dim=1).cpu().numpy()
                y_prob_buf.append(prob)

            epoch_loss = running_loss / max(1, n_samples)
            epoch_acc  = running_corrects / max(1, n_samples)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # compute epoch metrics (acc/f1/auc)
            y_true_np = np.concatenate(y_true_buf) if y_true_buf else np.array([])
            y_pred_np = np.concatenate(y_pred_buf) if y_pred_buf else np.array([])
            y_prob_np = np.concatenate(y_prob_buf) if len(y_prob_buf) > 0 else None

            # NEW: 모든 phase에서 AUC 계산
            m = compute_epoch_metrics(y_true_np, y_pred_np, y_prob_np, num_classes=num_classes, want_auc=True)
            history.append({
                "epoch": epoch_offset + epoch,
                "phase": phase,
                "loss": epoch_loss,
                "acc": m["acc"],
                "f1":  m["f1"],
                "auc": m["auc"],
            })

            if phase == "train":
                if epoch < warmup_epochs:
                    for pg in optimizer.param_groups:
                        base_lr = pg.get("initial_lr", pg["lr"])
                        pg["lr"] = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
                else:
                    scheduler.step()

            if phase == "val":
                # Save best by ACC (목표 acc 극대화)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    atomic_save(model, best_path)
                    print(f"[BEST] val acc={best_acc:.4f}")

                # Early stop by selected metric (loss/acc)
                current = epoch_loss if early_metric == "loss" else epoch_acc
                improved = (current < monitor_best - early_min_delta) if early_metric == "loss" else (current > monitor_best + early_min_delta)
                if improved:
                    monitor_best = current
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if early_stop_patience and epochs_no_improve >= early_stop_patience:
            print(f"[EARLY STOP] no improve for {early_stop_patience} epochs.")
            break

    if not os.path.isfile(best_path) or os.path.getsize(best_path) < 1024:
        atomic_save(model, best_path)
        print("[FALLBACK] saved final weights.")

    print(f"[RESULT] best_val_acc={best_acc:.6f}")
    # return history along with best
    return best_path, best_acc, history

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="densenet121",
                        choices=["densenet121","efficientnet_b3","efficientnet_b4","convnext_tiny","resnet50"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=320)  # B4 권장: 380
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # 과적합/일반화 제어 노브
    parser.add_argument("--bn-freeze", action="store_true")
    parser.add_argument("--early-metric", choices=["acc","loss"], default="loss")
    parser.add_argument("--early-min-delta", type=float, default=0.002)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--tta", type=int, default=0)                # 0(off) or 2(hflip avg)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)    # 0이면 미사용
    parser.add_argument("--mixup-epochs", type=int, default=6)
    parser.add_argument("--use-weighted-sampler", action="store_true", default=False)

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

    if args.backbone == "efficientnet_b4" and args.img_size < 360:
        print(f"[WARN] efficientnet_b4는 보통 img-size=380 권장입니다. 현재 {args.img_size}.")

    dataloaders, dataset_sizes, class_names = get_data_loaders(
        batch_size=args.batch_size, num_workers=num_cpus, img_size=args.img_size,
        use_weighted_sampler=args.use_weighted_sampler
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.backbone, num_classes=len(class_names), dropout_p=args.dropout).to(device)

    # 1단계: 헤드만 학습 (초기 안정화)
    for name, p in model.named_parameters():
        head_name = (("classifier" in name) or ("fc" in name) or ("heads" in name))
        p.requires_grad = head_name

    # 손실: 샘플러를 쓰지 않으면 클래스 가중치 적용
    if args.use_weighted_sampler:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        labels = [y for _, y in dataloaders["train"].dataset.imgs]
        cnt = Counter(labels)
        n_class = len(cnt)
        class_weights = torch.tensor([1.0 / cnt[i] for i in range(n_class)],
                                     dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=args.label_smoothing)

    # 1단계(헤드만 학습) 옵티마
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 프리즈 단계
    best_path, best_acc, hist1 = train_and_validate(
        model, dataloaders, dataset_sizes, criterion, optimizer,
        num_epochs=args.freeze_epochs, device=device, model_dir=sm_model_dir,
        warmup_epochs=min(2, args.freeze_epochs),
        early_stop_patience=None,                 # 프리즈 구간은 얼리스톱 X
        early_metric=args.early_metric, early_min_delta=args.early_min_delta,
        bn_freeze=args.bn_freeze, tta=0,          # 학습 중엔 TTA 비활성
        mixup_alpha=args.mixup_alpha, mixup_epochs=args.mixup_epochs,
        epoch_offset=0
    )

    # 2단계: 전층 unfreeze 후 미세튜닝
    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_path, best_acc, hist2 = train_and_validate(
        model, dataloaders, dataset_sizes, criterion, optimizer,
        num_epochs=args.epochs - args.freeze_epochs, device=device, model_dir=sm_model_dir,
        warmup_epochs=2,
        early_stop_patience=args.early_patience,
        early_metric=args.early_metric, early_min_delta=args.early_min_delta,
        bn_freeze=args.bn_freeze, tta=args.tta,
        mixup_alpha=args.mixup_alpha, mixup_epochs=args.mixup_epochs,
        epoch_offset=args.freeze_epochs
    )

    # === History merge & save ===
    history = hist1 + hist2
    csv_path = os.path.join(sm_output_data, "metrics_per_epoch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","phase","loss","acc","f1","auc"])
        w.writeheader()
        for row in history:
            w.writerow(row)
    print(f"[METRICS] per-epoch csv -> {csv_path}")

    # === 곡선 (train/val/test 각각 PNG) ===  # NEW
    for phase in ["train","val","test"]:
        phase_hist = [r for r in history if r["phase"]==phase]
        if len(phase_hist) > 0:
            curves_png = os.path.join(sm_output_data, f"metrics_curves_{phase}.png")
            plot_metrics_curves(phase_hist, curves_png, phase.upper())
            print(f"[PLOT] metrics curves ({phase}) -> {curves_png}")

    # === Val best table (기존 유지) ===
    val_hist = [r for r in history if r["phase"]=="val"]
    if len(val_hist) > 0:
        best_row = max(val_hist, key=lambda r: (r["acc"] if not np.isnan(r["acc"]) else -1))
        table_png = os.path.join(sm_output_data, "metrics_table_val_best.png")
        plot_metrics_table(best_row, table_png)
        print(f"[PLOT] metrics table -> {table_png}")

    # === Confusion Matrix (VAL & TEST, best model 로드) ===  # NEW (test 추가)
    try:
        model.load_state_dict(torch.load(os.path.join(sm_model_dir, "model.pth"), map_location=device))
        model.eval()
    except Exception as e:
        print("[WARN] could not load best model for confusion matrix:", e)

    def _eval_confmat(dataloader, name):
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device, non_blocking=True)
                logits = forward_tta(model, x, args.tta, torch.cuda.is_available())
                preds = logits.argmax(1).cpu().numpy()
                y_pred_all.append(preds)
                y_true_all.append(y.numpy())
        if len(y_true_all) > 0:
            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)
            cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(class_names))))
            cm_png = os.path.join(sm_output_data, f"confusion_matrix_{name}.png")
            plot_confmat(cm, class_names, cm_png, normalize=True)
            print(f"[PLOT] confusion matrix ({name}) -> {cm_png}")

    _eval_confmat(dataloaders["val"],  "val")   # 기존
    _eval_confmat(dataloaders["test"], "test")  # NEW

    # summary json
    metrics = {
        "best_val_acc": float(best_acc),
        "classes": class_names,
        "img_size": args.img_size,
        "backbone": args.backbone,
        "dropout": args.dropout,
        "bn_freeze": args.bn_freeze,
        "tta": args.tta,
        "mixup_alpha": args.mixup_alpha,
        "mixup_epochs": args.mixup_epochs,
        "early_metric": args.early_metric,
        "early_min_delta": args.early_min_delta,
        "early_patience": args.early_patience,
        "use_weighted_sampler": args.use_weighted_sampler,
    }
    with open(os.path.join(sm_output_data, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[METRICS] saved -> {os.path.join(sm_output_data, 'metrics.json')}")

if __name__ == "__main__":
    main()
