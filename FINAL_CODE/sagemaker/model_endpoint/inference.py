# inference.py — Hardcoded serving (no meta files)
# - BACKBONE / IMG_SIZE만 맞추면 바로 동작
# - labels.json이 있으면 사용, 없으면 class_0.. 자동 생성
# - 바이너리(image/jpeg, application/x-image) / JSON(image_b64, raw tensor) 모두 처리

import io, os, sys, json, base64, traceback
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

# =========================
# 🔧 하드코딩 지점 (학습 설정과 "동일하게")
# =========================
BACKBONE = "efficientnet_b4"   # 예: "densenet121", "efficientnet_b4", "efficientnet_b3", "convnext_tiny", "resnet50"
IMG_SIZE = 380                 # B4 권장 380 (DenseNet 224/320 등 학습값과 동일해야 함)
DROPOUT_P = 0.2               # 학습 때 head에 썼다면 맞춰줘도 됨(미사용이면 0)

torch.set_num_threads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _log_ex(e, msg=""):
    print(f"[SERVE][ERROR] {msg}: {e}", file=sys.stderr)
    traceback.print_exc()

# -------------------------
# 모델 빌더 (백본별 헤드 위치 주의)
# -------------------------
def _build_model(backbone: str, num_classes: int, dropout_p: float = 0.0):
    if backbone == "densenet121":
        m = models.densenet121(weights=None)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes)) if dropout_p>0 else nn.Linear(in_f, num_classes)
        return m

    if backbone == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[0] = nn.Dropout(dropout_p)
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "efficientnet_b4":
        m = models.efficientnet_b4(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[0] = nn.Dropout(dropout_p)
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        in_f = m.classifier[-1].in_features
        if dropout_p > 0:
            m.classifier = nn.Sequential(*list(m.classifier[:-1]), nn.Dropout(dropout_p), nn.Linear(in_f, num_classes))
        else:
            m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    if backbone == "resnet50":
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_f, num_classes)) if dropout_p>0 else nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unknown backbone: {backbone}")

def _build_eval_transform(img_size: int):
    # train.py의 eval 파이프라인(Resize→CenterCrop→Normalize)과 동일해야 함
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

# =========================
# SageMaker entrypoints
# =========================
def model_fn(model_dir):
    try:
        # 1) 라벨 로드(or 생성)
        labels_path = os.path.join(model_dir, "labels.json")
        labels = None
        if os.path.exists(labels_path):
            try:
                labels = json.load(open(labels_path, "r", encoding="utf-8"))
            except Exception as e:
                print(f"[SERVE][WARN] labels.json load failed: {e}")
        # labels.json이 없거나 실패하면, 나중에 num_classes로 자동 생성할 것

        # 2) 모델 구성
        ckpt_path = os.path.join(model_dir, "model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")

        # num_classes 추론 (헤드 가중치에서 행 수)
        def _infer_num_classes(sd):
            if isinstance(sd, dict):
                for k in ("classifier.weight", "module.classifier.weight",
                          "classifier.1.weight", "module.classifier.1.weight",
                          "fc.weight", "module.fc.weight"):
                    w = sd.get(k)
                    if w is not None and hasattr(w, "shape"):
                        return int(w.shape[0])
            return len(labels) if labels else 6  # 마지막 폴백
        num_classes = _infer_num_classes(state)

        if labels is None or len(labels) != num_classes:
            print(f"[SERVE][INFO] build default labels (num_classes={num_classes})")
            labels = [f"class_{i}" for i in range(num_classes)]

        model = _build_model(BACKBONE, num_classes, DROPOUT_P)

        # 3) state_dict 정리 로드
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        clean_sd = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[7:]
            clean_sd[k] = v
        msg = model.load_state_dict(clean_sd, strict=False)
        print(f"[SERVE] load_state_dict: {msg}")

        model.to(DEVICE).eval()
        tf = _build_eval_transform(IMG_SIZE)

        # 디버그용 헤드 노름
        try:
            head = getattr(model, "classifier", None)
            if isinstance(head, nn.Sequential) and hasattr(head[1], "weight"):
                print(f"[SERVE] head.weight.norm={head[1].weight.data.norm().item():.6f}")
            elif hasattr(head, "weight"):
                print(f"[SERVE] head.weight.norm={head.weight.data.norm().item():.6f}")
        except Exception:
            pass

        print(f"[SERVE] device={DEVICE}, backbone={BACKBONE}, img_size={IMG_SIZE}, classes={num_classes}")
        return {"model": model, "labels": labels, "transform": tf}
    except Exception as e:
        _log_ex(e, "model_fn failed")
        raise

def input_fn(request_body, request_content_type):
    try:
        ct = (request_content_type or "").lower()
        if ct.startswith("image/") or ct in ("application/octet-stream","application/x-image"):
            return Image.open(io.BytesIO(request_body)).convert("RGB")
        if ct == "application/json":
            obj = json.loads(request_body)
            if isinstance(obj, dict) and "image_b64" in obj:
                b = base64.b64decode(obj["image_b64"])
                return Image.open(io.BytesIO(b)).convert("RGB")
            if isinstance(obj, list):
                arr = np.array(obj, dtype=np.float32)
                t = torch.from_numpy(arr)
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                return t
        raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        _log_ex(e, "input_fn failed")
        raise

@torch.no_grad()
def predict_fn(inputs, context):
    try:
        model  = context["model"]
        labels = context["labels"]
        tf     = context["transform"]

        if isinstance(inputs, Image.Image):
            x = tf(inputs).unsqueeze(0).to(DEVICE)
        elif torch.is_tensor(inputs):
            x = inputs.to(DEVICE)
        else:
            raise ValueError("Unsupported input type for predict_fn")

        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        k = min(5, len(probs))
        top_idx = np.argsort(probs)[::-1][:k].tolist()
        topk = [{"label": labels[i], "conf": float(probs[i])} for i in top_idx]

        return {
            "label": labels[idx],
            "conf": conf,
            "labels": labels,
            "prob_list": [float(p) for p in probs.tolist()],
            "probs": {labels[i]: float(probs[i]) for i in range(len(labels))},
            "topk": topk
        }
    except Exception as e:
        _log_ex(e, "predict_fn failed")
        raise

def output_fn(prediction, accept):
    try:
        return json.dumps(prediction), "application/json"
    except Exception as e:
        _log_ex(e, "output_fn failed")
        raise
