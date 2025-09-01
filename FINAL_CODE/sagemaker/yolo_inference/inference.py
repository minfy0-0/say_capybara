import os
import glob
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
import torch
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_result_image(result, dst_path: Path):
    img = result.plot()  # BGR ndarray
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), img)

def save_yolo_txt(result, dst_txt_path: Path):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return False
    xywhn = boxes.xywhn.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (cx, cy, w, h), c in zip(xywhn, cls_ids)]
    dst_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True

def run_inference(args):
    logging.info("===== YOLO 배치 추론 시작 =====")
    input_data_path = args.data_dir
    in_model_dir = args.model_dir
    output_data_dir = args.output_dir
    out_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    model_file_path = os.path.join(in_model_dir, "best.pt")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found at {model_file_path}")

    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(out_model_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"YOLO 모델 로드: {model_file_path} (device={device})")
    model = YOLO(model_file_path)

    total_files, detected_count, not_detected_count = 0, 0, 0

    for split in ["train", "val", "test"]:
        split_path = os.path.join(input_data_path, split)
        if not os.path.isdir(split_path):
            logging.info(f"Split not found, skip: {split_path}")
            continue

        class_dirs = [os.path.join(split_path, d) for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        root_image_globs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
            root_image_globs.extend(glob.glob(os.path.join(split_path, ext)))

        # ── 루트 바로 아래 이미지 처리 ─────────────────────────────────────────
        if root_image_globs:
            class_name = "_root"
            for img_path in root_image_globs:
                total_files += 1
                try:
                    results = model.predict(
                        source=img_path,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        device=device,
                        save=True,                      # ← 크롭 저장을 확실히 하려면 True
                        save_crop=True,                 # ← 크롭 저장
                        project=os.path.join(output_data_dir, "predictions"),
                        name="yolo",
                        exist_ok=True,                  # ← 재실행 시 폴더 충돌 방지
                        verbose=False
                    )
                    r = results[0]
                    if r.boxes is None or len(r.boxes) == 0:
                        not_detected_count += 1
                        nd_dir = Path(output_data_dir) / "no_detections" / split / class_name
                        nd_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(img_path, nd_dir / os.path.basename(img_path))
                    else:
                        detected_count += 1
                        txt_dir = Path(output_data_dir) / "detected_labels" / split / class_name
                        dst_txt = txt_dir / (Path(img_path).stem + ".txt")
                        save_yolo_txt(r, dst_txt)
                        img_dir = Path(output_data_dir) / "detected_images" / split / class_name
                        dst_img = img_dir / os.path.basename(img_path)
                        save_result_image(r, dst_img)
                except Exception as e:
                    logging.error(f"[root] '{img_path}' 처리 중 예외: {e}")

        # ── 클래스 폴더 이미지 처리 ───────────────────────────────────────────
        for class_path in class_dirs:
            class_name = os.path.basename(class_path)
            image_paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                image_paths.extend(glob.glob(os.path.join(class_path, ext)))

            for img_path in image_paths:
                total_files += 1
                try:
                    results = model.predict(
                        source=img_path,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        device=device,
                        save=True,                      # ← 추가
                        save_crop=True,                 # ← 추가
                        project=os.path.join(output_data_dir, "predictions"),
                        name="yolo",
                        exist_ok=True,
                        verbose=False
                    )
                    r = results[0]
                    if r.boxes is None or len(r.boxes) == 0:
                        not_detected_count += 1
                        nd_dir = Path(output_data_dir) / "no_detections" / split / class_name
                        nd_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(img_path, nd_dir / os.path.basename(img_path))
                    else:
                        detected_count += 1
                        txt_dir = Path(output_data_dir) / "detected_labels" / split / class_name
                        dst_txt = txt_dir / (Path(img_path).stem + ".txt")
                        save_yolo_txt(r, dst_txt)
                        img_dir = Path(output_data_dir) / "detected_images" / split / class_name
                        dst_img = img_dir / os.path.basename(img_path)
                        save_result_image(r, dst_img)
                except Exception as e:
                    logging.error(f"'{img_path}' 처리 중 예외: {e}")

    logging.info(f"총 처리: {total_files} | 탐지: {detected_count} | 미탐지: {not_detected_count}")
    logging.info(f"라벨(txt) 저장 위치: {Path(output_data_dir) / 'detected_labels'}")
    logging.info(f"이미지 저장 위치: {Path(output_data_dir) / 'detected_images'}")
    with open(os.path.join(output_data_dir, "DONE.txt"), "w") as f:
        f.write(f"finished at {datetime.utcnow().isoformat()}Z\n")

    shutil.copy2(model_file_path, os.path.join(out_model_dir, "best.pt"))
    logging.info(f"Saved model artifact to: {os.path.join(out_model_dir, 'best.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--model-dir",  type=str, default=os.environ.get("SM_CHANNEL_MODEL"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--conf",       type=float, default=0.25)
    args, _ = parser.parse_known_args()

    logging.info(f"[ARGS] data={args.data_dir}, model={args.model_dir}, output={args.output_dir}")
    run_inference(args)
