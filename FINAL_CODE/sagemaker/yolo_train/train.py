import os
import argparse
import shutil
from ultralytics import YOLO
import torch

def train_yolo(args):
    """YOLOv8 학습 함수 (SageMaker 및 로컬 모두 호환)"""

    # --- 1. 환경 확인 ---
    print("--- 환경 정보 ---")
    print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch 버전: {torch.__version__}")
    print("--------------------")

    # --- 2. data.yaml 경로 ---
    data_yaml_path = "./data.yaml"

    # --- 3. 모델 학습 ---
    print("\n--- 모델 학습 시작 ---")
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.sm_model_dir,  # /opt/ml/model
        name='skin_disease_sagemaker_run'
    )

    print("--- 모델 학습 완료 ---")
    print(f"📁 결과 저장 위치: {results.save_dir}")

    # --- 4. best.pt 복사 (SageMaker 규약: /opt/ml/model/ 바로 아래) ---
    weights_dir = os.path.join(results.save_dir, "weights")
    best_model = os.path.join(weights_dir, "best.pt")
    if os.path.exists(best_model):
        # SageMaker가 아카이브하는 위치
        dst_path = os.path.join(args.sm_model_dir, "best.pt")
        shutil.copy2(best_model, dst_path)
        print(f"✅ best.pt를 {dst_path} 로 복사 완료")
    else:
        print("⚠️ best.pt를 찾을 수 없음!")

    print("☁️ SageMaker 환경에서는 이 경로가 S3로 자동 업로드됩니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker 환경 변수
    parser.add_argument('--train', type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get("SM_MODEL_DIR", "./output"), dest='sm_model_dir')

    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)

    args = parser.parse_args()
    train_yolo(args)