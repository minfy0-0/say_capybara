import io
import os
import base64
import json
from PIL import Image
import torch
from ultralytics import YOLO
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SageMaker가 엔드포인트 시작할 때 실행: 모델 로드
def model_fn(model_dir):
    """
    SageMaker가 엔드포인트를 로드할 때 모델을 메모리에 올리는 함수입니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"디바이스 {device}에서 YOLO 모델을 로드합니다...")
    
    model_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
    model = YOLO(model_path)
    model.to(device)
    logger.info("모델 로딩 성공!")
    return model


# HTTP 요청 body를 해석
def input_fn(request_body, request_content_type):
    """
    클라이언트로부터 받은 HTTP 요청을 모델이 이해할 수 있는 형태로 변환합니다.
    """
    logger.info(f"수신된 Content-Type: {request_content_type}")
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        if "image" not in payload:
            raise ValueError("JSON 요청에는 'image' 키와 base64 문자열이 포함되어야 합니다.")
        image_bytes = base64.b64decode(payload["image"])
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    elif request_content_type in ["image/jpeg", "image/png", "image/jpg", "application/x-image"]:
        return Image.open(io.BytesIO(request_body)).convert("RGB")

    else:
        raise ValueError(f"지원하지 않는 Content-Type입니다: {request_content_type}")


# 모델 추론
def predict_fn(input_object, model):
    """
    변환된 입력 데이터(PIL Image)를 사용하여 실제 모델 추론을 수행합니다.
    """
    logger.info("모델 추론을 시작합니다...")
    results = model.predict(
        source=input_object,
        imgsz=640,
        conf=0.25,
        verbose=False
    )
    logger.info("모델 추론 완료!")
    return results[0]


# HTTP 응답 직렬화
def output_fn(prediction, accept):
    """
    모델의 추론 결과를 클라이언트가 이해할 수 있는 HTTP 응답 형태로 변환합니다.
    """
    logger.info(f"응답을 생성합니다 (Accept: {accept})...")
    if accept == "application/json":
        preds = []
        if prediction.boxes is not None:
            # boxes.data는 [x1, y1, x2, y2, conf, cls_id] 텐서를 포함합니다.
            # .cpu().numpy().tolist()를 사용하여 Python 리스트로 변환합니다.
            detections = prediction.boxes.data.cpu().numpy().tolist()
            
            # ✅ [핵심 수정]
            # Streamlit 앱(app.py)은 [[x1, y1, x2, y2, conf, cls_id], ...] 형태의
            # 평탄한 리스트를 기대하므로, 그 형식에 맞게 응답을 재구성합니다.
            for det in detections:
                preds.append([
                    float(det[0]),  # x1
                    float(det[1]),  # y1
                    float(det[2]),  # x2
                    float(det[3]),  # y2
                    float(det[4]),  # confidence
                    int(det[5])     # class_id
                ])

        # 최종 결과를 Streamlit 앱이 기대하는 JSON 구조로 직렬화하여 반환합니다.
        return json.dumps({"detections": preds}), accept

    else:
        raise ValueError(f"지원하지 않는 Accept 타입입니다: {accept}")
