# 닥터 스팟 (Dr. Spot) 🔬👨‍⚕️
AI 기반 피부 병변 탐지 및 분류 웹 서비스  
YOLO 탐지 + EfficientNet 분류 + LLM 챗봇을 포함한 피부 병변 분석 웹 서비스  
**배포 스택:** Amazon ECR · ECS Fargate · S3 · CloudFront · Lambda(Bedrock) · SageMaker
---
## 📌 프로젝트 개요
닥터 스팟은 사용자가 업로드한 피부 이미지를 AI 모델이 분석하여 **병변 위치 탐지**와 **종류 분류**를 동시에 수행하는 웹 서비스다.  
YOLOv8을 기반으로 피부 병변을 탐지하고, EfficientNet B4 모델로 해당 병변을 분류한다. 또한, AWS Bedrock 기반 LLM을 연동하여 **실시간 상담 챗봇** 기능을 제공한다.  
이 시스템은 의료 접근성이 떨어지는 환경에서도 **자가 진단 보조**와 **의료 상담 전 초기 판단**에 도움을 줄 수 있다.
---
## 🚀 주요 기능
- **YOLOv8 기반 병변 탐지**: 업로드된 이미지에서 병변 위치를 자동 탐지  
- **EfficientNet B4 기반 분류**: 탐지된 병변을 6개 질환 클래스로 분류  
- **LLM 기반 챗봇 상담**: AWS Bedrock + Lambda + Knowledge Base를 통한 맞춤형 상담  
- **웹 서비스 제공**: Streamlit UI를 통해 사용자가 쉽게 접근 가능  
- **전처리 강화**: CLAHE, 감마 보정, Bilateral Filter 등으로 이미지 품질 개선  
---
## 💻 데모 / 시연
- 웹사이트: (https://d1lqrqkz5ky8at.cloudfront.net)  
- 시연 영상: *(추후 추가 예정)*  
---
## 📂 프로젝트 구조
```
├── FINAL_CODE/
├── preprocessing/
│ ├── brightness.ipynb
│ ├── data balance – leakplt.ipynb
│ └── data imbalance– dataleak.ipynb
├── sagemaker/
│ ├── model_endpoint
│ ├── model_run
│ ├── model_train
│ ├── yolo_endpoint
│ ├── yolo_inference
│ └── yolo_train
├── streamlit deploy/
│ ├── app.py
│ ├── Dockerfile
│ ├── new-task-def.json
│ ├── requirements.txt
│ └── task-definition.json
└── README.md
```

## ⚙️ 환경 변수 설정
`.env` 또는 배포 Task 정의에 따라 환경 변수를 설정해야 한다:
- `ENDPOINT_NAME` : SageMaker 분류 모델 엔드포인트 이름  
- `YOLO_ENDPOINT` : YOLO 탐지 엔드포인트 이름  
- `AWS_REGION` : AWS 리전 (예: ap-northeast-2)  
- `LAMBDA_NAME` : Bedrock Knowledge Base 호출용 Lambda 이름  
---
## 📊 실험 결과 요약
- **모델 성능 (EfficientNet B4 기준)**  
  - Accuracy: 0.86  
  - AUC: 0.97  
- **전처리 효과**  
  - CLAHE + 감마 보정 → 명암 대비 향상  
  - Bilateral Filter → 피부 질감 보존하며 노이즈 제거  
---
## 🩺 활용 시나리오
- **자가 피부 진단 보조**: 병원 방문 전 초기 확인  
- **의료 취약 지역 보조 도구**: 의료 접근성 부족 환경에서 1차 필터링  
- **의료 상담 지원**: 전문의 진료 전 참고용 정보 제공  
- **교육/연구 목적**: 의료 인공지능 학습 및 실험 자료 활용  
---
## 📜 라이선스
All rights reserved © 2025 [Your Name or Team Name].  
이 저장소의 소스코드와 자료는 무단 복제, 수정, 배포, 사용을 금지합니다.
