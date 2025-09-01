# 1. 베이스 이미지 선택 (경량화된 Python 3.10)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 라이브러리 설치 (소스코드 복사 전에 실행하여 Docker 캐시 효율 극대화)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. 프로젝트 폴더의 모든 파일을 컨테이너의 작업 디렉토리로 복사
COPY . .

# 5. Streamlit이 사용하는 포트(8501)를 외부에 노출
EXPOSE 8501

# 6. 컨테이너가 시작될 때 실행할 명령어
# --server.address=0.0.0.0 옵션은 컨테이너 외부에서 접근 가능하도록 합니다.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
