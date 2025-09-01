# app.py
# 닥터 스팟(Dr. Spot) — AI 피부과 전문의 (Streamlit)
# ----------------------------------------------------
# 구성:
# - 분석 파이프라인: YOLO(탐지) → SageMaker(분류)
# - 챗봇: LAMBDA_NAME(필수) → Mock 폴백 (Bedrock 미사용)
# - DISEASE_INFO: 영문 키 유지, 화면 표시는 한글 매핑

import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import boto3
import json
import numpy as np
import io
import torch
from datetime import datetime
from botocore.config import Config
import pandas as pd
import time, os, base64
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ===============================
# 기본 설정
# ===============================
st.set_page_config(
    page_title="닥터 스팟 | AI 피부과",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 전역 CSS (탭 가로 스크롤 & 드래그 지원용 JS 포함) ---
st.markdown(
    """
    <style>
      /* ===== 기본(데스크탑/태블릿) ===== */
      .stTabs { margin-top: 12px; }
      .stTabs [role="tablist"] {
        flex-wrap: nowrap;
        gap: 12px;
      }
      .stTabs [role="tab"] {
        padding: 10px 18px;
        font-size: 1rem;
        line-height: 1.2;
      }

      /* 상단 안전영역 패딩은 모바일에서만 적용 */
      @media (max-width: 900px){
        html, body, [data-testid="stAppViewContainer"] > .main {
          padding-top: calc(env(safe-area-inset-top, 0px) + 12px) !important;
        }
        /* ✅ 모바일: 탭을 한 줄 유지 + 가로 스크롤(스와이프/드래그) 허용 */
        .stTabs [role="tablist"],
        .stTabs [data-baseweb="tab-list"]{
          flex-wrap: nowrap !important;
          justify-content: flex-start !important;
          gap: 10px !important;
          overflow-x: auto !important;
          overflow-y: hidden !important;
          -webkit-overflow-scrolling: touch;
          white-space: nowrap !important;
          padding: 6px 8px !important;
          scroll-behavior: smooth;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar,
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{ display:none; }
        .stTabs [role="tab"],
        .stTabs [data-baseweb="tab"]{
          flex: 0 0 auto !important;      /* 줄바꿈 방지 */
        }
      }

      /* 헤더/사이드바 보정(필요 시) */
      header[data-testid="stHeader"]{
        position: sticky; top: 0; z-index: 1000; background: transparent;
      }
      section[data-testid="stSidebar"] { top: calc(env(safe-area-inset-top, 0px) + 8px); }
    </style>

    <script>
    // ===== 탭바 드래그 스크롤 기능 =====
    (function(){
      function makeDragScroll(el){
        if (!el || el.dataset.dragScrollReady === "1") return;
        el.dataset.dragScrollReady = "1";
        let isDown = false, startX = 0, startLeft = 0;

        el.addEventListener("pointerdown", (e)=>{
          isDown = true;
          startX = e.clientX;
          startLeft = el.scrollLeft;
          el.setPointerCapture(e.pointerId);
        });
        el.addEventListener("pointermove", (e)=>{
          if(!isDown) return;
          const dx = e.clientX - startX;
          el.scrollLeft = startLeft - dx;
        });
        const up = ()=>{ isDown = false; };
        el.addEventListener("pointerup", up);
        el.addEventListener("pointercancel", up);

        // 세로 휠을 가로 스크롤로 전환(모바일/트랙패드 친화)
        el.addEventListener("wheel", (e)=>{
          if(Math.abs(e.deltaY) > Math.abs(e.deltaX)){
            el.scrollLeft += e.deltaY;
          }
        }, {passive:true});
      }

      function enhanceAll(){
        const sels = ['.stTabs [role="tablist"]', '.stTabs [data-baseweb="tab-list"]'];
        document.querySelectorAll(sels.join(",")).forEach(makeDragScroll);
      }

      const mo = new MutationObserver(enhanceAll);
      mo.observe(document.documentElement, {childList:true, subtree:true});
      enhanceAll();
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# (선택) 시각적 여백 약간 추가
st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

# 환경 변수 설정
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")       # 분류 엔드포인트 이름
YOLO_ENDPOINT = os.environ.get("YOLO_ENDPOINT")       # YOLO 탐지 엔드포인트 이름
REGION_NAME = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
LAMBDA_NAME = os.environ.get("LAMBDA_NAME")           # 챗봇(지식베이스) Lambda 이름

# 필수 환경 변수 확인 (배포 시 활성화 권장)
_missing = [k for k, v in {
    "ENDPOINT_NAME": ENDPOINT_NAME,
    "YOLO_ENDPOINT": YOLO_ENDPOINT,
    "AWS_REGION": REGION_NAME,
    "LAMBDA_NAME": LAMBDA_NAME,
}.items() if not v]
# if _missing:
#     st.error(f"필수 환경 변수 누락: {', '.join(_missing)} — 배포 환경 변수에 값을 설정해주세요.")
#     st.stop()

# YOLO I/O 모드
YOLO_REQ_MODE = "binary"   # "binary" or "json"
YOLO_RES_MODE = "json"     # YOLO 응답은 JSON 가정: {"detections": [[x1,y1,x2,y2, conf, cls], ...]}

# 클래스(영문 키는 서버 표준과 매칭)
CLASS_NAMES = ['1. Enfeksiyonel', '2. Ekzama', '3. Akne', '4. Pigment', '5. Benign', '6. Malign']
CLASS_NAMES_KR = {
    '1. Enfeksiyonel': '감염성 질환 | 세균・바이러스・진균 감염 (예: 농가진, 봉와직염, 대상포진, 무좀)',
    '2. Ekzama': '습진 | 아토피・접촉피부염・지루피부염',
    '3. Akne': '여드름 | 모낭염',
    '4. Pigment': '색소 질환 | 멜라스마・주근깨・잡티・염증후 색소침착',
    '5. Benign': '양성 종양| 양성 모반・피지샘 증식・지루각화증・혈관종',
    '6. Malign': '악성 종양 | 흑색종・기저세포암(BCC)・편평세포암(SCC)'
}

# Pillow 호환
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS

# ===============================
# 질환 정보(교육·참고용) 데이터 — 영문 키 + 영문 필드명
# ===============================
DISEASE_INFO = {
    "1. Enfeksiyonel": {
        "alias": ["세균/바이러스/진균 감염 (예: 농가진, 봉와직염, 대상포진, 무좀)"],
        "summary": "감염성 병변 가능성. 통증·고름·딱지·급속한 홍반/부종이 단서가 될 수 있습니다.",
        "symptoms": [
            "따갑거나 욱신거리는 통증, 열감, 붓기",
            "노란 딱지/고름, 물집(대상포진 가능)",
            "병변이 빠르게 퍼짐, 림프절 압통"
        ],
        "selfcare": [
            "상처 부위를 깨끗하고 건조하게 유지",
            "비누 거품을 충분히 내어 부드럽게 세정, 강한 문지름 금지",
            "개인 수건/면도기 등 공유 금지"
        ],
        "avoid": ["스스로 바늘/칼로 짜기", "광범위한 스테로이드 연고 남용(세균/진균 악화 가능)"],
        "redflags": [
            "39℃ 이상 발열/오한/전신 권태",
            "홍반이 빠르게 확장, 줄무늬 모양(림프관염 의심)",
            "면역저하 상태(당뇨, 항암 등) + 상기 증상"
        ],
        "prevention": [
            "손 위생, 상처 소독 및 건조 유지",
            "공용 운동시설/수영장 후 발가락 사이 충분한 건조",
            "대상포진 의심 시 수포를 가리고 접촉 주의"
        ],
        "differentials": ["알레르기성 접촉피부염", "습진성 병변", "자기손상성 병변"]
    },
    "2. Ekzama": {
        "alias": ["아토피/접촉피부염/지루피부염 등"],
        "summary": "가려움+건조+염증이 핵심. 피부 장벽 회복과 트리거 회피가 중요합니다.",
        "symptoms": [
            "건조·비늘·균열, 심한 가려움",
            "긁은 자국과 진물, 야간 악화",
            "특정 금속/화장품/세정제/직물 접촉 후 악화"
        ],
        "selfcare": [
            "하루 2~3회 보습(목욕 후 3분 이내 도포)",
            "미지근한 샤워, 순한 비누/클렌저 사용",
            "의복은 면/부드러운 소재, 땀 난 후 즉시 샤워"
        ],
        "avoid": ["향료/알코올 강한 제품, 거친 스크럽", "과도한 뜨거운 물 샤워"],
        "redflags": ["피부가 심하게 붓고 뜨거움+고름/딱지(이차 감염 의심)", "눈꺼풀·입 주변 심한 부종"],
        "prevention": ["개인 트리거 리스트화 후 회피", "겨울철 가습, 여름철 땀 관리"],
        "differentials": ["건선", "균감염(무좀)", "두드러기"]
    },
    "3. Akne": {
        "alias": ["여드름/모낭염 포함"],
        "summary": "면포(블랙·화이트헤드) + 염증성 구진·농포. 자극↓, 꾸준한 스킨케어가 핵심.",
        "symptoms": ["T존 피지 증가, 면포/구진/농포", "월경 전후 악화, 마스크/헬멧 부위 악화"],
        "selfcare": [
            "약산성 클렌저 하루 2회, 과잉 세안 금지",
            "벤조일퍼옥사이드(BPO)·아다팔렌(일반의약품) 점진 도입",
            "논코메도제닉 보습제 사용"
        ],
        "avoid": ["손/도구로 짜기(흉터·염증 확산 위험)", "두꺼운 오일리 화장, 얼굴 문지르기"],
        "redflags": ["결절·낭종(깊고 아픈 종창)/흉터 진행", "갑작스런 전신성 폭발적 악화"],
        "prevention": ["헬멧/마스크 접촉부 청결, 운동 후 즉시 세안", "유분 많은 헤어/스킨제품 지양"],
        "differentials": ["주사(rosacea)", "스테로이드 여드름", "여드름 모낭염"]
    },
    "4. Pigment": {
        "alias": ["멜라스마/주근깨/잡티/염증후 색소침착"],
        "summary": "자외선이 최대 악화 요인. 차단+완만한 미백 관리가 기본.",
        "symptoms": ["얼굴의 대칭성 갈색 반점(멜라스마)", "염증 후 남은 갈색/회색 얼룩(PIH)"],
        "selfcare": ["아침 SPF50+ 자차, 2~3시간마다 덧바름", "아젤라산·나이아신아마이드·비타민C 서서히 도입"],
        "avoid": ["과도한 각질제거/필링 남용", "10~16시 자외선 무방비 노출"],
        "redflags": ["점/반점의 빠른 크기 증가·비대칭·경계 불규칙·출혈/가피(악성 의심)"],
        "prevention": ["모자/선글라스/그늘 활용", "열 노출(사우나/고열기기) 줄이기"],
        "differentials": ["오타모반", "상피내암/흑색종 초기", "지루각화증"]
    },
    "5. Benign": {
        "alias": ["양성 모반/피지샘 증식/지루각화증/혈관종 등"],
        "summary": "대개 무해. 다만 ‘변화’가 생기면 검진 필요.",
        "symptoms": ["오래된 점/사마귀 모양, 색 균일", "표면 거침 또는 밀크커피색 반점"],
        "selfcare": ["자외선 차단(변화 방지)", "옷/면도에 걸리면 마찰 최소화"],
        "avoid": ["자가 제거 시도(화상·감염 위험)"],
        "redflags": ["크기·색·모양·두께의 ‘최근 변화’", "출혈·궤양·딱지 반복"],
        "prevention": ["정기적 셀프 스킨체크(사진 기록), 고위험군은 정기 검진"],
        "differentials": ["흑색종", "기저세포암", "과각화 병변"]
    },
    "6. Malign": {
        "alias": ["악성 종양: 흑색종, 기저세포암(BCC), 편평세포암(SCC) 등"],
        "summary": "의심 시 지체 금물. ABCDE 규칙·비정형 변화·출혈/궤양이면 즉시 진료.",
        "symptoms": [
            "A 비대칭 · B 경계 불규칙 · C 색 다양 · D 지름>6mm · E 변화",
            "상처처럼 아물지 않는 궤양/출혈, 진주광택 결절(BCC)"
        ],
        "selfcare": ["임시 보호(거즈/밴드), 추가 조작 금지", "즉시 전문의 방문 일정 잡기"],
        "avoid": ["지연·방치, 강한 일광 노출"],
        "redflags": ["빠른 크기 증가, 다색/검은·회색 혼합, 융기+출혈", "신경병증 통증/무감각 동반"],
        "prevention": ["SPF50+ 자차, 긴 옷, 그늘, 인공탠 금지", "과거 병력·가족력 있으면 정기 검진"],
        "differentials": ["양성 모반/지루각화증(겉보기 유사)", "기타 색소병변"]
    },
}

# ===============================
# 세션 상태 초기화
# ===============================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None   # 예: "3. Akne"
if "prediction_confidence" not in st.session_state:
    st.session_state.prediction_confidence = None
if "analyzed_image" not in st.session_state:
    st.session_state.analyzed_image = None
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "안녕하세요! 닥터 스팟입니다. 사진 분석 후 ‘상담’ 탭에서 질문해 보세요."}
    ]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "timelog" not in st.session_state:
    st.session_state.timelog = None
if "topk_idx" not in st.session_state:
    st.session_state.topk_idx = None
if "topk_conf" not in st.session_state:
    st.session_state.topk_conf = None

# ===============================
# CSS 스타일링(브랜드 스타일)
# ===============================
def inject_custom_css():
    st.markdown("""
    <style>
    :root {
        --primary-blue: #2E5BBA;
        --secondary-blue: #4A90E2;
        --accent-teal: #40E0D0;
        --success-green: #10B981;
        --warning-red: #EF4444;
        --text-dark: #FFFFFF;
        --text-light: #E5E7EB;
        --bg-light: #1E293B;
        --bg-card: #334155;
        --border-light: #475569;
    }
    .main .block-container {
        padding-top: 1rem; padding-bottom: 2rem;
        max-width: 1200px; background-color: #0F172A; color: white;
    }
    .stApp { background-color: #0F172A; color: white; }
    .stMarkdown, .stText, .stHeader, .stSubheader, .stTitle, 
    .stCaption, p, span, div, h1, h2, h3, h4, h5, h6 { color: white !important; }

    .hero-section {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white; padding: 3rem 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;
    }
    .hero-title { font-size: 3rem; font-weight: 800; margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .hero-subtitle { font-size: 1.25rem; opacity: 0.9; margin-bottom: 2rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 1rem; justify-content: center; background-color: #0F172A; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; padding: 0 2rem; border-radius: 25px; border: 2px solid var(--border-light);
        background: var(--bg-card); color: white !important; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: var(--primary-blue); color: white !important; border-color: var(--primary-blue); }

    .feature-card {
        background: var(--bg-card); border-radius: 16px; padding: 2rem; border: 1px solid var(--border-light);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: all 0.3s ease; height: 100%; text-align: center; color: white !important;
    }
    .feature-card:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0,0,0,0.4); }
    .feature-card h4, .feature-card p, .feature-card div { color: white !important; }

    .card-icon { width: 60px; height: 60px; border-radius: 16px;
        background: linear-gradient(135deg, var(--accent-teal), var(--secondary-blue));
        display: flex; align-items: center; justify-content: center; font-size: 1.8rem; margin: 0 auto 1rem auto; color: white; }

    .process-step { text-align: center; padding: 1.5rem; margin: 1rem 0; }
    .step-number { width: 50px; height: 50px; border-radius: 50%; background: var(--accent-teal); color: white;
        display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; margin: 0 auto 1rem auto; }
    .step-title { font-weight: 600; color: white !important; margin-bottom: 0.5rem; }
    .step-description { color: var(--text-light) !important; font-size: 0.9rem; }

    .result-card { background: linear-gradient(135deg, var(--success-green), #22D3EE);
        color: white; padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0; }
    .result-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; }
    .confidence-score { font-size: 3rem; font-weight: 800; margin: 1rem 0; }

    .stat-card { background: var(--bg-card); border-radius: 12px; padding: 1.5rem; text-align: center;
        border-left: 4px solid var(--accent-teal); margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.3); color: white !important; }
    .stat-number { font-size: 2.5rem; font-weight: 800; color: var(--accent-teal) !important; margin-bottom: 0.5rem; }
    .stat-label { color: var(--text-light) !important; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px; }

    .disease-card { background: var(--bg-card); border-radius: 16px; padding: 2rem; margin: 1rem 0;
        border-left: 4px solid var(--secondary-blue); box-shadow: 0 2px 8px rgba(0,0,0,0.3); color: white !important; }
    .disease-title { color: var(--accent-teal) !important; font-weight: 700; font-size: 1.3rem; margin-bottom: 1rem; }
    .disease-card p, .disease-card div { color: white !important; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stButton > button { border-radius: 25px; border: none; padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .main .block-container { padding-left: 1rem; padding-right: 1rem; }
        .feature-card { padding: 1.5rem; }
    }

    /* --- Brand Topbar --- */
    .topbar {
      position: sticky; top: 0; z-index: 1000;
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 18px; margin-bottom: 10px; border-bottom: 1px solid var(--border-light);
      background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%); color: white; border-radius: 12px;
    }
    .topbar .brand { display:flex; align-items:center; gap:12px; font-weight:900; letter-spacing:.2px; }
    .topbar .logo-badge { width: 40px; height: 40px; border-radius: 10px; display:flex; align-items:center; justify-content:center;
      background: rgba(255,255,255,.16); font-size: 22px; }
    .topbar .brand-name { font-size: 22px; }
    .topbar .sub { opacity:.85; font-size: 12px; font-weight:700; }
    .tab-spacer { height: 6px; }
    .stTabs { margin-top: 10px; }
    .stTabs [data-baseweb="tab-list"] { z-index: 5; }
    .stTabs [data-baseweb="tab"] { box-shadow: 0 2px 0 rgba(0,0,0,.25) inset; }

    .hero-banner {
      background: linear-gradient(135deg, #2E5BBA, #4A90E2);
      text-align: center; color: white; padding: 28px 16px; border-radius: 14px; margin: 8px 0 12px;
    }
    .hero-banner h1 { font-size: 2.8rem; font-weight: 900; margin: 0 0 6px; }
    .hero-banner p { font-size: 1.05rem; opacity: .92; margin:0; }
    @media (max-width: 768px){
      .topbar { padding: 10px 12px; border-radius: 10px; }
      .topbar .brand-name { font-size: 18px; }
      .hero-banner h1 { font-size: 2.1rem; }
    }

    /* ✅ 분석 탭 하단 안내 배너 */
    .next-step-banner{
      margin-top: 18px;
      background: linear-gradient(90deg, rgba(59,130,246,.18), rgba(16,185,129,.18));
      border: 1px dashed rgba(148,163,184,.45);
      color: #E8F2FF; padding: 14px 16px; border-radius: 14px;
      display:flex; align-items:center; gap:12px;
    }
    .next-step-banner .icon{
      width: 36px; height: 36px; border-radius: 10px;
      display:flex; align-items:center; justify-content:center;
      background: rgba(255,255,255,.12); font-size: 18px;
    }
    .next-step-banner b{ color: #A7F3D0; }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# 이미지 전처리 함수들
# ===============================
def calculate_brightness(image: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    grayscale = image.convert('L')
    return float(np.mean(np.array(grayscale)))

def calculate_sharpness(image: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    grayscale = image.convert('L')
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    return float(np.var(np.array(edges)))

def apply_gamma_correction(image: Image.Image, gamma_value: float):
    if gamma_value == 1.0:
        return image
    inv_gamma = 1.0 / gamma_value
    table = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
    if image.mode == 'RGB':
        r, g, b = image.split()
        r = r.point(table); g = g.point(table); b = b.point(table)
        return Image.merge('RGB', (r, g, b))
    return image.point(table)

def enhance_contrast_adaptive(image: Image.Image, clip_limit: float):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    contrast_factor = 1.0 + (clip_limit - 1.0) * 0.3
    return ImageEnhance.Contrast(image).enhance(contrast_factor)

def apply_bilateral_filter_approximation(image: Image.Image, d=9, sigma=30):
    blur_radius = max(0.5, int(sigma / 50))
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def adaptive_preprocess_image(image: Image.Image, preserve_size=True):
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    brightness = calculate_brightness(image)
    sharpness = calculate_sharpness(image)

    if brightness < 80:
        gamma_value = 0.7; brightness_status = "어두움 → 밝게 조정"
    elif brightness > 180:
        gamma_value = 1.3; brightness_status = "밝음 → 어둡게 조정"
    else:
        gamma_value = 1.0; brightness_status = "적정 밝기"

    if sharpness < 100:
        clip_limit = 3.0; sharpness_status = "흐림 → 선명도 증가"
    else:
        clip_limit = 1.5; sharpness_status = "적정 선명도"

    processed = image.copy()
    processed = enhance_contrast_adaptive(processed, clip_limit)
    processed = apply_gamma_correction(processed, gamma_value)
    if sharpness < 50:
        processed = apply_bilateral_filter_approximation(processed, d=9, sigma=30)
    if not preserve_size:
        processed = processed.resize((224, 224), RESAMPLE_LANCZOS)
    return processed, brightness, sharpness, brightness_status, sharpness_status

# ===============================
# AWS 연결/추론 유틸 (공통)
# ===============================
def to_jpeg_bytes(img: Image.Image, max_side=512, quality=90):
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), RESAMPLE_LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

@st.cache_resource(show_spinner=False)
def get_sagemaker_client(region: str):
    try:
        return boto3.client(
            'sagemaker-runtime',
            region_name=region,
            config=Config(
                retries={"max_attempts": 3, "mode": "standard"},
                read_timeout=90,
                connect_timeout=5,
                tcp_keepalive=True,
            ),
        )
    except Exception as e:
        st.error(f"AWS Boto3 클라이언트 생성 오류: {e}")
        return None

# ===============================
# YOLO 탐지
# ===============================
@dataclass
class Detection:
    box: Tuple[float, float, float, float]  # x1,y1,x2,y2
    conf: float
    cls_id: int

def clip_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

def invoke_yolo(image_bytes: bytes) -> List[Detection]:
    smrt = get_sagemaker_client(REGION_NAME)
    if not smrt:
        st.error("YOLO 호출 실패: SageMaker 클라이언트를 만들 수 없습니다.")
        return []

    if YOLO_REQ_MODE == "binary":
        body = image_bytes
        content_type = "application/x-image"
    else:
        body = json.dumps({"image": base64.b64encode(image_bytes).decode()})
        content_type = "application/json"

    res = smrt.invoke_endpoint(
        EndpointName=YOLO_ENDPOINT,
        ContentType=content_type,
        Accept="application/json",
        Body=body
    )
    payload = res["Body"].read().decode("utf-8")

    if YOLO_RES_MODE != "json":
        st.error("YOLO_RES_MODE는 json만 지원합니다.")
        return []

    data = json.loads(payload)
    det_list = data.get("detections", data)
    out: List[Detection] = []
    for det in det_list:
        x1, y1, x2, y2, conf, cls_id = det
        out.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(conf), int(cls_id)))
    return out

# === YOLO 입력 준비: 바이트 + (모델입력크기, 원본크기) 반환 ===
def prepare_yolo_image(image: Image.Image, max_side=1024, quality=90):
    img = ImageOps.exif_transpose(image).convert("RGB")
    W0, H0 = img.size
    scale = max_side / max(W0, H0) if max(W0, H0) > max_side else 1.0
    if scale < 1.0:
        W1, H1 = int(W0*scale), int(H0*scale)
        img_resized = img.resize((W1, H1), RESAMPLE_LANCZOS)
    else:
        W1, H1 = W0, H0
        img_resized = img
    buf = io.BytesIO()
    img_resized.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), (W1, H1), (W0, H0)

# === 박스를 xyxy로 정규화하고 (모델입력크기)->(원본크기)로 매핑 ===
def map_box_to_original(det_box, model_size, orig_size):
    x1, y1, x2, y2 = [float(v) for v in det_box]

    # 1) xywh 케이스 자동 판별: x2<=x1 또는 y2<=y1 면 (cx,cy,w,h)로 간주
    if x2 <= x1 or y2 <= y1:
        cx, cy, w, h = x1, y1, x2, y2
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2

    Wm, Hm = model_size
    Wo, Ho = orig_size

    # 2) 정규화(0~1.x)면 모델입력 해상도로 스케일
    if 0 <= max(x2, y2) <= 1.5:
        x1, x2 = x1 * Wm, x2 * Wm
        y1, y2 = y1 * Hm, y2 * Hm

    # 3) (모델입력)->(원본) 매핑 (우리가 YOLO에 넣은 리사이즈 비율 반영)
    sx, sy = Wo / Wm, Ho / Hm
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy

    # 4) 정렬 + 정수화
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
# ===============================
# 분류 (항상 SageMaker 엔드포인트로만)
# ===============================
def get_sagemaker_prediction(image):
    smrt = get_sagemaker_client(REGION_NAME)
    if not smrt:
        st.session_state["timelog"] = None
        st.error(f"예측 중 오류가 발생했습니다. 엔드포인트({ENDPOINT_NAME}, {REGION_NAME}) 설정/권한을 확인하세요.")
        return None, None
    def _save_timelog(path, prep, invoke, parse):
        st.session_state["timelog"] = {
            "path": path,
            "prep": round(prep, 3),
            "invoke": round(invoke, 3),
            "parse": round(parse, 3),
            "total": round(prep + invoke + parse, 3),
        }
    try:
        t0 = time.time()
        jpg = to_jpeg_bytes(image, max_side=512, quality=90)
        prep_sec = time.time() - t0

        t1 = time.time()
        response = smrt.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/x-image",
            Accept="application/json",
            Body=jpg,
        )
        invoke_sec = time.time() - t1

        t2 = time.time()
        obj = json.loads(response["Body"].read().decode("utf-8"))
        parse_sec = time.time() - t2
        _save_timelog("binary", prep_sec, invoke_sec, parse_sec)

        def _align_with_class_names(labels_infer, probs_infer):
            if labels_infer and set(labels_infer) == set(CLASS_NAMES):
                name2p = {str(lbl): float(p) for lbl, p in zip(labels_infer, probs_infer)}
                return [name2p.get(name, 0.0) for name in CLASS_NAMES]
            if labels_infer and all(str(x).startswith("class_") for x in labels_infer):
                def idx_of(l):
                    try:
                        return int(str(l).split("_")[-1])
                    except Exception:
                        return 10**9
                order = np.argsort([idx_of(l) for l in labels_infer]).tolist()
                return [float(probs_infer[i]) for i in order]
            return [float(x) for x in probs_infer]

        prob_list = None
        if isinstance(obj, dict):
            if "prob_list" in obj:
                labels_resp = obj.get("labels")
                prob_list = _align_with_class_names(labels_resp, obj["prob_list"])
            elif isinstance(obj.get("probs"), dict):
                probs_by_name = obj["probs"]
                labels_resp = obj.get("labels")
                if labels_resp:
                    ordered = [float(probs_by_name.get(l, 0.0)) for l in labels_resp]
                    prob_list = _align_with_class_names(labels_resp, ordered)
                elif set(probs_by_name.keys()) >= set(CLASS_NAMES):
                    prob_list = [float(probs_by_name.get(name, 0.0)) for name in CLASS_NAMES]
                else:
                    keys = list(probs_by_name.keys())
                    prob_list = [float(probs_by_name[k]) for k in keys]
            elif "logits" in obj:
                t = torch.tensor(obj["logits"], dtype=torch.float32)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                prob_list = torch.softmax(t, dim=1)[0].tolist()
            elif any(k in obj for k in ("probabilities", "output", "predictions")):
                arr = obj.get("probabilities") or obj.get("output") or obj.get("predictions")
                t = torch.tensor(arr, dtype=torch.float32)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                prob_list = torch.softmax(t, dim=1)[0].tolist()
            elif "label" in obj and "conf" in obj:
                lbl = str(obj["label"])
                idx = None
                if lbl.startswith("class_"):
                    try:
                        idx = int(lbl.split("_")[-1])
                    except Exception:
                        idx = None
                if idx is None:
                    try:
                        idx = CLASS_NAMES.index(lbl)
                    except ValueError:
                        idx = 0
                idx = max(0, min(len(CLASS_NAMES) - 1, idx))
                prob_list = [0.0] * len(CLASS_NAMES)
                prob_list[idx] = float(obj.get("conf", 1.0))
        else:
            t = torch.tensor(obj, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            prob_list = torch.softmax(t, dim=1)[0].tolist()

        if prob_list is None:
            st.error(f"모델 응답 파싱 실패: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
            return None, None

        n = len(CLASS_NAMES)
        if len(prob_list) != n:
            prob_list = (prob_list + [0.0] * n)[:n]
        probs = torch.tensor([prob_list], dtype=torch.float32)
        top_k = min(st.session_state.top_k, n)
        top_conf, top_idx = torch.topk(probs, k=top_k, dim=1)
        return top_idx[0].tolist(), top_conf[0].tolist()
    except Exception as e_json:
        st.session_state["timelog"] = None
        st.error("예측 중 오류가 발생했습니다. 엔드포인트 스키마와 권한, CloudWatch 로그를 확인하세요.")
        st.exception(e_json)
        return None, None

def add_history(image: Image.Image, top_idx, top_conf):
    """메모리 절약: 썸네일 바이트만 세션에 저장"""
    if top_idx is None:
        return
    top_class = CLASS_NAMES[top_idx[0]] if max(top_idx) < len(CLASS_NAMES) else f"#{top_idx[0]}"
    top_prob = float(top_conf[0]) if top_conf else None

    thumb = image.copy()
    thumb.thumbnail((256, 256))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    thumb_bytes = buf.getvalue()

    st.session_state.history.insert(0, {
        "class_name": top_class,
        "confidence": top_prob,
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "thumb_bytes": thumb_bytes,
    })

# ===============================
# 챗봇: LAMBDA_NAME만 사용 
# ===============================
@st.cache_resource(show_spinner=False)
def get_lambda_client(region: str):
    """KB Lambda 호출용 클라이언트 (최소 변경)"""
    return boto3.client(
        "lambda",
        region_name=region,
        config=Config(
            retries={"max_attempts": 2, "mode": "standard"},
            read_timeout=30,
            connect_timeout=3,
            tcp_keepalive=True,
        ),
    )

def _mock_kb_answer(prompt: str):
    p = prompt.lower()
    if "여드름" in p or "acne" in p:
        return ("여드름은 모공 막힘/피지/염증이 핵심입니다. "
                "벤조일퍼옥사이드(BPO)·아다팔렌은 밤에 얇게 바르고, 자극 시 격일로 조절하세요. "
                "자외선 차단과 논코메도제닉 보습을 병행하세요. 임신·수유 중엔 의사와 상의 필수.", [])
    if "습진" in p or "eczema" in p or "진물" in p:
        return ("급성 진물성 습진은 부드러운 세정, 젖은 드레싱, 단기 국소 스테로이드가 도움 됩니다. "
                "긁지 않기, 보습 강화, 자극 회피(향/알코올)를 함께 하세요. 38℃ 미지근한 샤워 권장.", [])
    if "반점" in p or "멜라스마" in p or "색소" in p:
        return ("색소 병변이 커지거나 모양/색이 변하면 전문 진료가 필요합니다. "
                "SPF50+ 차단, 비타민 C/알부틴 등 저자극 미백 루틴을 꾸준히, 과도한 필링은 지양.", [])
    return ("기본 수칙: 온화한 세안(하루 2회), 충분한 보습, 자외선 차단, 손으로 만지지 않기. "
            "지속·악화 시 피부과 전문의 상담 권장.", [])

def ask_kb(prompt: str, top_k: int = 5):
    """KB Lambda 호출 → 실패/미설정 시 Mock 폴백."""
    if not LAMBDA_NAME:
        return _mock_kb_answer(prompt)

    lam = get_lambda_client(REGION_NAME)
    payload = { "query": prompt, "prompt": prompt, "top_k": int(top_k) }
    try:
        resp = lam.invoke(
            FunctionName=LAMBDA_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        raw = resp["Payload"].read().decode("utf-8").strip()

        if raw.startswith("{"):
            data = json.loads(raw)
            if isinstance(data, dict) and "statusCode" in data and "body" in data:
                body = data["body"]
                if isinstance(body, str) and body.strip().startswith("{"):
                    body = json.loads(body)
                elif isinstance(body, str):
                    body = {"answer": body}
            else:
                body = data
        else:
            body = {"answer": raw}

        answer = (body.get("answer") or body.get("message") or "").strip()
        sources = body.get("sources") or body.get("references") or []
        if not answer:
            return _mock_kb_answer(prompt)
        return answer, sources
    except Exception as e:
        st.warning(f"KB Lambda 호출 실패: {e}")
        return _mock_kb_answer(prompt)

# ===============================
# 페이지 렌더링
# ===============================
SHOW_HOME_HERO = True

def render_home_page():
    if SHOW_HOME_HERO:
        st.markdown(f"""
        <div class="hero-section">
            <div class="hero-title">Dr. spot</div>
            <div class="hero-subtitle">AI가 즉시 피부 사진을 분석하고 개인 맞춤 가이드를 제공합니다.</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">1</div>
            <div class="step-title">Take a photo</div>
            <div class="step-description">of a skin problem</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">2</div>
            <div class="step-title">AI Analysis</div>
            <div class="step-description">instantly analyzes your photo</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">3</div>
            <div class="step-title">Get Result</div>
            <div class="step-description">Go to the tab and check the results</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="process-step">
            <div class="step-number">4</div>
            <div class="step-title">AI Consultant</div>
            <div class="step-description">explains your result</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("GET INSTANT RESULT", type="primary", use_container_width=True, key="main_cta"):
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="color: var(--text-dark);">Why should you use AI Dermatologist?</h2>
        <p style="color: var(--text-light); font-size: 1.1rem;">Developed with dermatologists and powered by artificial intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">🔍</div>
            <div class="card-title">Smart Detection</div>
            <div class="card-description">
                Detects 6+ skin diseases, including melanoma and skin cancer with over 87% accuracy.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">⚡</div>
            <div class="card-title">Instant Results</div>
            <div class="card-description">
                Get your result within 1 minute. 24/7 personal AI Consultant available anytime.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">📱</div>
            <div class="card-title">Accessible Anywhere</div>
            <div class="card-description">
                Available on any device. Keep your health in check at your fingertips even when you are on the go.
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_scan_page():
    st.markdown('<h1 style="text-align: center; color: var(--primary-blue); margin-bottom: 2rem;">피부 분석</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 이미지 업로드")
        source = st.radio("입력 방법을 선택하세요:", ["사진 촬영", "이미지 업로드"], horizontal=True)
        
        uploaded_file = None
        if source == "이미지 업로드":
            uploaded_file = st.file_uploader(
                "이미지 파일을 선택하세요",
                type=['png', 'jpg', 'jpeg'],
                help="분석할 피부 부위의 선명하고 잘 보이는 사진을 업로드하세요"
            )
        else:
            uploaded_file = st.camera_input("피부 부위 사진 촬영")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="업로드된 이미지", use_container_width=True)
            
            use_preprocessing = st.checkbox(
                "자동 이미지 전처리 사용", 
                help="더 나은 분석을 위해 이미지 밝기와 대비를 자동으로 최적화합니다"
            )
            
            if st.button("이미지 분석하기", type="primary", use_container_width=True):
                with st.spinner("이미지를 분석하는 중..."):
                    progress_bar = st.progress(0)
                    
                    if use_preprocessing:
                        progress_bar.progress(25)
                        st.info("이미지 전처리 중...")
                        processed_image, brightness, sharpness, brightness_status, sharpness_status = adaptive_preprocess_image(image, preserve_size=True)
                        st.success(f"전처리 완료: {brightness_status}, {sharpness_status}")
                        col_before, col_after = st.columns(2)
                        with col_before: st.image(image, caption="원본", use_container_width=True)
                        with col_after: st.image(processed_image, caption="전처리됨", use_container_width=True)
                        analysis_image = processed_image
                    else:
                        analysis_image = image
                        
                    progress_bar.progress(45)
                    st.info("병변 탐지(YOLO) 진행 중...")

                    target_for_cls = None
                    try:
                        # YOLO에 보낸 '리사이즈된 입력'의 크기를 확보해야 정확히 역매핑 가능
                        img_bytes_for_yolo, (Wm, Hm), (Wo, Ho) = prepare_yolo_image(analysis_image, max_side=1024, quality=90)
                        dets = invoke_yolo(img_bytes_for_yolo)
                    except Exception as e:
                        st.error(f"YOLO 호출 실패: {e}")
                        return

                    yolo_conf_thr = 0.25
                    dets = [d for d in dets if d.conf >= yolo_conf_thr]

                    if dets:
                        dets.sort(key=lambda d: d.conf, reverse=True)
                        best = dets[0]

                        # (모델입력좌표) -> (원본좌표)로 변환 (xywh/xyxy, 정규화 자동 대응)
                        x1, y1, x2, y2 = map_box_to_original(best.box, (Wm, Hm), (Wo, Ho))

                        # 안전 패딩 (박스가 너무 타이트하면 분류 성능 저하)
                        pad = int(0.06 * max(x2 - x1, y2 - y1))
                        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad

                        # 이미지 경계 클리핑
                        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, Wo, Ho)

                        # 유효성 체크
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            st.warning("탐지 박스가 너무 작거나 이상합니다. 전체 이미지로 분석합니다.")
                            target_for_cls = analysis_image
                        else:
                            crop_img = analysis_image.crop((x1, y1, x2, y2))
                            st.info(f"병변 탐지됨 (conf={best.conf:.2f}) — 해당 영역을 분석합니다.")
                            st.image(crop_img, caption="분석 대상(탐지된 영역)", use_container_width=True)
                            target_for_cls = crop_img
                    else:
                        st.warning("병변을 특정하지 못했습니다. 전체 이미지를 분석합니다.")
                        target_for_cls = analysis_image

                    progress_bar.progress(70)
                    st.info("AI 분류 진행 중...")
                    top_idx, top_conf = get_sagemaker_prediction(target_for_cls or analysis_image)
                    progress_bar.progress(100)
                    st.session_state.topk_idx = top_idx
                    st.session_state.topk_conf = [float(c) for c in top_conf]  # tensor 방지용 float 캐스팅
                    
                    if top_idx and top_conf:
                        if max(top_idx) >= len(CLASS_NAMES):
                            st.error(f"모델 출력({max(top_idx)+1})과 CLASS_NAMES({len(CLASS_NAMES)}) 개수가 불일치합니다.")
                            return
                        predicted_class = CLASS_NAMES[top_idx[0]]
                        st.session_state.prediction_result = predicted_class
                        st.session_state.prediction_confidence = top_conf[0]
                        st.session_state.analyzed_image = image
                        add_history(image, top_idx, top_conf)
                        st.success("분석이 완료되었습니다! 아래 안내를 확인하세요.")
                        #time.sleep(1)
                        #st.rerun()
                    else:
                        st.error("분석 결과를 받을 수 없습니다. 엔드포인트를 확인해주세요.")
    
    with col2:
        st.markdown("### 더 나은 결과를 위한 팁")
        st.markdown("""
        <div class="feature-card">
            <h4>사진 촬영 가이드</h4>
            <ul style="text-align: left; color: var(--text-light);">
                <li>충분한 조명 확보</li>
                <li>카메라를 안정적으로 고정하고 초점 맞추기</li>
                <li>피부 부위를 프레임 중앙에 배치</li>
                <li>그림자와 반사 피하기</li>
                <li>10-30cm 거리에서 촬영</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        stats = [("87%", "정확도"), ("6+", "탐지 질환 수"), ("< 1분", "분석 시간"), ("24/7", "이용 가능")]
        for stat, label in stats:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stat}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ✅ 하단 안내 배너 (항상 노출, 간결하고 깔끔)
    st.markdown("""
    <div class="next-step-banner">
      <div class="icon">📊</div>
      <div>
        <div><b>분석이 끝나면 ‘결과’ 탭</b>에서 상세 결과와 가이드를 확인하세요.</div>
        <div style="opacity:.85; font-size:.92rem;">필요하면 ‘상담’ 탭에서 추가 질문을 할 수 있어요.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def _bullets(title: str, items):
    if items:
        st.markdown(f"**{title}**")
        for it in items:
            st.write(f"• {it}")

def render_results_page():
    st.markdown('<h1 style="text-align: center; color: var(--primary-blue); margin-bottom: 1rem;">📊 결과</h1>', unsafe_allow_html=True)


    if not st.session_state.prediction_result:
        st.warning("분석 결과가 없습니다. 먼저 🔍 분석 탭에서 이미지를 스캔해 주세요.")
        return

    predicted_class = st.session_state.prediction_result
    predicted_kr = CLASS_NAMES_KR.get(predicted_class, predicted_class)
    confidence = float(st.session_state.prediction_confidence or 0.0)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">탐지된 질환(추정)</div>
        <div style="font-size: 2rem; font-weight: 800; margin-bottom: 0.25rem;">{predicted_kr}</div>
        <div class="confidence-score">{confidence*100:.1f}%</div>
        <div style="font-size: 0.95rem; opacity: 0.9;">신뢰도(모델 추정치)</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.analyzed_image:
        st.image(st.session_state.analyzed_image, caption="분석된 이미지", use_container_width=True)

    topk_idx = st.session_state.get("topk_idx")
    topk_conf = st.session_state.get("topk_conf")
    
    if topk_idx and topk_conf:
        st.markdown("### 상위 예측 분포")
        df = pd.DataFrame({
            "클래스": [CLASS_NAMES_KR.get(CLASS_NAMES[i], CLASS_NAMES[i]) for i in topk_idx],
            "확률(%)": [round(float(c)*100, 2) for c in topk_conf],
        })
        # 막대차트 + 표
        st.bar_chart(df.set_index("클래스"))
        with st.expander("다른 후보도 보기", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
            # 후보 텍스트 요약(Top-1 제외)
            for name, p in zip(df["클래스"][1:], df["확률(%)"][1:]):
                st.info(f"**{name}**: {p:.1f}%")
    

    st.markdown("### 결과 이해하기")
    info = DISEASE_INFO.get(predicted_class, {})
    desc = info.get("summary", "정확한 진단을 위해 의료 전문가와 상담하세요.")
    st.markdown(f"""
    <div class="disease-card">
        <div class="disease-title">{predicted_kr}</div>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    _bullets("별칭/예시", info.get("alias"))
    _bullets("주요 증상", info.get("symptoms"))
    _bullets("자가 관리", info.get("selfcare"))
    _bullets("피해야 할 것", info.get("avoid"))
    _bullets("주의 신호(🚩)", info.get("redflags"))
    _bullets("예방", info.get("prevention"))
    _bullets("감별 진단", info.get("differentials"))

    if predicted_class == "6. Malign":
        st.error("**긴급:** 암성 가능성이 의심됩니다. 즉시 의료진의 진료를 받으세요!")
    else:
        st.warning("**중요:** 본 결과는 교육/보조 목적이며, 전문 의료진의 진단을 대체할 수 없습니다.")

# ===============================
# 챗봇 페이지 — LAMBDA_NAME → Mock
# ===============================
def render_chatbot():
    st.markdown('<h1 style="text-align: center; color: var(--primary-blue); margin-bottom: 1rem;">💬 상담</h1>', unsafe_allow_html=True)

    predicted = st.session_state.get("prediction_result")
    predicted_kr = CLASS_NAMES_KR.get(predicted, predicted) if predicted else None

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "안녕하세요! 피부질환 사후관리·주의사항·약물 정보 등 무엇이든 물어보세요."}
        ]
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    with st.container(border=True):
        left, right = st.columns([3,1])
        with left:
            if predicted_kr:
                st.caption(f"최근 예측: **{predicted_kr}**")
            st.markdown("필요 시 **증상/기간/부위/복용약** 등 맥락을 함께 적어주세요.")
        with right:
            top_k = st.slider("검색 문서 수", 1, 10, min(5, st.session_state.get("top_k", 3)))

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"], avatar=("🧑‍⚕️" if m["role"]=="assistant" else "🙂")):
            st.markdown(m["content"])

    ex_cols = st.columns(3)
    examples = [
        "여드름에 BPO/아다팔렌 사용법과 주의점 알려줘",
        "진물 나는 습진, 집에서 뭘 하면 좋아요?",
        "검은 반점이 커지는 중인데 언제 병원 가야 하나요?"
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, use_container_width=True, key=f"suggest_{i}"):
            st.session_state.chat_messages.append({"role":"user","content":ex})
            st.session_state.pending_query = ex
            st.rerun()

    user_q = st.chat_input(placeholder="예: ‘여드름 악화 요인과 생활관리 팁’")
    if user_q:
        st.session_state.chat_messages.append({"role":"user","content":user_q})
        st.session_state.pending_query = user_q
        st.rerun()

    if st.session_state.pending_query:
        q = st.session_state.pending_query
        if predicted_kr and predicted_kr not in q:
            q = f"[최근 예측: {predicted_kr}] {q}"

        with st.chat_message("assistant", avatar="🧑‍⚕️"):
            with st.status("답변 생성 중...", expanded=False) as s:
                try:
                    answer, sources = ask_kb(q, top_k)
                    s.update(label="완료", state="complete")
                except Exception as e:
                    st.error(f"오류: {e}")
                    answer, sources = "", []

            if answer.strip():
                st.markdown(answer)
            else:
                st.warning("응답이 비어 있습니다. Lambda/KB 구성을 확인해주세요.")

            if sources:
                st.markdown("###### 출처")
                for i, src in enumerate(sources, 1):
                    title = src.get("title") or src.get("source") or f"문서 {i}"
                    url = src.get("url") or src.get("uri") or ""
                    with st.container(border=True):
                        st.write(f"**{i}. {title}**")
                        if isinstance(url, str) and url.startswith(("http://","https://")):
                            st.markdown(f"[열기]({url})")

        st.session_state.chat_messages.append({"role":"assistant","content":answer or "죄송해요. 다시 시도해주세요."})
        st.session_state.pending_query = None
        st.rerun()

def render_info_page():
    st.markdown('<h1 style="text-align: center; color: var(--primary-blue); margin-bottom: 1rem;">📚 피부 질환 가이드</h1>', unsafe_allow_html=True)

    options = [(k, CLASS_NAMES_KR.get(k, k)) for k in CLASS_NAMES]
    labels = [label for _, label in options]

    selected_label = st.selectbox("자세히 볼 질환을 선택하세요", labels, index=0)
    selected_key = next(k for k, l in options if l == selected_label)

    info = DISEASE_INFO.get(selected_key, {})
    st.markdown(f"""
    <div class="disease-card">
        <div class="disease-title">{selected_label}</div>
        <p><strong>요약:</strong> {info.get('summary', '—')}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        _bullets("별칭/예시", info.get("alias"))
        _bullets("주요 증상", info.get("symptoms"))
    with c2:
        _bullets("자가 관리", info.get("selfcare"))
        _bullets("피해야 할 것", info.get("avoid"))
    with c3:
        _bullets("주의 신호(🚩)", info.get("redflags"))
        _bullets("예방", info.get("prevention"))
        _bullets("감별 진단", info.get("differentials"))

    if selected_key == "6. Malign":
        st.error("**긴급:** 암성 의심 시 즉시 의료진 상담이 필요합니다. 조기 발견이 가장 중요합니다!")

    st.markdown("---")
    st.info("의료 고지: 본 정보는 교육 목적이며, 실제 진단/치료는 의료 전문가의 판단을 따르세요.")

def render_history_page():
    st.markdown('<h1 style="text-align: center; color: var(--primary-blue); margin-bottom: 1rem;">📝 기록</h1>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🗂️</div>
            <h4>분석 기록이 없습니다</h4>
            <p class="card-description">이곳에서 과거 분석 기록을 확인할 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame([
        {
            "날짜": h["when"],
            "결과": CLASS_NAMES_KR.get(h["class_name"], h["class_name"]),
            "신뢰도": f"{(h['confidence']*100):.1f}%" if h["confidence"] is not None else "N/A"
        } for h in st.session_state.history
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("CSV로 다운로드", data=csv, file_name='skin_analysis_history.csv', mime='text/csv')

    if st.checkbox("이미지 썸네일 보기"):
        cols = st.columns(3)
        for i, h in enumerate(st.session_state.history):
            with cols[i % 3]:
                if h.get("thumb_bytes"):
                    st.image(h["thumb_bytes"], caption=f"{h['when']} - {CLASS_NAMES_KR.get(h['class_name'], h['class_name'])}", use_container_width=True)

# ===============================
# 메인 앱 실행
# ===============================
def main():
    inject_custom_css()

    # 상단 브랜드 탑바(고정)
    st.markdown("""
    <div class="topbar">
      <div class="brand">
        <div class="logo-badge">🔬</div>
        <div class="brand-name">Dr. Spot</div>
      </div>
      <div class="sub">AI 피부과 어시스턴트</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tab-spacer"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 홈", "🔍 분석", "📊 결과", "💬 상담", "📚 정보", "📝 기록"
    ])

    with tab1: render_home_page()
    with tab2: render_scan_page()
    with tab3: render_results_page()
    with tab4: render_chatbot()
    with tab5: render_info_page()
    with tab6: render_history_page()

if __name__ == "__main__":
    main()