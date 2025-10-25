# 🦺 Safety Helmet Detection

건설 현장에서 **안전모 미착용 사고 예방**을 목표로, 영상 속 인물의 **헬멧 착용 여부를 자동 감지**하는 Object Detection 프로젝트입니다. Roboflow Public Dataset을 기반으로 **Faster R-CNN**을 전이학습(fine-tuning)하여 YouTube 영상에서 실시간 감지 테스트를 수행했습니다.

> **기간:** 2025.10.23 ~ 2025.10.24  
> **발표 자료:** [object detect - 헬멧착용여부](https://github.com/pythoncide/safety-helmet-detection/blob/main/object%20detect%20-%20%ED%97%AC%EB%A9%A7%EC%B0%A9%EC%9A%A9%EC%97%AC%EB%B6%80.pdf)

---

## 🚨 1. 프로젝트 개요

산업 현장에서는 여전히 **안전모 미착용 사고가 다수 발생**하고 있으며, 특히 추락·낙하 사고는 사망 사고의 상당 비중을 차지합니다. 기존 CCTV 모니터링은 **사람이 직접 확인해야 하는 한계**가 있어, 이를 자동화할 수 있는 **AI 기반 영상 감지 기술의 수요가 증가**하고 있습니다.

**본 프로젝트의 목표**  
- 영상에서 **헬멧 착용 여부를 자동 감지하는 모델 구현**  
- 추후 **실시간 경고 시스템과 연동 가능한 형태로 확장 가능성 제시**

---

## 📌 2. 데이터 및 모델 구성

### 📌 Dataset
- **Roboflow Public Dataset (COCO 형식)**
- 라벨: `with_helmet`, `without_helmet`
- 다양한 작업 환경(거리, 조명, 배경)을 포함하여 일반화 성능 확보
- Label Format: COCO JSON (Bounding Box 기반)

### 📌 Model
- **Faster R-CNN (ResNet50-FPN)**
- 작은 객체(helmet) 검출에 유리한 FPN 구조
- 2-stage detector → 비교적 높은 검출 정확도
- COCO Pretrained Weight 기반 전이학습 수행

---

## ⚙️ 3. 학습 및 실험 환경

| 항목 | 내용 |
|---|---|
| Framework | PyTorch |
| Training Image Size | 640 × 640 |
| Optimizer | Adam |
| Dataset Split | train / val / test |
| Test Video | YouTube 안전 UCC 영상 |

Pruning 20~50% 실험 진행 → **속도 향상** / **Precision 소폭 감소**

---

## 📊 4. 결과 (Result)

- 테스트 영상에서 안정적으로 **헬멧 착용 여부 검출 성공**
- 실시간 감지 환경에서도 적용 가능 확인

| 단계 | Pruning 비율 | 모델 크기(MB) | Precision | Recall |
|---|---|---|---|---|
| Original | 0% | 158MB | 0.876 | 0.952 |
| Iter#1 | 20% | 135MB | 0.900 | 0.943 |
| Iter#2 | 30% | 112MB | 0.923 | 0.947 |
| Iter#3 | 50% | 93MB | 0.877 | 0.930 |

---

## ⚠️ 5. 한계점

| 한계 | 원인 |
|---|---|
| 얼굴이 가려진 상태의 착용 여부 오인식 | 데이터셋 부재 |
| 카메라 상단(Top View) 각도에서 인식률 저하 | 시야 차이 + 미학습 샘플 |
| 작은 객체 검출 시 Confidence 흔들림 | Faster R-CNN 특성 |

---

## 🚀 6. 향후 개선 방향

- Top-view / Mask 착용 작업자 이미지 추가 후 재학습
- YOLO 계열 모델로 속도 비교 실험 (실시간성 강화)
- **실시간 경고 시스템 + 출입 통제 시스템 연동**

---

