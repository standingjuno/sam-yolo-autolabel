# sam-yolo-autolabel

SAM3/SAM3.1 텍스트 프롬프트로 이미지를 자동 세그멘테이션한 뒤, Ultralytics YOLO 세그멘테이션 데이터셋(`images/`, `labels/`, `data.yaml`)으로 내보내는 프로젝트입니다.

이 문서는 아래 순서대로 따라오면 바로 실행 가능하도록 작성했습니다.

1. 세팅 방법  
2. git clone  
3. JSON 파일 사용법

---

## 1) 세팅 방법

### 1-1. 사전 준비

- OS: Windows/macOS/Linux
- Python: 3.12 권장
- GPU: NVIDIA + CUDA 사용 권장 (`--device cuda`)
- Conda 설치 권장 (가상환경 분리 목적)  

### 1-2. Conda 환경 생성

```bash
conda create -n SAM3 python=3.12 -y
conda activate SAM3
python -m pip install -U pip setuptools wheel
```

### 1-3. PyTorch(CUDA) 설치

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

설치 확인:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()`가 `True`여야 GPU 추론이 가능합니다.

### 1-4. Hugging Face 설정 (중요)

SAM3.1 체크포인트는 Hugging Face의 gated model에서 내려받습니다.  
즉, **코드 설치와 별개로 HF 권한/로그인**이 필요합니다.

#### (A) 모델 접근 권한 확인

아래 페이지에서 접근 상태를 확인합니다.

- [https://huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1)

페이지에서 `You have been granted access`가 보여야 다운로드 가능합니다.

#### (B) 토큰 생성

1. [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 이동  
2. `Create new token` 클릭  
3. `Read` 또는 fine-grained token 생성  
4. 권한은 공개 gated repo 읽기 권한이 포함되도록 설정

#### (C) CLI 로그인

```bash
python -m pip install -U huggingface_hub
hf auth login
```

프롬프트가 나오면:

- `Token:` 에 `hf_...` 토큰 붙여넣기
- `Add token as git credential? [y/N]:` 는 `N` 권장

로그인 확인:

```bash
hf auth whoami
```

#### (D) 보안 주의사항

- 토큰(`hf_...`)을 `.txt`, `.env`, 코드 파일에 평문 저장하지 마세요.
- 토큰이 노출되면 Hugging Face에서 즉시 revoke 후 재발급하세요.
- `.gitignore`에 민감 파일(`SAM.txt`, `.env*`) 제외 규칙을 반드시 유지하세요.

### 1-5. 프로젝트 의존성 설치

이 저장소 루트에서 실행:

```bash
python -m pip install -r requirements.txt
```

의존성 참고:

- `numpy>=1.26,<2` 고정 (SAM3 호환)
- `opencv-python==4.10.0.84` 고정
- `triton-windows<3.7`는 Windows에서만 설치되도록 조건부로 포함됨

---

## 2) git clone

이 프로젝트는 **현재 저장소 + SAM3 원본 저장소**가 모두 필요합니다.

> 중요한 점: 이 저장소를 `git clone`해도 `sam3` 저장소가 자동으로 clone되지는 않습니다.  
> 현재 저장소에는 `.gitmodules`가 없어서 submodule 자동 동기화가 설정되어 있지 않습니다.

### 2-1. 현재 프로젝트 clone

원격 저장소 URL이 있다면:

```bash
git clone <YOUR_REPO_URL>
cd sam-yolo-autolabel
```

이미 로컬 폴더가 있다면 해당 폴더로 이동만 하면 됩니다.

### 2-2. SAM3 저장소 clone 및 설치

`sam_yolo26_autolabel.py`는 `sam3` 패키지를 import하므로, SAM3 코드 설치가 필수입니다.

```bash
cd ..
git clone https://github.com/facebookresearch/sam3.git
cd sam3
python -m pip install -e .
cd ..
```

설치 확인:

```bash
python -c "from sam3.model.sam3_image_processor import Sam3Processor; print('sam3 import ok')"
```

업데이트 필요 시:

```bash
cd sam3
git pull
python -m pip install -e .
cd ..
```

---

## 3) JSON 파일 사용법

이제 이 프로젝트는 `--config` JSON 한 파일로 파라미터를 관리할 수 있습니다.

기본 설정 파일:

- `autolabel_config.json`

### 3-1. 설정 파일 구조

아래는 권장 템플릿입니다.

```json
{
  "images": "./dataset/input_images",
  "out": "./dataset/output_yolo",
  "classes": {
    "stainless_pipe": "vertical stainless steel tube with a circular opening"
  },
  "val_ratio": 0.2,
  "seed": 42,
  "shuffle_split": false,
  "device": "cuda",
  "amp_dtype": "bfloat16",
  "sam_version": "sam3.1",
  "min_score": 0.85,
  "min_area": 100,
  "simplify": 0.003,
  "max_points": 300,
  "checkpoint_path": null,
  "copy_mode": "copy",
  "overwrite": true,
  "no_save_previews": false
}
```

### 3-2. JSON 속성 설명

아래는 `autolabel_config.json`의 각 속성이 의미하는 내용입니다.

- `images`
  - 입력 이미지 루트 폴더 경로입니다.
  - 하위 폴더까지 재귀적으로 탐색하며 이미지 파일(`.jpg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`)을 읽습니다.
  - 상대 경로를 쓰면 `autolabel_config.json` 파일 위치 기준으로 해석됩니다.

- `out`
  - YOLO 데이터셋 출력 폴더입니다.
  - 실행 후 `images/`, `labels/`, `data.yaml`(및 선택적으로 `overlays/`, `masks/`)가 생성됩니다.

- `classes`
  - 클래스와 프롬프트를 정의하는 핵심 필드입니다.
  - 객체(dict) 또는 리스트(list) 형식을 지원합니다.
  - 문자열 경로(`"./classes_prompt.json"`)로 지정해서 외부 파일을 참조할 수도 있습니다.

- `val_ratio` (기본 `0.2`)
  - 검증 데이터 비율입니다.
  - `0 <= val_ratio < 1` 범위여야 합니다.
  - 예: `0.2`면 전체의 20%를 `val`로 분할합니다.

- `seed` (기본 `42`)
  - 분할 셔플 시 사용되는 랜덤 시드입니다.
  - 같은 시드와 같은 입력이면 동일한 분할 결과를 재현할 수 있습니다.

- `shuffle_split` (기본 `false`)
  - `true`면 분할 전에 이미지를 랜덤 셔플합니다.
  - `false`면 파일명을 자연 정렬한 순서로 분할합니다.

- `device` (기본 `"cuda"`)
  - 추론 디바이스를 지정합니다.
  - 허용값: `"cuda"`, `"cpu"`.
  - `cuda` 선택 시 PyTorch CUDA 환경이 정상이어야 합니다.

- `amp_dtype` (기본 `"bfloat16"`)
  - CUDA 자동 mixed precision dtype입니다.
  - 허용값: `"none"`, `"float16"`, `"bfloat16"`.
  - 구형 GPU에서 BF16 문제가 있으면 `"float16"`을 사용하세요.

- `sam_version` (기본 `"sam3.1"`)
  - `checkpoint_path`를 지정하지 않았을 때 Hugging Face에서 받을 SAM 계열 버전입니다.
  - 허용값: `"sam3.1"`, `"sam3"`.

- `min_score` (기본 `0.85`)
  - SAM이 반환한 마스크 score의 최소 임계값입니다.
  - 값이 높을수록 보수적으로 라벨이 생성됩니다.

- `min_area` (기본 `100`)
  - 폴리곤으로 변환할 최소 마스크 면적(픽셀)입니다.
  - 작은 노이즈 영역을 제거할 때 사용합니다.

- `simplify` (기본 `0.003`)
  - 폴리곤 단순화 계수입니다.
  - `epsilon = simplify * contour_perimeter`로 계산되어 점 개수를 줄입니다.
  - `0`이면 단순화를 적용하지 않습니다.

- `max_points` (기본 `300`)
  - 객체당 폴리곤 최대 점 개수입니다.
  - 단순화 후에도 점이 많으면 균등 샘플링으로 제한합니다.

- `checkpoint_path` (기본 `null`)
  - 로컬 체크포인트 파일 경로를 직접 지정할 때 사용합니다.
  - `null`이면 `sam_version` 기준으로 Hugging Face 다운로드를 사용합니다.

- `copy_mode` (기본 `"copy"`)
  - 원본 이미지를 출력셋으로 옮기는 방식입니다.
  - 허용값: `"copy"`, `"link"`.
  - `"link"`는 하드링크를 시도하고 실패하면 자동으로 복사합니다.

- `overwrite` (기본 `false`)
  - 출력 폴더가 이미 존재하고 비어 있지 않을 때 덮어쓰기 허용 여부입니다.
  - `false`면 안전하게 에러를 내고 중단합니다.

- `no_save_previews` (기본 `false`)
  - `true`면 `overlays/`, `masks/` 미리보기 이미지를 저장하지 않습니다.
  - 디스크 사용량/속도를 줄이고 싶을 때 유용합니다.

### 3-3. classes 작성 규칙

`classes`는 아래 2가지 형식을 지원합니다.

#### 형식 A: 객체(dict) 방식 (간단, 권장)

```json
{
  "classes": {
    "dome_nut": "gray round object with a central circular hole",
    "hex_nut": "hexagonal metal nut"
  }
}
```

- key: YOLO 클래스 이름
- value: SAM 텍스트 프롬프트
- 클래스 ID는 0부터 자동 할당

#### 형식 B: 리스트 방식 (ID 직접 지정 가능)

```json
{
  "classes": [
    { "id": 0, "name": "dome_nut", "prompt": "gray round object with a central circular hole" },
    { "id": 1, "name": "hex_nut", "prompt": "hexagonal metal nut" }
  ]
}
```

- `id`는 0부터 연속이어야 함
- 연속이 아니면 오류 발생

### 3-4. 기존 `classes_prompt.json` 통합

기존 `classes_prompt.json` 내용은 `autolabel_config.json`의 `classes`로 통합해서 사용하면 됩니다.  
즉, 앞으로는 `--classes classes_prompt.json` 없이도 실행 가능합니다.

### 3-5. 실행 방법

#### 기본 실행 (config만 사용)

```bash
python sam_yolo26_autolabel.py --config autolabel_config.json
```

#### CLI로 일부 값 덮어쓰기

CLI 인자가 config보다 우선합니다.

```bash
python sam_yolo26_autolabel.py --config autolabel_config.json --out ./dataset/output_yolo_v2 --device cpu
```

#### 클래스만 별도 파일로 분리하고 싶을 때

`config`에서 `classes`를 파일 경로 문자열로 지정할 수도 있습니다.

```json
{
  "images": "./dataset/input_images",
  "out": "./dataset/output_yolo",
  "classes": "./classes_prompt.json"
}
```

---

## 출력 결과 구조

```text
output_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  overlays/
    train/
    val/
  masks/
    train/
    val/
  data.yaml
```

- `images/`, `labels/`, `data.yaml`는 YOLO 학습에 필수
- `overlays/`, `masks/`는 품질 점검(QA)용

---

## 자주 쓰는 실행 예시

### 랜덤 분할 사용

```bash
python sam_yolo26_autolabel.py --config autolabel_config.json --shuffle-split --seed 42
```

### 미리보기 저장 끄기

```bash
python sam_yolo26_autolabel.py --config autolabel_config.json --no-save-previews
```

### FP16으로 실행

```bash
python sam_yolo26_autolabel.py --config autolabel_config.json --amp-dtype float16
```

---

## 트러블슈팅

### `ModuleNotFoundError: No module named 'sam3'`

```bash
cd ../sam3
python -m pip install -e .
```

### `CUDA was requested, but torch.cuda.is_available() is false`

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

CPU로 우선 실행하려면:

```bash
python sam_yolo26_autolabel.py --config autolabel_config.json --device cpu
```

### Hugging Face 인증 문제 (`403`, gated access 에러)

1. 모델 접근 권한 확인: [https://huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1)  
2. 토큰 재생성: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
3. 재로그인:

```bash
hf auth logout
hf auth login
hf auth whoami
```

### NumPy 관련 충돌

```bash
python -m pip uninstall numpy opencv-python -y
python -m pip install "numpy>=1.26,<2" "opencv-python==4.10.0.84"
```

---

## 참고 링크

- SAM3 GitHub: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SAM3.1 Hugging Face: [https://huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
- Hugging Face 토큰 설정: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Ultralytics YOLO 문서: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
