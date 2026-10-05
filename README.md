# 바이브컷 (VibeCut)

게임 플레이 녹화 영상(약 1시간)을 유튜브용 하이라이트 편집 초안으로 만들어 **다빈치 리졸브**로 넘겨주는 AI 에이전트.
편집자가 원하는 방향을 말로 설명하면, 에이전트가 장면을 찾고 컷을 구성해 타임라인·자막·마커 파일로 내보낸다.
최종 편집은 편집자가 다빈치 리졸브에서 한다.

## 흐름

```
게임 녹화 영상
 → 음성 추출 → 음성 인식(Whisper API) → 신호 탐지(음량·반응·화면·무음·지루한 구간) → 하이라이트 점수화
 → 편집 에이전트(Claude): 목표 이해 → 후보 조회 → 컷 구성 → 길이 검증·재조정 → 애매한 장면 질문 → 선호도 저장
 → timeline.otio + subtitles.srt + markers.edl
 → 다빈치 리졸브에서 열기
```

## 설치 (Windows)

필요한 것: Python 3.11 이상, FFmpeg, Anthropic API 키, OpenAI API 키

```
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

설치 후 **터미널을 새로 열고** 프로젝트 폴더에서:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

`.env`에 키를 입력한다 (`.env`는 git에 올라가지 않는다).

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## 실행

### 화면으로 실행 (권장)

```
python cli.py ui
```

브라우저에서 http://127.0.0.1:7860 을 연다.

1. **영상**: `data/` 폴더의 영상을 고르거나 업로드하고 [분석]을 누른다. (영상당 한 번, 이후 캐시 사용)
2. **편집 에이전트**: 채팅으로 방향을 말한다. 예: `10분짜리로, 반응 크고 재밌는 장면 위주로 만들어줘`
3. 에이전트가 애매한 장면을 물으면 왼쪽 **구간 미리보기**로 확인하고 답한다.
4. **판단 로그**에서 에이전트의 판단 → 도구 실행 → 결과와 호출별 비용을 볼 수 있다.
5. **편집 결과**에서 클립 목록을 확인하고 다빈치 리졸브용 파일을 받는다.

### 명령어로 단계별 실행

| 명령 | 하는 일 |
|-|-|
| `python cli.py analyze data/영상.mp4` | 아래 1~4단계를 한 번에 |
| `python cli.py ingest data/영상.mp4` | 1. 영상 등록: video_id, 영상 정보, 음성 추출, 분석용 저해상도 사본 |
| `python cli.py transcribe <video_id>` | 2. 음성 인식 (영상당 1회 호출, 캐시) |
| `python cli.py detect all <video_id>` | 3. 신호 탐지 (`all` 대신 silence / loudness / reaction / visual / dead_time) |
| `python cli.py detect loudness <video_id> --set threshold_db=6` | 기준값을 바꿔 특정 탐지기만 다시 실행 |
| `python cli.py score <video_id>` | 4. 하이라이트 점수화 |
| `python cli.py edit <video_id>` | 5. 편집 에이전트와 대화 (빈 줄 입력 시 종료) |
| `python cli.py export <video_id>` | 6. 최신 편집 결정 목록을 다빈치 리졸브용 파일로 내보내기 |

결과는 `data/cache/<video_id>/` 아래에 저장된다. 에이전트 판단 로그는 `logs/`에 남는다.

## 다빈치 리졸브에서 열기 (18.5 이상)

결과 폴더: `data/cache/<video_id>/exports/<시각>/`

1. 새 프로젝트 → Edit 페이지 → **File → Import → Timeline** → `timeline.otio`
   - 옵션에서 *Automatically set project settings*, *Automatically import source clips into media pool* 체크
   - 원본 영상이 자동으로 연결된다. 원본 파일을 옮기면 미디어 풀에서 Relink 한다.
2. 클립마다 마커(장면 이름, 선택 이유)가 붙어 있다. 안 보이면 타임라인 우클릭 → **Timelines → Import → Timeline Markers from EDL** → `markers.edl`
3. `subtitles.srt`를 미디어 풀에 넣고 타임라인 위로 끌어 올리면 컷에 맞춘 자막 트랙이 생긴다.

## 조정 가능한 값

모든 기준값은 `config.yaml`에 있다. 주요 항목:

* `detectors.*` — 무음·음량 급증·반응 키워드·장면 민감도·지루한 구간 기준
* `scoring.weights` — 하이라이트 점수 가중치
* `edit` — 클립 경계 보정 (말 시작 전 여유 1초, 2초 이하 간격은 합치기)
* `agent.model`, `agent.effort` — 에이전트 모델과 판단 깊이 (`claude-sonnet-5-5`로 바꾸면 비용 절반)

## 비용 참고 (추정)

* 음성 인식: 영상당 1회, 1시간에 약 $0.36
* 에이전트: 편집 1회에 약 $0.3~1 (판단 로그에 실제 비용이 기록된다)

## 폴더 구조

```
cli.py            단계별 실행 명령어
config.yaml       조정 가능한 수치
vibecut/
  media/          음성 추출, 영상 정보, 분석용 사본, 미리보기
  stt/            음성 인식 (공통 인터페이스 + OpenAI Whisper 어댑터)
  detectors/      silence, loudness, reaction, visual, dead_time
  scoring/        하이라이트 점수화
  agent/          에이전트 루프, 도구, 프롬프트, 선호도 메모리, 판단 로그
  export/         다빈치 리졸브 내보내기 (OTIO, SRT, 마커 EDL)
  storage/        분석 결과·편집 결정 목록 저장
  ui/             Gradio 화면
data/             (git 제외) 원본 영상, 분석 캐시
memory/           (git 제외) 사용자 선호도
logs/             (git 제외) 에이전트 판단 로그
```

출처·라이선스는 [SOURCES.md](SOURCES.md), 설계 기준은 [CLAUDE.md](CLAUDE.md)를 참고.
