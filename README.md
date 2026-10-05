# 바이브컷 (VibeCut)

게임 플레이 녹화 영상(약 1시간)을 유튜브용 하이라이트(약 10분) 편집 초안으로 만들어 다빈치 리졸브로 넘겨주는 AI 에이전트.

## 설치

필요한 것: Python 3.11 이상, FFmpeg

```
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # 그다음 .env에 API 키 입력
```

## 실행

(모듈 구현 후 작성)
