# 출처·라이선스 기록

출처·AI 활용 신고서의 근거 자료다. 새 라이브러리, 모델, 외부 API, 공개 코드를 추가하면 한 줄씩 기록한다.

## 라이브러리·도구

|이름|용도|라이선스|
|-|-|-|
|FFmpeg|음성 추출, 무음 탐지, 영상 정보|LGPL 2.1+ / GPL (빌드에 따라 다름, Gyan 빌드는 GPL)|
|anthropic (Python SDK)|Claude API 호출, 에이전트 루프|MIT|
|openai (Python SDK)|Whisper 음성 인식 API 호출|Apache-2.0|
|python-dotenv|.env에서 API 키 읽기|BSD-3-Clause|
|PyYAML|config.yaml 읽기|MIT|
|NumPy|수치 계산|BSD-3-Clause|
|librosa|음량(RMS) 분석|ISC|
|soundfile|오디오 파일 읽기|BSD-3-Clause|
|PySceneDetect|장면 전환 탐지|BSD-3-Clause|
|OpenCV (opencv-python)|밝기·프레임 변화량 분석|Apache-2.0|
|OpenTimelineIO|다빈치 리졸브용 타임라인 내보내기|Apache-2.0|
|Gradio|UI|Apache-2.0|

## 외부 API·모델

|이름|용도|비고|
|-|-|-|
|Anthropic Claude API|편집 에이전트의 판단|모델명은 config.yaml `agent.model`|
|OpenAI Whisper API (whisper-1)|음성 인식, 단어 단위 타임스탬프|영상당 1회 호출 후 캐싱|

## AI 코딩 도구

|이름|용도|
|-|-|
|Claude Code|설계 문서 기반 코드 작성 보조|

## 테스트 영상·게임 영상 이용 정책

* 시연 영상: 사용자 본인의 게임 플레이 녹화 영상
* 게임명: (기록 필요)
* 게임사 영상 이용 정책: (확인 후 링크와 요약 기록 필요)
