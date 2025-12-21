import os
import re
import json
import time
import uuid
import random
import pathlib
import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import feedparser
from pydantic import BaseModel, Field, ValidationError

# OpenAI GPT (replaces anthropic/claude)
from openai import OpenAI

# Gemini Developer API (google-genai)
from google import genai
from google.genai import types

# Pillow (image write)
from PIL import Image
from io import BytesIO


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =========================
# Schemas
# =========================

class Scene(BaseModel):
    idx: int
    title: str
    visual: str
    narration: str
    dialogue: Optional[str] = None
    image_prompt: str
    sfx_tags: List[str] = Field(default_factory=list)
    duration_sec_hint: Optional[float] = None


class ScriptPack(BaseModel):
    video_title: str
    video_description: str
    tags: List[str] = Field(default_factory=list)
    scenes: List[Scene]


# =========================
# Utils
# =========================

def ensure_dir(p: str) -> str:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

def save_text(path: str, text: str) -> None:
    ensure_dir(str(pathlib.Path(path).parent))
    pathlib.Path(path).write_text(text, encoding="utf-8")

def safe_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r"[^\w\s\-\.\(\)\[\]]+", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    return (name[:max_len].strip() or str(uuid.uuid4()))

def seconds_for_text_korean(text: str, wpm: int = 160) -> float:
    # 쇼츠는 템포가 빠르니 wpm을 조금 높게 잡음
    words = max(1, len(text.split()))
    minutes = words / max(60, wpm)
    return minutes * 60.0

def get_ffmpeg_path() -> str:
    """
    1) 환경변수 FFMPEG_PATH가 있으면 그걸 사용
    2) 없으면 PATH에서 ffmpeg를 찾음
    """
    p = os.getenv("FFMPEG_PATH")
    if p and os.path.exists(p):
        return p
    return "ffmpeg"

def run_ffmpeg(args: List[str]) -> None:
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg] + args
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg 실행파일을 찾지 못했습니다.\n"
            "해결 방법:\n"
            "1) ffmpeg 설치 후 PATH 등록\n"
            "   - Windows: winget install Gyan.FFmpeg  (또는 chocolatey/scoop)\n"
            "2) 또는 FFMPEG_PATH 환경변수에 ffmpeg.exe 전체 경로 지정\n"
            "   예: setx FFMPEG_PATH \"C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe\"\n"
        ) from e


# =========================
# Step 0: Google News RSS (optional)
# =========================

def fetch_google_news_headlines_kr(limit: int = 12) -> List[Dict[str, str]]:
    url = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:limit]:
        items.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", "")
        })
    return items


# =========================
# Step 1: Issue -> Novel (GPT via OpenAI Responses API)
# =========================

def gpt_make_novel(
    issue_context: str,
    model: str = "gpt-4.1",
    max_output_tokens: int = 2500,
) -> Dict[str, str]:
    """
    Claude 대신 OpenAI GPT로 소설 생성.
    OpenAI Python SDK + Responses API 패턴 사용. :contentReference[oaicite:2]{index=2}
    """
    #api_key = os.getenv("OPENAI_API_KEY")
    api_key = "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"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
너는 사회 이슈를 '현대 한국 배경의 단편소설'로 바꾸는 작가야.

[입력 이슈/맥락]
{issue_context}

[요구사항]
- 길이: 1,200~1,800자(한국어)
- 인물 2~3명, 갈등-전개-반전-여운 구조
- 선정적/혐오/차별 조장 금지, 실존인물 비방 금지
- 마지막에 한 문장 여운(한 줄)

[출력 형식: JSON만]
{{
  "title": "...",
  "logline": "...",
  "novel": "..."
}}
""".strip()

    # Responses API: output_text로 텍스트 얻는 방식이 권장 흐름 :contentReference[oaicite:3]{index=3}
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    raw = resp.output_text or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # JSON이 깨졌을 때도 파이프라인이 죽지 않도록 최소 복구
        return {"title": "untitled", "logline": "", "novel": raw}


# =========================
# Step 2: Novel -> Script (Gemini Developer API)
# =========================

def make_genai_client_dev():
    # ✅ 절대 코드에 API 키를 하드코딩하지 마세요.
    # 환경변수 GOOGLE_API_KEY로 넣어야 안전합니다.
    api_key = "AIzaSyAjIHVyGxdJ_ek3icPrd47RnwaX7kNow0U"
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY 환경변수가 필요합니다 (Gemini Developer API).")
    return genai.Client(api_key=api_key)

def _extract_json_object(text: str) -> str:
    """
    Gemini가 앞뒤로 설명을 붙여도, 가장 바깥 JSON 객체 하나를 최대한 추출.
    """
    if not text:
        return ""

    # 1) ```json ... ``` 코드블록이면 그 안을 우선
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        # candidate가 JSON 객체/배열일 가능성이 높음
        text = candidate

    # 2) 가장 첫 '{'부터 마지막 '}'까지 잘라보기
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text.strip()


def _gemini_force_json_only(*, client, model: str, sys: str, user: str, max_output_tokens: int) -> str:
    # Pydantic 스키마를 dict(JSON schema)로 뽑아서 넣어도 되고,
    # google-genai의 types.Schema를 직접 구성해도 됩니다.
    schema = ScriptPack.model_json_schema()

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=user)])],
        config=types.GenerateContentConfig(
            system_instruction=sys,                      # ✅ system을 진짜 system으로
            max_output_tokens=max_output_tokens,
            temperature=0.4,
            response_mime_type="application/json",       # ✅ JSON 모드
            response_schema=schema,                      # ✅ 스키마 강제
        ),
    )
    return resp.text or ""

def extract_first_json_value(s: str) -> str | None:
    s = s.strip()

    # 코드블록 제거
    if s.startswith("```"):
        s = s.strip("`").strip()
        # ```json\n ... \n``` 형태도 대충 제거
        if "\n" in s:
            first_line, rest = s.split("\n", 1)
            if first_line.lower().startswith("json"):
                s = rest.strip()

    # 시작 위치 찾기
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None:
        return None

    stack = []
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # 문자열 시작
        if ch == '"':
            in_str = True
            continue

        # 구조 문자 처리
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            top = stack.pop()
            if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                return None
            if not stack:
                return s[start:i+1]

    return None

def try_load_json_loose(raw: str) -> dict:
    s = raw.strip()

    # 이상 따옴표 정리
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    # JSON 덩어리만 정확히 추출
    candidate = extract_first_json_value(s)
    if candidate is None:
        raise json.JSONDecodeError("No JSON object/array found", s, 0)

    # 트레일링 콤마 제거
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    return json.loads(candidate)

def gemini_scriptify(
    novel: str,
    *,
    client,
    model: str = "gemini-2.5-flash",
    max_output_tokens: int = 4096,
) -> ScriptPack:
    sys = """
너는 유튜브 쇼츠(9:16)용 '이미지 기반 스토리 영상'의 각본가야.
소설을 8~12개 장면으로 나누고, 각 장면마다:
- 화면에 보이는 것(visual)
- 내레이션(narration): 한국어, 쇼츠 템포(짧고 리듬감 있게)
- 이미지 생성용 프롬프트(image_prompt): 세로 9:16 구도, 시네마틱, 한국 배경, 일관된 스타일
- 효과음 태그(sfx_tags)
- duration_sec_hint(권장 3~7초)
을 만든다.

중요:
- 반드시 JSON만 출력(추가 텍스트 금지).
- 첫 장면에서 주인공/조연 외형을 구체적으로 정의하고, 이후 장면 prompt에 반복해 일관성 유지.
""".strip()

    user = f"""
[소설]
{novel}

[출력 JSON 스키마]
{{
  "video_title": "쇼츠 제목(한국어, 60자 이내)",
  "video_description": "유튜브 설명(한국어, 2~4줄)",
  "tags": ["...", "..."],
  "scenes": [
    {{
      "idx": 0,
      "title": "...",
      "visual": "...",
      "narration": "...",
      "dialogue": null,
      "image_prompt": "...",
      "sfx_tags": ["...","..."],
      "duration_sec_hint": 5.0
    }}
  ]
}}

[출력 규칙]
- JSON만 출력. 절대 다른 문장/설명/백틱 금지.
""".strip()

    # 1) 1차 시도
    raw = _gemini_force_json_only(
        client=client, model=model, sys=sys, user=user, max_output_tokens=max_output_tokens
    )
    j = _extract_json_object(raw)

    # 2) 파싱 실패 시 1회 재시도(더 강하게)
    for attempt in range(2):
        try:
            data = try_load_json_loose(raw)
            return ScriptPack.model_validate(data)
        except Exception:
            if attempt == 1:
                raise RuntimeError(f"Gemini가 JSON을 깨뜨렸습니다. 원문:\n---\n{raw}\n---")

            repair_user = user + "\n\n너의 이전 출력은 JSON 파싱이 실패했다. JSON 객체만 다시 출력하라."
            raw = _gemini_force_json_only(
                client=client, model=model, sys=sys, user=repair_user, max_output_tokens=max_output_tokens
            )
            j = _extract_json_object(raw)

    # 여기에 도달하진 않음
    raise RuntimeError("Unexpected JSON parse failure.")


# =========================
# Step 3: Images (Gemini Flash Image / nano-banana 계열)
# =========================

def gen_image_with_gemini_image(
    *,
    client,
    prompt: str,
    out_path: str,
    model: str = "gemini-2.5-flash-image-preview",
) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.7),
    )

    img_bytes = None
    for cand in (resp.candidates or []):
        for part in (cand.content.parts or []):
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                img_bytes = part.inline_data.data
                break
        if img_bytes:
            break

    if not img_bytes:
        raise RuntimeError("이미지 데이터를 받지 못했습니다(응답에 inline_data 없음).")

    ensure_dir(str(pathlib.Path(out_path).parent))

    # 일부 SDK/모델은 bytes가 바로 PNG가 아닌 경우가 있어 Pillow로 한 번 로드/저장 시도
    try:
        im = Image.open(BytesIO(img_bytes))
        im.save(out_path)
    except Exception:
        with open(out_path, "wb") as f:
            f.write(img_bytes)

    return out_path


# =========================
# Step 4: Shorts video render (FFmpeg only)
#   - 1080x1920 (9:16)
#   - center crop + slight zoom
# =========================

def write_concat_list(paths: List[str], list_path: str) -> str:
    """
    ffmpeg concat demuxer용 리스트 파일.
    ✅ f-string escape 문제 회피(사용자 syntax error 라인 해결)
    """
    ensure_dir(str(pathlib.Path(list_path).parent))
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            # ffmpeg concat: file 'path'
            # 작은따옴표는  ' -> '\''
            p2 = p.replace("\\", "/")
            p2 = p2.replace("'", "'\\''")
            f.write("file '" + p2 + "'\n")
    return list_path

def render_scene_video_9x16(
    image_path: str,
    duration: float,
    out_path: str,
    size: Tuple[int, int] = (1080, 1920),
    fps: int = 30,
    zoom: float = 1.10,
) -> str:
    """
    한 장면 = 한 이미지로 duration초 영상 생성.
    - 이미지 비율 무관하게 9:16으로 center crop
    - duration 동안 약한 줌 인
    """
    w, h = size
    ensure_dir(str(pathlib.Path(out_path).parent))

    frames = max(1, int(duration * fps))
    inc = (zoom - 1.0) / max(1, frames)

    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},"
        f"zoompan=z='1+{inc}*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h},"
        f"fps={fps}"
    )

    run_ffmpeg([
        "-y",
        "-loop", "1",
        "-i", image_path,
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path
    ])
    return out_path

def concat_videos(video_paths: List[str], out_path: str) -> str:
    """
    여러 mp4를 concat demuxer로 결합(코덱 동일 전제).
    """
    ensure_dir(str(pathlib.Path(out_path).parent))
    list_path = os.path.join(str(pathlib.Path(out_path).parent), "concat_list.txt")
    write_concat_list(video_paths, list_path)

    run_ffmpeg([
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_path
    ])
    return out_path


# =========================
# Step 5: (템플릿) 무음 오디오 트랙 추가
# =========================

def add_silent_audio(video_path: str, out_path: str) -> str:
    """
    유튜브 업로드/편집 호환을 위해 무음 오디오 트랙 추가.
    """
    ensure_dir(str(pathlib.Path(out_path).parent))
    run_ffmpeg([
        "-y",
        "-i", video_path,
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path
    ])
    return out_path


# =========================
# Orchestrator (Shorts only)
# =========================

@dataclass
class PipelineConfig:
    workdir: str = "workdir_shorts"
    # GPT
    gpt_model: str = "gpt-4.1"
    # Gemini
    gemini_text_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-2.5-flash-image-preview"
    # Shorts video
    shorts_size: Tuple[int, int] = (1080, 1920)
    fps: int = 30
    min_scene_sec: float = 3.0
    max_scene_sec: float = 7.0
    use_test_novel_for_gemini: bool = True

def load_json(path: str) -> Optional[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def script_base_dir() -> str:
    """
    workdir_shorts가 '실행 위치'가 아니라 '코드 파일 위치' 기준으로 생기게 해서
    폴더를 못 찾는 문제를 줄임.
    """
    try:
        return str(pathlib.Path(__file__).resolve().parent)
    except NameError:
        # 인터프리터/노트북 등에서 __file__이 없을 수 있음
        return os.getcwd()
    
def run_pipeline_shorts(issue_context: str, cfg: PipelineConfig) -> str:
    base = script_base_dir()
    wd = ensure_dir(os.path.join(base, cfg.workdir))

    texts_dir  = ensure_dir(os.path.join(wd, "texts"))
    images_dir = ensure_dir(os.path.join(wd, "images"))
    scenes_dir = ensure_dir(os.path.join(wd, "scenes"))
    final_dir  = ensure_dir(os.path.join(wd, "final"))

    logging.info(f"[WORKDIR] {wd}")
    logging.info("[OUTPUTS]")
    logging.info(f" - novel:  {os.path.join(texts_dir, 'novel.json')}")
    logging.info(f" - script: {os.path.join(texts_dir, 'script.json')}")
    logging.info(f" - images: {images_dir}")
    logging.info(f" - scenes: {scenes_dir}")
    logging.info(f" - final:  {final_dir}")

    # -------------------------
    # 1) NOVEL (OpenAI) with cache
    # -------------------------
    novel_json_path = os.path.join(texts_dir, "novel.json")
    novel_txt_path  = os.path.join(texts_dir, "novel.txt")

    novel_pack = load_json(novel_json_path)
    if cfg.use_test_novel_for_gemini:
        logging.info("TEST MODE: GPT 생략, 임시 텍스트로 Gemini 테스트")
        novel = TEST_NOVEL_TEXT
        save_text(os.path.join(texts_dir, "novel_test.txt"), novel)
    else:
        novel_json_path = os.path.join(texts_dir, "novel.json")
        novel_txt_path  = os.path.join(texts_dir, "novel.txt")

        novel_pack = load_json(novel_json_path)
        if novel_pack and isinstance(novel_pack, dict) and novel_pack.get("novel"):
            logging.info("CACHE HIT: texts/novel.json -> OpenAI 호출 생략")
        else:
            logging.info("CACHE MISS: texts/novel.json 없음/깨짐 -> OpenAI로 소설 생성")
            novel_pack = gpt_make_novel(issue_context, model=cfg.gpt_model)
            save_text(novel_json_path, json.dumps(novel_pack, ensure_ascii=False, indent=2))
            save_text(novel_txt_path, novel_pack.get("novel", ""))

        novel = (novel_pack or {}).get("novel", "")
    if not novel.strip():
        raise RuntimeError("novel이 비어있습니다. texts/novel.json을 확인하세요.")

    # -------------------------
    # 2) SCRIPT (Gemini) with cache
    # -------------------------
    script_json_path = os.path.join(texts_dir, "script.json")

    script_data = load_json(script_json_path)
    if script_data:
        logging.info("CACHE HIT: texts/script.json -> Gemini scriptify 호출 생략")
        script = ScriptPack.model_validate(script_data)
    else:
        logging.info("CACHE MISS: texts/script.json 없음 -> Gemini로 스크립트 생성")
        gen_client = make_genai_client_dev()
        script = gemini_scriptify(novel, client=gen_client, model=cfg.gemini_text_model)
        save_text(script_json_path, script.model_dump_json(indent=2, ensure_ascii=False))

    if not script.scenes:
        raise RuntimeError("script.scenes가 비어있습니다. texts/script.json을 확인하세요.")

    # -------------------------
    # 3) IMAGES (Gemini Image) with cache
    # -------------------------
    gen_client = None  # 이미지 생성이 필요할 때만 초기화
    img_paths: List[str] = []
    durations: List[float] = []

    for sc in script.scenes:
        out_img = os.path.join(images_dir, f"scene_{sc.idx:02d}.png")

        # duration 계산(항상 필요)
        d = sc.duration_sec_hint or seconds_for_text_korean(sc.narration)
        d = float(max(cfg.min_scene_sec, min(cfg.max_scene_sec, d)))
        durations.append(d)

        if os.path.exists(out_img) and os.path.getsize(out_img) > 0:
            logging.info(f"CACHE HIT: 이미지 존재 -> {out_img}")
        else:
            logging.info(f"GEN: 이미지 생성 -> {out_img}")
            if gen_client is None:
                gen_client = make_genai_client_dev()

            prompt = (
                sc.image_prompt
                + "\n\n[출력 제약]\n"
                  "- 반드시 세로 9:16(쇼츠)\n"
                  "- 피사체 중앙, 상하 여백 고려(자막 공간)\n"
                  "- 텍스트(글자) 삽입 금지\n"
            )
            gen_image_with_gemini_image(
                client=gen_client,
                prompt=prompt,
                out_path=out_img,
                model=cfg.gemini_image_model,
            )

        img_paths.append(out_img)

    # 이미지 파일이 실제로 만들어졌는지 안전 체크
    for p in img_paths:
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            raise RuntimeError(f"이미지 생성 실패 또는 파일 없음: {p}")

    # -------------------------
    # 4) SCENE RENDER (ffmpeg) with cache
    # -------------------------
    scene_videos: List[str] = []
    for sc, imgp, dur in zip(script.scenes, img_paths, durations):
        out_clip = os.path.join(scenes_dir, f"clip_{sc.idx:02d}.mp4")

        if os.path.exists(out_clip) and os.path.getsize(out_clip) > 0:
            logging.info(f"CACHE HIT: 장면 영상 존재 -> {out_clip}")
        else:
            logging.info(f"RENDER: 장면 영상 생성 -> {out_clip}")
            render_scene_video_9x16(
                image_path=imgp,
                duration=dur,
                out_path=out_clip,
                size=cfg.shorts_size,   # 1080x1920
                fps=cfg.fps,
                zoom=1.10,
            )
        scene_videos.append(out_clip)

    # -------------------------
    # 4.5) CONCAT
    # -------------------------
    raw = os.path.join(final_dir, "raw_concat.mp4")
    if os.path.exists(raw) and os.path.getsize(raw) > 0:
        logging.info("CACHE HIT: final/raw_concat.mp4 존재 -> concat 생략")
    else:
        logging.info("CONCAT: 장면 결합 -> final/raw_concat.mp4")
        concat_videos(scene_videos, raw)

    # -------------------------
    # 5) ADD SILENT AUDIO (template)
    # -------------------------
    final_name = safe_filename(script.video_title) + "_SHORTS.mp4"
    final_path = os.path.join(final_dir, final_name)

    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
        logging.info(f"CACHE HIT: 최종 파일 존재 -> {final_path}")
    else:
        logging.info(f"MUX: 무음 오디오 트랙 추가 -> {final_path}")
        add_silent_audio(raw, final_path)

    logging.info(f"[DONE] FINAL VIDEO = {final_path}")
    return final_path

TEST_NOVEL_TEXT = """
서울의 겨울 밤, 지하철 막차를 놓친 민수는 편의점 앞에서 우연히 고등학교 동창 지연을 만난다.
두 사람은 각자 다른 삶을 살고 있었고, 짧은 대화 속에서 서로의 실패와 후회를 조심스럽게 꺼낸다.
눈이 내리는 거리에서, 민수는 지금의 삶을 계속 이렇게 살아도 되는지 스스로에게 묻는다.
지연은 말없이 미소를 지으며, 내일은 오늘과 다를 수 있다고 말한다.
그날 밤 이후 민수는 작은 선택 하나를 바꾸기로 결심한다.
""".strip()

if __name__ == "__main__":
    cfg = PipelineConfig(
        use_test_novel_for_gemini=True  # ✅ 여기서 제어
    )
    out = run_pipeline_shorts("TEST ISSUE (unused)", cfg)
    print(out)
