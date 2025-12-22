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

# NEW: news article extraction
import requests
from bs4 import BeautifulSoup


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
# NEW: News URL -> Article text
# =========================

def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", (s or "").strip(), re.I))

def fetch_html(url: str, timeout: int = 15) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def extract_naver_news(html: str) -> tuple[str, str]:
    """
    Naver 뉴스(모바일/PC 공통)에서 제목/본문 추출.
    - 본문: div#dic_area 우선
    - 제목: og:title 또는 <title>
    """
    soup = BeautifulSoup(html, "html.parser")

    # title
    title = ""
    og = soup.select_one("meta[property='og:title']")
    if og and og.get("content"):
        title = og["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    # body
    body = ""
    dic = soup.select_one("#dic_area")
    if dic:
        body = dic.get_text("\n", strip=True)

    # 폴백 후보
    if not body:
        for sel in ["#articeBody", "#articleBodyContents", ".newsct_article", ".article_body", "article"]:
            el = soup.select_one(sel)
            if el:
                txt = el.get_text("\n", strip=True)
                if len(txt) > len(body):
                    body = txt

    body = re.sub(r"\s+\n", "\n", body).strip()
    return title, body

def extract_article_with_readability(html: str) -> tuple[str, str]:
    """
    DOM이 바뀌거나 일부 영역이 비어있을 때 폴백.
    """
    try:
        from readability import Document
    except Exception:
        return "", ""

    doc = Document(html)
    title = (doc.short_title() or "").strip()
    cleaned_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(cleaned_html, "html.parser")
    body = soup.get_text("\n", strip=True)
    body = re.sub(r"\s+\n", "\n", body).strip()
    return title, body

def fetch_article_text(url: str) -> tuple[str, str]:
    html = fetch_html(url)
    title, body = extract_naver_news(html)

    if len(body) < 400:
        t2, b2 = extract_article_with_readability(html)
        if len(b2) > len(body):
            title = title or t2
            body = b2

    if not body:
        raise RuntimeError("기사 본문을 추출하지 못했습니다. (DOM 변경/접근 제한 가능)")
    return title or "뉴스", body


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
    OpenAI Python SDK + Responses API 패턴 사용.
    """
    api_key = os.getenv("OPENAI_API_KEY")
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

    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    raw = resp.output_text or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"title": "untitled", "logline": "", "novel": raw}


# =========================
# Step 2: Script generation (Gemini Developer API)
# =========================

def make_genai_client_dev():
    # ✅ 절대 코드에 API 키를 하드코딩하지 마세요.
    # 환경변수 GOOGLE_API_KEY로 넣어야 안전합니다.
    api_key = os.getenv("GOOGLE_API_KEY")
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
        text = candidate

    # 2) 가장 첫 '{'부터 마지막 '}'까지 잘라보기
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text.strip()

def _gemini_force_json_only(*, client, model: str, sys: str, user: str, max_output_tokens: int) -> str:
    schema = ScriptPack.model_json_schema()

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=user)])],
        config=types.GenerateContentConfig(
            system_instruction=sys,
            max_output_tokens=max_output_tokens,
            temperature=0.4,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return resp.text or ""

def extract_first_json_value(s: str) -> str | None:
    s = s.strip()

    # 코드블록 제거
    if s.startswith("```"):
        s = s.strip("`").strip()
        if "\n" in s:
            first_line, rest = s.split("\n", 1)
            if first_line.lower().startswith("json"):
                s = rest.strip()

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

        if ch == '"':
            in_str = True
            continue

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
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    candidate = extract_first_json_value(s)
    if candidate is None:
        raise json.JSONDecodeError("No JSON object/array found", s, 0)

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

    raw = _gemini_force_json_only(
        client=client, model=model, sys=sys, user=user, max_output_tokens=max_output_tokens
    )

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

    raise RuntimeError("Unexpected JSON parse failure.")

# NEW: News -> ScriptPack (Gemini, JSON schema enforced)
def gemini_news_scriptify(
    *,
    article_title: str,
    article_body: str,
    client,
    model: str = "gemini-2.5-flash",
    max_scenes: int = 7,
    max_output_tokens: int = 4096,
) -> ScriptPack:
    sys = """
너는 유튜브 쇼츠(9:16)용 '뉴스 요약 이미지 기반 영상' 제작자야.
입력된 한국어 뉴스 기사(제목/본문)를 바탕으로:
- 30~45초 분량
- 장면은 최대 N개
- 사실 기반, 과장 금지, 기사에 없는 단정 금지
- 각 장면에 (title, visual, narration, image_prompt, sfx_tags, duration_sec_hint)을 채워 ScriptPack 스키마로 JSON만 출력한다.

중요:
- 반드시 JSON만 출력(추가 텍스트 금지).
- 첫 장면에서 전체 톤(스타일/색감/카메라)을 정의하고, 이후 장면 image_prompt에 반복해 일관성 유지.
""".strip()

    body_trim = article_body[:6000]

    user = f"""
[뉴스 제목]
{article_title}

[뉴스 본문]
{body_trim}

[요구사항]
- 장면 수는 최대 {max_scenes}개
- narration은 한국어(짧고 리듬감 있게)
- visual은 화면에 보이는 장면 설명(한국어)
- image_prompt는 세로 9:16, 시네마틱, 한국 뉴스/다큐 톤, 텍스트 삽입 금지
- duration_sec_hint는 3~7초 범위 권장

[출력 JSON 스키마]
{{
  "video_title": "쇼츠 제목(한국어, 60자 이내)",
  "video_description": "유튜브 설명(한국어, 2~4줄)",
  "tags": ["...", "..."],
  "scenes": [
    {{
      "idx": 1,
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

    raw = _gemini_force_json_only(
        client=client,
        model=model,
        sys=sys,
        user=user,
        max_output_tokens=max_output_tokens,
    )

    for attempt in range(2):
        try:
            data = try_load_json_loose(raw)
            # 스키마 검증으로 일관성 확보
            pack = ScriptPack.model_validate(data)
            # 장면 수 상한 방어
            if len(pack.scenes) > max_scenes:
                pack.scenes = pack.scenes[:max_scenes]
            return pack
        except Exception:
            if attempt == 1:
                raise RuntimeError(f"Gemini(뉴스)가 JSON을 깨뜨렸습니다. 원문:\n---\n{raw}\n---")
            repair_user = user + "\n\n너의 이전 출력은 JSON 파싱이 실패했다. JSON 객체만 다시 출력하라."
            raw = _gemini_force_json_only(
                client=client,
                model=model,
                sys=sys,
                user=repair_user,
                max_output_tokens=max_output_tokens,
            )

    raise RuntimeError("Unexpected JSON parse failure (news).")


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

    try:
        im = Image.open(BytesIO(img_bytes))
        im.save(out_path)
    except Exception:
        with open(out_path, "wb") as f:
            f.write(img_bytes)

    return out_path


# =========================
# Step 4: Shorts video render
#   - 1080x1920 (9:16)
#   - Veo(i2v) preferred (if configured), else ffmpeg zoompan fallback
# =========================

def write_concat_list(paths: List[str], list_path: str) -> str:
    """
    ffmpeg concat demuxer용 리스트 파일.
    ✅ f-string escape 문제 회피(사용자 syntax error 라인 해결)
    """
    ensure_dir(str(pathlib.Path(list_path).parent))
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            p2 = p.replace("\\", "/")
            p2 = p2.replace("'", "'\\''")
            f.write("file '" + p2 + "'\n")
    return list_path


def can_use_veo(cfg: "PipelineConfig") -> bool:
    if not cfg.use_veo:
        return False
    return bool(os.getenv(cfg.veo_api_key_env))


def generate_scene_video_with_veo(
    *,
    cfg: "PipelineConfig",
    image_path: str,
    prompt: str,
    duration: float,
    out_path: str,
    size: Tuple[int, int] = (1080, 1920),
    fps: int = 30,
) -> str:
    """
    (스텁) Veo 이미지→비디오 생성.
    - 실제 구현은 Vertex AI / Gemini API 방식에 맞게 교체 필요.
    """
    # TODO:
    # 1) 이미지 bytes 로드
    # 2) Veo i2v 요청(prompt + image + duration + aspect(9:16) + fps)
    # 3) 응답 video bytes를 out_path에 저장
    raise NotImplementedError("Veo 호출 구현이 필요합니다. (환경/SDK에 맞게 채우세요)")


def render_scene_video_9x16_ffmpeg(
    image_path: str,
    duration: float,
    out_path: str,
    size: Tuple[int, int] = (1080, 1920),
    fps: int = 30,
    zoom: float = 1.10,
) -> str:
    """
    (폴백) 한 장면 = 한 이미지로 duration초 영상 생성.
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


def render_scene_video_9x16(
    *,
    cfg: "PipelineConfig",
    image_path: str,
    duration: float,
    out_path: str,
    scene_prompt: str,
    size: Tuple[int, int] = (1080, 1920),
    fps: int = 30,
    zoom: float = 1.10,
) -> str:
    """
    고퀄 우선: Veo(i2v) → 실패/미설정 시 ffmpeg 줌팬 폴백
    """
    ensure_dir(str(pathlib.Path(out_path).parent))

    if can_use_veo(cfg):
        try:
            return generate_scene_video_with_veo(
                cfg=cfg,
                image_path=image_path,
                prompt=scene_prompt,
                duration=duration,
                out_path=out_path,
                size=size,
                fps=fps,
            )
        except Exception as e:
            logging.warning(f"Veo failed, fallback to ffmpeg zoompan: {e}")

    return render_scene_video_9x16_ffmpeg(
        image_path=image_path,
        duration=duration,
        out_path=out_path,
        size=size,
        fps=fps,
        zoom=zoom,
    )


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

    # Existing test flag (kept)
    use_test_novel_for_gemini: bool = True

    # NEW: Veo preference (i2v)
    use_veo: bool = True
    veo_model: str = "veo-3.1"  # placeholder label
    veo_api_key_env: str = "GOOGLE_API_KEY"  # if you use a different key, change this


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
    # NEW: URL input -> news mode
    # -------------------------
    news_mode = False
    article_title = ""
    article_body = ""

    if is_url(issue_context):
        news_mode = True
        logging.info(f"[INPUT] URL detected -> news mode: {issue_context}")
        article_title, article_body = fetch_article_text(issue_context)
        logging.info(f"[NEWS] title={article_title} body_len={len(article_body)}")

    # -------------------------
    # 1) NOVEL (OpenAI) with cache
    #    - skipped in news_mode
    # -------------------------
    novel = ""
    if not news_mode:
        novel_json_path = os.path.join(texts_dir, "novel.json")
        novel_txt_path  = os.path.join(texts_dir, "novel.txt")

        novel_pack = load_json(novel_json_path)
        if cfg.use_test_novel_for_gemini:
            logging.info("TEST MODE: GPT 생략, 임시 텍스트로 Gemini 테스트")
            novel = TEST_NOVEL_TEXT
            save_text(os.path.join(texts_dir, "novel_test.txt"), novel)
        else:
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
    #    - in news_mode: article -> ScriptPack
    # -------------------------
    script_json_path = os.path.join(texts_dir, "script.json")

    script_data = load_json(script_json_path)
    if script_data:
        logging.info("CACHE HIT: texts/script.json -> Gemini 호출 생략")
        script = ScriptPack.model_validate(script_data)
    else:
        logging.info("CACHE MISS: texts/script.json 없음 -> Gemini로 스크립트 생성")
        gen_client = make_genai_client_dev()

        if news_mode:
            script = gemini_news_scriptify(
                article_title=article_title,
                article_body=article_body,
                client=gen_client,
                model=cfg.gemini_text_model,
                max_scenes=7,
            )
        else:
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

    for p in img_paths:
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            raise RuntimeError(f"이미지 생성 실패 또는 파일 없음: {p}")

    # -------------------------
    # 4) SCENE RENDER (Veo preferred, fallback ffmpeg) with cache
    # -------------------------
    scene_videos: List[str] = []
    for sc, imgp, dur in zip(script.scenes, img_paths, durations):
        out_clip = os.path.join(scenes_dir, f"clip_{sc.idx:02d}.mp4")

        if os.path.exists(out_clip) and os.path.getsize(out_clip) > 0:
            logging.info(f"CACHE HIT: 장면 영상 존재 -> {out_clip}")
        else:
            logging.info(f"RENDER: 장면 영상 생성 -> {out_clip}")
            # Veo에 줄 프롬프트는 "visual + narration + image_prompt"를 합쳐 좀 더 안정적으로
            scene_prompt = (
                f"[SCENE TITLE] {sc.title}\n"
                f"[VISUAL] {sc.visual}\n"
                f"[NARRATION] {sc.narration}\n"
                f"[IMAGE PROMPT] {sc.image_prompt}\n"
                f"[STYLE] vertical 9:16, cinematic, documentary/news tone, no on-screen text"
            )

            render_scene_video_9x16(
                cfg=cfg,
                image_path=imgp,
                duration=dur,
                out_path=out_clip,
                scene_prompt=scene_prompt,
                size=cfg.shorts_size,
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
        use_test_novel_for_gemini=True,  # ✅ 뉴스 URL을 넣으면 이 플래그와 무관하게 news_mode로 동작
        use_veo=True,                    # ✅ 키+구현이 있으면 Veo 우선, 아니면 자동 폴백
    )

    # ✅ 예시 1) 뉴스 링크로 실행
    out = run_pipeline_shorts("https://n.news.naver.com/article/008/0005295323?cds=news_media_pc&type=editn", cfg)
    print(out)

    # ✅ 예시 2) 기존 텍스트(소설 파이프라인)로 실행하고 싶으면 아래처럼:
    # cfg.use_test_novel_for_gemini = False
    # out2 = run_pipeline_shorts("어떤 사회 이슈 텍스트", cfg)
    # print(out2)
