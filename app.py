import io
import os
import re
import json
import math
import base64
import socket
import statistics
import urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_fontja  # noqa: F401  matplotlibに日本語フォントを設定
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
import anthropic

app = Flask(__name__)

# ── 画像ストア ──
_photo_bytes:   bytes | None = None
_cover_bytes:   bytes | None = None
_tasting_bytes: bytes | None = None
_sticker_bytes: bytes | None = None   # iPhone切り抜きステッカー（透過PNG）

# ── レーダーチャートキャッシュ（スコアが同じなら再生成しない）──
_chart_cache: dict = {"key": None, "img": None}

# ── 日本語フォント（PIL用） ──
def _find_font() -> str | None:
    # Windows
    for p in [r"C:\Windows\Fonts\YuGothB.ttc",
              r"C:\Windows\Fonts\meiryo.ttc",
              r"C:\Windows\Fonts\msgothic.ttc"]:
        if os.path.exists(p):
            return p
    # matplotlib-fontja が同梱しているIPAexゴシックを流用
    try:
        import matplotlib_fontja as mf
        fp = os.path.join(os.path.dirname(mf.__file__), "fonts", "ipaexg.ttf")
        if os.path.exists(fp):
            return fp
    except Exception:
        pass
    return None

_JP_FONT = _find_font()

# ── フォント管理（ゴシック / 明朝）──
_FONT_PATHS: dict[str, str | None] = {"gothic": _JP_FONT, "mincho": None}

def _download_mincho() -> None:
    """IPAex明朝をバックグラウンドでダウンロードしてキャッシュ"""
    cache_dir = "/tmp/ieditor_fonts"
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "ipaexm.ttf")
    if os.path.exists(path) and os.path.getsize(path) > 500_000:
        _FONT_PATHS["mincho"] = path
        return
    urls = [
        "https://raw.githubusercontent.com/ootaharuki99/IPAexfont/master/ipaexm.ttf",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if len(data) > 500_000:
                with open(path, "wb") as f:
                    f.write(data)
                _FONT_PATHS["mincho"] = path
                return
        except Exception:
            continue

import threading as _threading
_threading.Thread(target=_download_mincho, daemon=True).start()


def get_font(size: int, style: str = "gothic") -> ImageFont.FreeTypeFont:
    path = _FONT_PATHS.get(style) or _JP_FONT
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            if _JP_FONT and path != _JP_FONT:
                try:
                    return ImageFont.truetype(_JP_FONT, size)
                except Exception:
                    pass
    return ImageFont.load_default()


# ════════════════════════════════════════════
#  共通ユーティリティ
# ════════════════════════════════════════════
def _to_base64(img: Image.Image, max_h: int = 600) -> str:
    """プレビュー用：モバイル転送速度を考慮して小さめに"""
    img = img.copy()
    if img.height > max_h:
        r = max_h / img.height
        img = img.resize((int(img.width * r), max_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def _crop45(img: Image.Image) -> Image.Image:
    """4:5にセンタークロップ"""
    w, h = img.size
    if (w / h) > 4 / 5:
        nw = int(h * 4 / 5)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w * 5 / 4)
        img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return img.resize((1080, 1350), Image.LANCZOS)

def _gradient_overlay(img: Image.Image, top_alpha: int, bottom_alpha: int,
                       top_pct: float = 0.4, bottom_pct: float = 0.55) -> Image.Image:
    """上下グラデーションオーバーレイ"""
    w, h = img.size
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d  = ImageDraw.Draw(ov)
    for y in range(h):
        a = 0
        if y < h * top_pct:
            a = int(top_alpha * (1 - y / (h * top_pct)))
        elif y > h * bottom_pct:
            a = int(bottom_alpha * ((y - h * bottom_pct) / (h * (1 - bottom_pct))))
        if a:
            d.line([(0, y), (w, y)], fill=(8, 5, 18, a))
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

def _draw_gold_line(draw: ImageDraw.Draw, cx: int, y: int, width: int = 360):
    draw.line([(cx - width // 2, y), (cx + width // 2, y)], fill="#c9a96e", width=2)


# ════════════════════════════════════════════
#  単品写真エディター
# ════════════════════════════════════════════
def _rotate_no_black(img: Image.Image, angle: float) -> Image.Image:
    if angle == 0:
        return img
    a = math.radians(abs(angle))
    scale = 1.0 + math.sin(a) * 0.7
    w, h = img.size
    bw, bh = int(w * scale), int(h * scale)
    big = img.resize((bw, bh), Image.LANCZOS)
    rotated = big.rotate(angle, resample=Image.BICUBIC, expand=False)
    x, y = (bw - w) // 2, (bh - h) // 2
    return rotated.crop((x, y, x + w, y + h))

def process_image(img: Image.Image, p: dict, for_save: bool = False) -> Image.Image:
    angle = float(p.get("rotate", 0))
    if angle != 0:
        img = _rotate_no_black(img, angle)
    crop = p.get("crop", "4:5")
    if crop != "original":
        target = 4 / 5 if crop == "4:5" else 1.0
        w, h = img.size
        if (w / h) > target:
            nw = int(h * target)
            img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w / target)
            img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    warmth = int(float(p.get("warmth", 0)))
    if warmth:
        r, g, b = img.split()
        r = r.point(lambda x: min(255, max(0, x + warmth)))
        b = b.point(lambda x: min(255, max(0, x - warmth)))
        img = Image.merge("RGB", (r, g, b))
    img = ImageEnhance.Brightness(img).enhance(float(p.get("brightness", 1.0)))
    img = ImageEnhance.Contrast(img).enhance(float(p.get("contrast", 1.0)))
    img = ImageEnhance.Color(img).enhance(float(p.get("saturation", 1.0)))
    img = ImageEnhance.Sharpness(img).enhance(float(p.get("sharpness", 1.0)))
    if for_save:
        if crop == "4:5":   img = img.resize((1080, 1350), Image.LANCZOS)
        elif crop == "1:1": img = img.resize((1080, 1080), Image.LANCZOS)
        else:
            w, h = img.size
            s = 1080 / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    text = p.get("text", "").strip()
    if text:
        img = _draw_text_on_photo(img, p)
    return img

def _draw_text_on_photo(img: Image.Image, p: dict) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    fs = max(10, int(float(p.get("text_size", 72)) * w / 540))
    font = get_font(fs)
    x = int(w * float(p.get("text_x", 50)) / 100)
    y = int(h * float(p.get("text_y", 15)) / 100)
    color  = p.get("text_color", "#FFFFFF")
    shadow = str(p.get("text_shadow", "true")).lower() == "true"
    lines  = p.get("text", "").split("\n")
    lh     = int(fs * 1.35)
    sy     = y - (lh * len(lines)) // 2
    for i, line in enumerate(lines):
        ly = sy + i * lh
        if shadow:
            off = max(2, fs // 20)
            draw.text((x + off, ly + off), line, font=font, fill=(0, 0, 0), anchor="mm")
        draw.text((x, ly), line, font=font, fill=color, anchor="mm")
    return img


# ════════════════════════════════════════════
#  表紙テンプレート
# ════════════════════════════════════════════
def generate_cover(bg: Image.Image, names: list[str],
                    text_color: str = "#ffffff",
                    accent_color: str = "#c9a96e") -> Image.Image:
    img = _crop45(bg)
    img = _gradient_overlay(img, top_alpha=190, bottom_alpha=210,
                             top_pct=0.38, bottom_pct=0.52)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = w // 2

    font_title = get_font(92)
    font_sub   = get_font(44)
    font_name  = get_font(58)
    font_bullet = get_font(28)

    n = len(names)
    title = f"{n}種の飲み比べ"
    ty = int(h * 0.15)

    # 装飾線
    draw.line([(cx - 160, int(h * 0.09)), (cx + 160, int(h * 0.09))], fill=accent_color, width=2)
    # タイトル影
    for ox, oy in [(3, 3), (-3, 3)]:
        draw.text((cx + ox, ty + oy), title, font=font_title, fill=(0, 0, 0), anchor="mm")
    draw.text((cx, ty), title, font=font_title, fill=text_color, anchor="mm")
    draw.line([(cx - 160, int(h * 0.22)), (cx + 160, int(h * 0.22))], fill=accent_color, width=2)

    # 銘柄リスト
    list_top = int(h * 0.67)
    item_h   = int(h * 0.065)
    for i, name in enumerate(names):
        y = list_top + i * item_h
        if i > 0:
            draw.line([(cx - 220, y - item_h // 2), (cx + 220, y - item_h // 2)],
                      fill=accent_color + "60" if len(accent_color) == 7 else accent_color, width=1)
        draw.text((cx - 200, y), "◆", font=font_bullet, fill=accent_color, anchor="mm")
        draw.text((cx + 10, y), name, font=font_name, fill=text_color, anchor="mm")

    return img


# ════════════════════════════════════════════
#  テイスティングカード ＋ レーダーチャート
# ════════════════════════════════════════════
TASTE_LABELS = ["甘味", "酸味", "旨味", "苦味", "渋み", "香り"]

def make_radar_chart(scores: list[float], for_save: bool = False) -> Image.Image:
    # キャッシュチェック（スコアが同じなら再生成しない）
    cache_key = (tuple(scores), for_save)
    if _chart_cache["key"] == cache_key and _chart_cache["img"] is not None:
        return _chart_cache["img"].copy()

    n = len(TASTE_LABELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals   = scores + [scores[0]]
    angs   = angles + [angles[0]]

    dpi = 150 if for_save else 90  # プレビューは低DPIで高速化
    bg_color = (0.031, 0.020, 0.071)  # 濃紺 — チャート専用背景色
    fig = plt.figure(figsize=(5, 5), facecolor=bg_color)
    ax  = fig.add_subplot(111, polar=True, facecolor=bg_color)

    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2","4","6","8","10"], fontsize=7, color="white", alpha=0.45)
    ax.set_xticks(angles)
    ax.set_xticklabels(TASTE_LABELS, size=20, color="white", fontweight="bold")
    ax.tick_params(pad=12)

    # グリッドを金色に — ジョジョのスタンドカード風
    ax.grid(color="#c9a96e", alpha=0.30, linewidth=1.0)
    ax.spines["polar"].set_color("#c9a96e")
    ax.spines["polar"].set_alpha(0.85)
    for gl in ax.yaxis.get_gridlines():
        gl.set_color("#c9a96e"); gl.set_alpha(0.25)

    ax.fill(angs, vals, color="#c9a96e", alpha=0.50)
    ax.plot(angs, vals, color="#c9a96e", linewidth=3.0, zorder=5)
    ax.scatter(angles, scores, color="#ffd966", s=65, zorder=6, edgecolors="white", linewidths=1.0)

    for ang, sc in zip(angles, scores):
        ax.text(ang, sc + 0.9, str(int(sc)), ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")

    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=False, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    result = Image.open(buf).convert("RGBA")

    _chart_cache["key"] = cache_key
    _chart_cache["img"] = result.copy()
    return result

def generate_tasting_card(bg: Image.Image, sake_name: str, scores: list[float],
                           chart_x: int = 50, chart_y: int = 55, chart_size: int = 75,
                           name_x: int = 50, name_y: int = 10, name_size: int = 82,
                           name_color: str = "#ffffff", name_shadow: bool = True,
                           for_save: bool = False, font_style: str = "mincho") -> Image.Image:
    img = _crop45(bg)
    w, h = img.size

    # 下半分を暗くするオーバーレイ（チャート位置に合わせて調整）
    overlay_start = max(0.1, chart_y / 100 - 0.2)
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d  = ImageDraw.Draw(ov)
    for y in range(h):
        if y > h * overlay_start:
            a = int(215 * min(1.0, (y - h * overlay_start) / (h * 0.45)))
            d.line([(0, y), (w, y)], fill=(8, 5, 18, a))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    draw = ImageDraw.Draw(img)

    img_rgba = img.convert("RGBA")

    # ── 銘柄名（背景バー付き）──
    nx = int(w * name_x / 100)
    ny = int(h * name_y / 100)
    font_title = get_font(name_size, font_style)
    # 背景バー：テキストが何色の写真の上でも読める
    bar_h = name_size + 28
    bar = Image.new("RGBA", (w, bar_h), (0, 0, 0, 170))
    img_rgba.paste(bar, (0, ny - bar_h // 2), bar)
    draw = ImageDraw.Draw(img_rgba)
    # デコライン（上下2本でジョジョ的格式感）
    line_y_top = ny - bar_h // 2 + 2
    line_y_bot = ny + bar_h // 2 - 2
    draw.line([(0, line_y_top), (w, line_y_top)], fill="#c9a96e", width=2)
    draw.line([(0, line_y_bot), (w, line_y_bot)], fill="#c9a96e", width=2)
    # テキスト
    draw.text((nx, ny), sake_name, font=font_title, fill=name_color, anchor="mm")

    # ── レーダーチャート ──
    radar    = make_radar_chart(scores, for_save=for_save)
    chart_w  = int(w * chart_size / 100)
    # アスペクト比を維持してリサイズ（ラベルが切れないよう）
    rw, rh   = radar.size
    scale    = chart_w / max(rw, rh)
    radar    = radar.resize((int(rw * scale), int(rh * scale)), Image.LANCZOS)
    rw2, rh2 = radar.size
    # chart_x/chart_y はチャート中心の位置（%）
    cx_r = int(w * chart_x / 100) - rw2 // 2
    cy_r = int(h * chart_y / 100) - rh2 // 2

    # 金色の枠線付きパネルをチャートに合成
    pad   = 6
    pw, ph = rw2 + pad * 2, rh2 + pad * 2
    border_layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border_layer)
    bd.rectangle([(0, 0), (pw - 1, ph - 1)], outline="#c9a96e", width=3)
    bg_panel = Image.new("RGBA", (pw, ph), (8, 5, 18, 255))
    bg_panel.paste(radar.convert("RGBA"), (pad, pad))
    bg_panel = Image.alpha_composite(bg_panel, border_layer)
    img_rgba.paste(bg_panel, (cx_r - pad, cy_r - pad), bg_panel)
    img = img_rgba.convert("RGB")

    return img


# ════════════════════════════════════════════
#  ジョジョ風スタンドカード（横長）
# ════════════════════════════════════════════
_GRADE = {10:"A",9:"A",8:"B",7:"B",6:"C",5:"C",4:"D",3:"D",2:"E",1:"E"}

def _score_grade(s: float) -> str:
    return _GRADE.get(max(1, min(10, int(round(s)))), "C")

def generate_stand_card(
    sticker: "Image.Image | None",
    sake_name: str,
    scores: list[float],
    font_style: str = "mincho",
    bg_color: str = "#0a0710",
    for_save: bool = False,
) -> Image.Image:
    """ジョジョ風スタンドカード — 横長 1080×810"""
    W, H = 1080, 810
    GOLD = "#c9a96e"

    # ── 背景（左上が少し明るい微グラデ）──
    r, g, b = int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)
    card = Image.new("RGBA", (W, H), (r, g, b, 255))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ovd = ImageDraw.Draw(ov)
    for px in range(W):
        a = int(35 * (1 - px / W))
        ovd.line([(px, 0), (px, H)], fill=(255, 255, 255, a))
    card = Image.alpha_composite(card, ov)
    draw = ImageDraw.Draw(card)

    # ── 二重枠線 ──
    draw.rectangle([(12, 12), (W - 13, H - 13)], outline=GOLD, width=3)
    draw.rectangle([(20, 20), (W - 21, H - 21)], outline=GOLD, width=1)

    # ── タイトル帯（y=27〜120）──
    draw.rectangle([(25, 27), (W - 26, 118)], fill=(0, 0, 0, 210))
    draw.line([(25, 27),  (W - 26, 27)],  fill=GOLD, width=2)
    draw.line([(25, 118), (W - 26, 118)], fill=GOLD, width=2)

    sub_font  = get_font(22, "gothic")
    name_font = get_font(68 if for_save else 56, font_style)
    draw.text((42, 48), "NIHONSHU STAND", font=sub_font, fill=GOLD)
    draw.text((W // 2, 73), sake_name, font=name_font, fill="#ffffff", anchor="mm")

    # ── コンテンツエリア（y=130〜H-25）──
    C_TOP = 130
    C_H   = H - C_TOP - 25
    L_W   = 450    # 左ゾーン幅（ステッカー）
    DIV_X = L_W + 30
    R_X   = DIV_X + 12   # 右ゾーン開始X
    R_W   = W - R_X - 25  # 右ゾーン幅

    # 中央分割線
    draw.line([(DIV_X + 5, C_TOP + 8), (DIV_X + 5, H - 28)], fill=GOLD, width=1)

    # ── 左側：ステッカー ──
    if sticker:
        st = sticker.convert("RGBA")
        sw, sh = st.size
        max_w, max_h = L_W - 20, C_H - 10
        scale = min(max_w / sw, max_h / sh, 1.0)
        nw, nh = int(sw * scale), int(sh * scale)
        st = st.resize((nw, nh), Image.LANCZOS)
        sx = 25 + (max_w - nw) // 2 + 10
        sy = C_TOP + (C_H - nh) // 2
        card.paste(st, (sx, sy), st)
    else:
        # ステッカーなし：装飾的な「？」プレースホルダ
        ph_font = get_font(120, "gothic")
        draw.text((25 + L_W // 2, C_TOP + C_H // 2), "🍶", font=ph_font,
                  fill=(100, 80, 30), anchor="mm")

    # ── 右側上部：チャート ──
    chart_area_h = int(C_H * 0.60)
    chart_img = make_radar_chart(scores, for_save=for_save)
    cw, ch = chart_img.size
    scale = min((R_W - 10) / cw, chart_area_h / ch, 1.0)
    ncw, nch = int(cw * scale), int(ch * scale)
    chart_img = chart_img.resize((ncw, nch), Image.LANCZOS)
    chart_img_rgba = chart_img.convert("RGBA")
    cx = R_X + (R_W - ncw) // 2
    cy = C_TOP + (chart_area_h - nch) // 2
    card.paste(chart_img_rgba, (cx, cy), chart_img_rgba)

    # ── 右側下部：スコアバー（JoJo風 A〜E）──
    bar_top   = C_TOP + chart_area_h + 8
    bar_total = H - 28 - bar_top
    n_rows    = 3
    row_h     = bar_total // n_rows
    lbl_font  = get_font(20, font_style)
    grd_font  = get_font(22, "gothic")

    for i, (label, score) in enumerate(zip(TASTE_LABELS, scores)):
        col = i % 2
        row = i // 2
        bx  = R_X + col * (R_W // 2)
        by  = bar_top + row * row_h + (row_h - 28) // 2
        bar_w = R_W // 2 - 8
        grade = _score_grade(score)

        draw.text((bx, by + 14), label, font=lbl_font, fill=GOLD, anchor="lm")
        bar_left = bx + 52
        bar_right = bx + bar_w - 30
        bar_inner = bar_right - bar_left
        draw.rectangle([(bar_left, by + 4), (bar_right, by + 22)],
                       fill=(40, 30, 10, 200))
        fill_w = int(bar_inner * score / 10)
        if fill_w > 0:
            draw.rectangle([(bar_left, by + 4), (bar_left + fill_w, by + 22)],
                           fill=GOLD)
        draw.text((bx + bar_w - 22, by + 14), grade, font=grd_font,
                  fill="#ffffff", anchor="mm")

    return card.convert("RGB")


# ════════════════════════════════════════════
#  Claude Vision 自動分析（単品エディター用）
# ════════════════════════════════════════════
def claude_analyze(img: Image.Image, text: str) -> dict:
    client = anthropic.Anthropic()
    preview = img.copy()
    preview.thumbnail((1000, 1250), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = f"""あなたはInstagramの写真編集とビジュアルデザインの専門家です。
この写真を見て、インスタ映えする投稿画像にするための最適な編集パラメータをJSONで返してください。

画像に重ねるテキスト: 「{text if text else "（テキストなし）"}」

返すJSONのキーと範囲：
{{
  "rotate": -15〜15（傾き補正）,
  "brightness": 0.5〜2.0,
  "contrast": 0.5〜2.0,
  "saturation": 0.5〜2.5,
  "warmth": -50〜50,
  "sharpness": 0〜3.0,
  "crop": "4:5" または "1:1" または "original",
  "text_x": 0〜100,
  "text_y": 0〜100,
  "text_color": "#FFFFFF" など,
  "text_size": 20〜160,
  "text_shadow": "true" または "false",
  "reason": "判断の根拠を日本語で3〜4文"
}}
JSONのみ返してください。"""

    res = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=600,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": prompt}
        ]}]
    )
    raw = re.sub(r"```(?:json)?", "", res.content[0].text).strip().rstrip("`").strip()
    return json.loads(raw)

def pixel_analyze(img: Image.Image, text: str) -> dict:
    w, h = img.size
    if (w / h) > 4 / 5:
        nw = int(h * 4 / 5)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    w, h = img.size
    gray   = img.convert("L")
    pixels = list(gray.getdata())[::max(1, len(list(gray.getdata())) // 5000)]
    avg_b  = sum(pixels) / len(pixels)
    brightness = round(min(1.8, max(0.7, 130 / max(avg_b, 1))), 2)
    try:   std = statistics.stdev(pixels)
    except: std = 40
    contrast = round(min(1.6, max(1.0, 70 / max(std, 1))), 2)
    r_avg = sum(list(img.split()[0].getdata())[::10]) / max(1, len(list(img.split()[0].getdata())[::10]))
    b_avg = sum(list(img.split()[2].getdata())[::10]) / max(1, len(list(img.split()[2].getdata())[::10]))
    warmth = 0 if (r_avg - b_avg) > 25 else 8
    candidates = [(50,12),(50,88),(50,20),(50,80),(25,15),(75,15),(25,85),(75,85)]
    best, min_var = (50, 15), float("inf")
    zw, zh = w // 4, h // 8
    for cx_p, cy_p in candidates:
        cx, cy = int(w*cx_p/100), int(h*cy_p/100)
        region = gray.crop((max(0,cx-zw//2), max(0,cy-zh//2),
                             min(w,cx+zw//2), min(h,cy+zh//2)))
        rp = list(region.getdata())
        try:
            var = statistics.variance(rp)
            if var < min_var: min_var, best = var, (cx_p, cy_p)
        except: pass
    tx, ty = best
    sx, sy = int(w*tx/100), int(h*ty/100)
    zw = min(60, w//4)
    sp = list(gray.crop((max(0,sx-zw),max(0,sy-zw//2),min(w,sx+zw),min(h,sy+zw//2))).getdata())
    text_color = "#000000" if sum(sp)/max(len(sp),1) > 140 else "#FFFFFF"
    nc = len(text.replace("\n","").replace(" ",""))
    text_size = 96 if nc<=8 else 76 if nc<=15 else 60 if nc<=30 else 46
    return {"rotate":0,"crop":"4:5","brightness":brightness,"contrast":contrast,
            "saturation":1.45,"warmth":warmth,"sharpness":1.5,
            "text_x":tx,"text_y":ty,"text_color":text_color,"text_size":text_size,"text_shadow":"true"}


# ════════════════════════════════════════════
#  Flask ルート
# ════════════════════════════════════════════
def _get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80)); ip = s.getsockname()[0]; s.close(); return ip
    except: return "localhost"

@app.route("/")
def index(): return render_template("index.html")

# ── 単品編集 ──
@app.route("/upload", methods=["POST"])
def upload():
    global _photo_bytes
    f = request.files.get("image")
    if not f: return jsonify({"error":"no file"}), 400
    img = ImageOps.exif_transpose(Image.open(f.stream).convert("RGB"))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=95)
    _photo_bytes = buf.getvalue()
    return jsonify({"preview": _to_base64(img)})

@app.route("/preview", methods=["POST"])
def preview():
    if not _photo_bytes: return jsonify({"error":"no image"}), 400
    img = Image.open(io.BytesIO(_photo_bytes)).convert("RGB")
    return jsonify({"preview": _to_base64(process_image(img, request.get_json() or {}))})

@app.route("/analyze", methods=["POST"])
def analyze():
    if not _photo_bytes: return jsonify({"error":"no image"}), 400
    data = request.get_json() or {}
    img  = ImageOps.exif_transpose(Image.open(io.BytesIO(_photo_bytes)).convert("RGB"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            p = claude_analyze(img, data.get("text",""))
            p["source"] = "claude"; return jsonify(p)
        except Exception as e:
            print(f"[Claude] {e}")
    p = pixel_analyze(img, data.get("text",""))
    p["source"] = "pixel"; return jsonify(p)

@app.route("/download", methods=["POST"])
def download():
    if not _photo_bytes: return jsonify({"error":"no image"}), 400
    img = Image.open(io.BytesIO(_photo_bytes)).convert("RGB")
    img = process_image(img, request.get_json() or {}, for_save=True)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=95); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="insta_photo.jpg")

# ── 表紙テンプレート ──
@app.route("/upload_cover", methods=["POST"])
def upload_cover():
    global _cover_bytes
    f = request.files.get("image")
    if not f: return jsonify({"error":"no file"}), 400
    img = ImageOps.exif_transpose(Image.open(f.stream).convert("RGB"))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=95)
    _cover_bytes = buf.getvalue()
    return jsonify({"preview": _to_base64(img)})

def _cover_params(data: dict):
    return dict(
        names=[n.strip() for n in data.get("names", []) if n.strip()],
        text_color=data.get("text_color", "#ffffff"),
        accent_color=data.get("accent_color", "#c9a96e"),
    )

@app.route("/preview_cover", methods=["POST"])
def preview_cover():
    if not _cover_bytes: return jsonify({"error":"no image"}), 400
    p = _cover_params(request.get_json() or {})
    if not p["names"]: return jsonify({"error":"no names"}), 400
    img = Image.open(io.BytesIO(_cover_bytes)).convert("RGB")
    return jsonify({"preview": _to_base64(generate_cover(img, p["names"], p["text_color"], p["accent_color"]))})

@app.route("/download_cover", methods=["POST"])
def download_cover():
    if not _cover_bytes: return jsonify({"error":"no image"}), 400
    p   = _cover_params(request.get_json() or {})
    img = Image.open(io.BytesIO(_cover_bytes)).convert("RGB")
    result = generate_cover(img, p["names"], p["text_color"], p["accent_color"])
    buf = io.BytesIO(); result.save(buf, "JPEG", quality=95); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="cover.jpg")

# ── テイスティングカード ──
@app.route("/upload_tasting", methods=["POST"])
def upload_tasting():
    global _tasting_bytes
    f = request.files.get("image")
    if not f: return jsonify({"error":"no file"}), 400
    img = ImageOps.exif_transpose(Image.open(f.stream).convert("RGB"))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=95)
    _tasting_bytes = buf.getvalue()
    return jsonify({"preview": _to_base64(img)})

def _tasting_params(data: dict):
    name   = data.get("sake_name", "銘柄名")
    scores = [float(data.get(k, 5)) for k in ["甘味","酸味","旨味","苦味","渋み","香り"]]
    return dict(
        sake_name=name, scores=scores,
        chart_x=int(data.get("chart_x", 50)),
        chart_y=int(data.get("chart_y", 55)),
        chart_size=int(data.get("chart_size", 75)),
        name_x=int(data.get("name_x", 50)),
        name_y=int(data.get("name_y", 10)),
        name_size=int(data.get("name_size", 82)),
        name_color=data.get("name_color", "#ffffff"),
        name_shadow=str(data.get("name_shadow", "true")).lower() == "true",
        font_style=data.get("font_style", "mincho"),
    )

@app.route("/preview_tasting", methods=["POST"])
def preview_tasting():
    if not _tasting_bytes: return jsonify({"error":"no image"}), 400
    p   = _tasting_params(request.get_json() or {})
    img = Image.open(io.BytesIO(_tasting_bytes)).convert("RGB")
    result = generate_tasting_card(img, p["sake_name"], p["scores"],
                                    p["chart_x"], p["chart_y"], p["chart_size"],
                                    p["name_x"], p["name_y"], p["name_size"],
                                    p["name_color"], p["name_shadow"],
                                    font_style=p["font_style"])
    return jsonify({"preview": _to_base64(result)})

@app.route("/download_tasting", methods=["POST"])
def download_tasting():
    if not _tasting_bytes: return jsonify({"error":"no image"}), 400
    p   = _tasting_params(request.get_json() or {})
    img = Image.open(io.BytesIO(_tasting_bytes)).convert("RGB")
    # 保存時は高品質チャートを使う
    _chart_cache["key"] = None
    result = generate_tasting_card(img, p["sake_name"], p["scores"],
                                    p["chart_x"], p["chart_y"], p["chart_size"],
                                    p["name_x"], p["name_y"], p["name_size"],
                                    p["name_color"], p["name_shadow"],
                                    for_save=True, font_style=p["font_style"])
    _chart_cache["key"] = None  # キャッシュをリセットして次回プレビューに戻す
    buf = io.BytesIO(); result.save(buf, "JPEG", quality=95); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="tasting.jpg")


# ── スタンドカード ──
@app.route("/upload_sticker", methods=["POST"])
def upload_sticker():
    global _sticker_bytes
    f = request.files.get("image")
    if not f: return jsonify({"error": "no file"}), 400
    img = Image.open(f.stream).convert("RGBA")
    buf = io.BytesIO(); img.save(buf, "PNG")
    _sticker_bytes = buf.getvalue()
    # チェッカーボード背景でプレビュー
    checker = Image.new("RGB", img.size, (180, 180, 180))
    checker.paste(img, mask=img.split()[3])
    return jsonify({"preview": _to_base64(checker)})

def _stand_params(data: dict):
    scores = [float(data.get(k, 5)) for k in ["甘味","酸味","旨味","苦味","渋み","香り"]]
    return dict(
        sake_name=data.get("sake_name", "銘柄名"),
        scores=scores,
        font_style=data.get("font_style", "mincho"),
        bg_color=data.get("bg_color", "#0a0710"),
    )

@app.route("/preview_stand", methods=["POST"])
def preview_stand():
    p = _stand_params(request.get_json() or {})
    sticker = Image.open(io.BytesIO(_sticker_bytes)).convert("RGBA") if _sticker_bytes else None
    card = generate_stand_card(sticker, p["sake_name"], p["scores"],
                               p["font_style"], p["bg_color"])
    return jsonify({"preview": _to_base64(card, max_h=500)})

@app.route("/download_stand", methods=["POST"])
def download_stand():
    p = _stand_params(request.get_json() or {})
    sticker = Image.open(io.BytesIO(_sticker_bytes)).convert("RGBA") if _sticker_bytes else None
    _chart_cache["key"] = None
    card = generate_stand_card(sticker, p["sake_name"], p["scores"],
                               p["font_style"], p["bg_color"], for_save=True)
    _chart_cache["key"] = None
    buf = io.BytesIO(); card.save(buf, "JPEG", quality=95); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="stand_card.jpg")


@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    """Claude APIでInstagramキャプションを生成"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "APIキーが設定されていません"}), 400
    data = request.get_json() or {}
    mode = data.get("mode", "tasting")  # "tasting" or "cover"

    if mode == "cover":
        names = data.get("names", [])
        prompt = f"""あなたはInstagramで日本酒を発信している26歳の男性です。
アカウント名：よう🍶（@nihonshu_you）
コンセプト：難しくない、堅苦しくない日本酒を同世代に広める

今日の投稿：{len(names)}種の飲み比べ
銘柄：{' / '.join(names)}

この投稿のInstagramキャプションを書いてください。

文体の条件：
- 男性的または中性的な口調（「〜だった」「〜だな」「〜してみた」など）
- 柔らかすぎず、でも堅くもない自然な話し言葉
- 「〜ですね！」「〜でした♪」のような女性的・過度に丁寧な表現は避ける

内容の条件：
- 共感を呼ぶ書き出し（体験談・驚き・問いかけなど）
- 飲み比べの感想を1〜2文（具体的かつ親しみやすく）
- 締めの一言
- 改行を使って読みやすく
- 絵文字は各段落に1個程度、自然に使う（合計4〜6個）
- ハッシュタグは最後にまとめて10個程度
- 全体で200〜280文字程度

キャプション本文とハッシュタグを分けて返してください。"""

    else:
        name   = data.get("sake_name", "")
        scores = {k: data.get(k, 3) for k in ["甘味","酸味","旨味","苦味","渋み","香り"]}
        score_text = "、".join([f"{k}{v}" for k, v in scores.items()])
        prompt = f"""あなたはInstagramで日本酒を発信している26歳の男性です。
アカウント名：よう🍶（@nihonshu_you）
コンセプト：難しくない、堅苦しくない日本酒を同世代に広める

今日の銘柄：{name}
テイスティングスコア（5点満点）：{score_text}

この銘柄のInstagramキャプションを書いてください。

文体の条件：
- 男性的または中性的な口調（「〜だった」「〜だな」「〜してみた」など）
- 柔らかすぎず、でも堅くもない自然な話し言葉
- 「〜ですね！」「〜でした♪」のような女性的・過度に丁寧な表現は避ける

内容の条件：
- 共感を呼ぶ書き出し（体験談・驚き・問いかけなど）
- スコアをもとにした飲んだ感想を2〜3文（専門用語NG、共感できる言葉で）
- どんな場面・料理に合うか一言
- 締めの一言
- 改行を使って読みやすく
- 絵文字は各段落に1個程度、自然に使う（合計4〜6個）
- ハッシュタグは最後にまとめて10個程度
- 全体で250〜320文字程度

キャプション本文とハッシュタグを分けて返してください。"""

    try:
        client = anthropic.Anthropic()
        res = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"caption": res.content[0].text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    ip = _get_lan_ip()
    print(f"\n🍶 起動中...  PC: http://localhost:5000  iPhone: http://{ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
