import io
import os
import re
import math
import base64
import secrets
import shutil
import tempfile
import urllib.request
import threading
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_fontja  # noqa: F401

from flask import Flask, render_template, request, jsonify, send_file, session
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# ── セッションごとの一時ストレージ ──
_SESSION_DIR = os.path.join(tempfile.gettempdir(), "standcard_sessions")
os.makedirs(_SESSION_DIR, exist_ok=True)

def _user_dir() -> str:
    if "uid" not in session:
        session["uid"] = secrets.token_hex(16)
    d = os.path.join(_SESSION_DIR, session["uid"])
    os.makedirs(d, exist_ok=True)
    return d

def _get_bytes(name: str) -> bytes | None:
    path = os.path.join(_user_dir(), name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def _set_bytes(name: str, data: bytes) -> None:
    with open(os.path.join(_user_dir(), name), "wb") as f:
        f.write(data)

def _cleanup_old_sessions(max_age_hours: int = 3) -> None:
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    try:
        for name in os.listdir(_SESSION_DIR):
            d = os.path.join(_SESSION_DIR, name)
            if os.path.isdir(d) and datetime.fromtimestamp(os.path.getmtime(d)) < cutoff:
                shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass

# ── フォント管理 ──
def _find_gothic() -> str | None:
    for p in [r"C:\Windows\Fonts\YuGothB.ttc",
              r"C:\Windows\Fonts\meiryo.ttc",
              r"C:\Windows\Fonts\msgothic.ttc"]:
        if os.path.exists(p):
            return p
    try:
        import matplotlib_fontja as mf
        fp = os.path.join(os.path.dirname(mf.__file__), "fonts", "ipaexg.ttf")
        if os.path.exists(fp):
            return fp
    except Exception:
        pass
    return None

_FONT_PATHS: dict[str, str | None] = {"gothic": _find_gothic(), "mincho": None}

def _download_mincho() -> None:
    cache_dir = os.path.join(tempfile.gettempdir(), "standcard_fonts")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "ipaexm.ttf")
    if os.path.exists(path) and os.path.getsize(path) > 500_000:
        _FONT_PATHS["mincho"] = path
        return
    try:
        url = "https://raw.githubusercontent.com/ootaharuki99/IPAexfont/master/ipaexm.ttf"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) > 500_000:
            with open(path, "wb") as f:
                f.write(data)
            _FONT_PATHS["mincho"] = path
    except Exception:
        pass

threading.Thread(target=_download_mincho, daemon=True).start()

def get_font(size: int, style: str = "gothic") -> ImageFont.FreeTypeFont:
    path = _FONT_PATHS.get(style) or _FONT_PATHS["gothic"]
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

# ── レーダーチャート ──
TASTE_LABELS = ["甘味", "酸味", "旨味", "苦味", "渋み", "香り"]
_chart_cache: dict = {"key": None, "img": None}

def make_radar_chart(scores: list[float], for_save: bool = False) -> Image.Image:
    cache_key = (tuple(scores), for_save)
    if _chart_cache["key"] == cache_key and _chart_cache["img"] is not None:
        return _chart_cache["img"].copy()

    n = len(TASTE_LABELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals = scores + [scores[0]]
    angs = angles + [angles[0]]

    dpi = 150 if for_save else 90
    bg_color = (0.031, 0.020, 0.071)
    fig = plt.figure(figsize=(5, 5), facecolor=bg_color)
    ax = fig.add_subplot(111, polar=True, facecolor=bg_color)

    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color="white", alpha=0.45)
    ax.set_xticks(angles)
    ax.set_xticklabels(TASTE_LABELS, size=20, color="white", fontweight="bold")
    ax.tick_params(pad=12)

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

# ── ステッカーエッジ処理 ──
def _clean_sticker_alpha(img: Image.Image, erode_px: int = 2) -> Image.Image:
    if erode_px == 0:
        return img.convert("RGBA")
    img = img.convert("RGBA")
    r, g, b, a = img.split()
    threshold = min(erode_px * 15, 230)
    a = a.point(lambda x: 0 if x < threshold else x)
    erosion_rounds = max(0, (erode_px - 3) // 3)
    for _ in range(erosion_rounds):
        a = a.filter(ImageFilter.MinFilter(3))
    a = a.filter(ImageFilter.GaussianBlur(radius=min(erode_px * 0.3, 2.0)))
    img.putalpha(a)
    return img

# ── スタンドカード生成 ──
_GRADE = {10: "A", 9: "A", 8: "B", 7: "B", 6: "C", 5: "C", 4: "D", 3: "D", 2: "E", 1: "E"}

def _score_grade(s: float) -> str:
    return _GRADE.get(max(1, min(10, int(round(s)))), "C")

def generate_stand_card(
    sticker: "Image.Image | None",
    name: str,
    scores: list[float],
    font_style: str = "mincho",
    bg_color: str = "#0a0710",
    for_save: bool = False,
    sticker_scale: float = 1.0,
    sticker_angle: float = 0,
    sticker_erode: int = 2,
    bg_pattern: str = "none",
) -> Image.Image:
    W, H = 1080, 810
    GOLD = "#c9a96e"

    r, g, b = int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)
    card = Image.new("RGBA", (W, H), (r, g, b, 255))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ovd = ImageDraw.Draw(ov)
    for px in range(W):
        a = int(35 * (1 - px / W))
        ovd.line([(px, 0), (px, H)], fill=(255, 255, 255, a))
    card = Image.alpha_composite(card, ov)
    draw = ImageDraw.Draw(card)

    draw.rectangle([(12, 12), (W - 13, H - 13)], outline=GOLD, width=3)
    draw.rectangle([(20, 20), (W - 21, H - 21)], outline=GOLD, width=1)

    draw.rectangle([(25, 27), (W - 26, 118)], fill=(0, 0, 0, 210))
    draw.line([(25, 27),  (W - 26, 27)],  fill=GOLD, width=2)
    draw.line([(25, 118), (W - 26, 118)], fill=GOLD, width=2)

    sub_font  = get_font(22, "gothic")
    name_font = get_font(68 if for_save else 56, font_style)
    draw.text((42, 48), "NIHONSHU STAND", font=sub_font, fill=GOLD)
    draw.text((W // 2, 73), name, font=name_font, fill="#ffffff", anchor="mm")

    C_TOP = 130
    C_H   = H - C_TOP - 25
    L_W   = 450
    DIV_X = L_W + 30
    R_X   = DIV_X + 12
    R_W   = W - R_X - 25

    draw.line([(DIV_X + 5, C_TOP + 8), (DIV_X + 5, H - 28)], fill=GOLD, width=1)

    ZL, ZT, ZR, ZB = 25, C_TOP, DIV_X, H - 25
    ZCX, ZCY = (ZL + ZR) // 2, (ZT + ZB) // 2

    if bg_pattern in ("dots", "combined"):
        spacing = 16
        for ri, dy in enumerate(range(ZT + 8, ZB - 4, spacing)):
            offset = spacing // 2 if ri % 2 else 0
            for dx in range(ZL + 5 + offset, ZR - 4, spacing):
                draw.ellipse([(dx, dy), (dx + 1, dy + 1)], fill=(150, 120, 50, 45))

    if bg_pattern == "hatch":
        zone_w, zone_h = ZR - ZL, ZB - ZT
        for offset in range(-zone_h, zone_w + zone_h, 18):
            x1, y1 = ZL, ZT + offset
            x2, y2 = ZR, ZT + offset + zone_w
            if y1 < ZT: x1 += ZT - y1; y1 = ZT
            if y2 > ZB: x2 -= y2 - ZB; y2 = ZB
            if x1 < ZR and x2 > ZL and y1 <= y2:
                draw.line([(x1, y1), (x2, y2)], fill=(100, 80, 30, 35), width=1)

    if bg_pattern in ("glow", "combined"):
        gw, gh = (ZR - ZL) // 2 + 20, (ZB - ZT) // 2 + 20
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse(
            [(ZCX - gw, ZCY - gh), (ZCX + gw, ZCY + gh)],
            fill=(140, 110, 45, 100)
        )
        glow = glow.filter(ImageFilter.GaussianBlur(radius=55))
        card = Image.alpha_composite(card, glow)
        draw = ImageDraw.Draw(card)

    if sticker:
        st = sticker.convert("RGBA")
        alpha_bbox = st.split()[3].getbbox()
        if alpha_bbox:
            st = st.crop(alpha_bbox)
        if sticker_erode > 0:
            st = _clean_sticker_alpha(st, erode_px=sticker_erode)
        if sticker_angle != 0:
            st = st.rotate(-sticker_angle, expand=True, resample=Image.BICUBIC)
        sw, sh = st.size
        max_w, max_h = L_W - 20, C_H - 10
        base_scale = min(max_w / sw, max_h / sh)
        scale = base_scale * sticker_scale
        nw, nh = int(sw * scale), int(sh * scale)
        st = st.resize((nw, nh), Image.LANCZOS)
        sx = 25 + (max_w - nw) // 2 + 10
        sy = C_TOP + (C_H - nh) // 2
        card.paste(st, (sx, sy), st)
    else:
        ph_font = get_font(120, "gothic")
        draw.text((25 + L_W // 2, C_TOP + C_H // 2), "🍶", font=ph_font,
                  fill=(100, 80, 30), anchor="mm")

    chart_area_h = int(C_H * 0.60)
    chart_img = make_radar_chart(scores, for_save=for_save)
    cw, ch = chart_img.size
    scale = min((R_W - 10) / cw, chart_area_h / ch, 1.0)
    ncw, nch = int(cw * scale), int(ch * scale)
    chart_img = chart_img.resize((ncw, nch), Image.LANCZOS)
    cx = R_X + (R_W - ncw) // 2
    cy = C_TOP + (chart_area_h - nch) // 2
    card.paste(chart_img.convert("RGBA"), (cx, cy), chart_img.convert("RGBA"))

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
        bar_left  = bx + 52
        bar_right = bx + bar_w - 30
        bar_inner = bar_right - bar_left
        draw.rectangle([(bar_left, by + 4), (bar_right, by + 22)], fill=(40, 30, 10, 200))
        fill_w = int(bar_inner * score / 10)
        if fill_w > 0:
            draw.rectangle([(bar_left, by + 4), (bar_left + fill_w, by + 22)], fill=GOLD)
        draw.text((bx + bar_w - 22, by + 14), grade, font=grd_font, fill="#ffffff", anchor="mm")

    return card.convert("RGB")

def _to_base64(img: Image.Image, max_h: int = 500) -> str:
    img = img.copy()
    if img.height > max_h:
        r = max_h / img.height
        img = img.resize((int(img.width * r), max_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def _stand_params(data: dict) -> dict:
    scores = [float(data.get(k, 5)) for k in ["甘味", "酸味", "旨味", "苦味", "渋み", "香り"]]
    return dict(
        name=data.get("name", "銘柄名"),
        scores=scores,
        font_style=data.get("font_style", "mincho"),
        bg_color=data.get("bg_color", "#0a0710"),
        sticker_scale=float(data.get("sticker_scale", 100)) / 100.0,
        sticker_angle=float(data.get("sticker_angle", 0)),
        sticker_erode=int(data.get("sticker_erode", 2)),
        bg_pattern=data.get("bg_pattern", "none"),
    )

# ── ルート ──
@app.route("/")
def index():
    return render_template("stand.html")

@app.route("/upload_sticker", methods=["POST"])
def upload_sticker():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no file"}), 400
    _cleanup_old_sessions()
    img = Image.open(f.stream).convert("RGBA")
    buf = io.BytesIO(); img.save(buf, "PNG")
    _set_bytes("sticker", buf.getvalue())
    checker = Image.new("RGB", img.size, (180, 180, 180))
    checker.paste(img, mask=img.split()[3])
    return jsonify({"preview": _to_base64(checker)})

@app.route("/preview_stand", methods=["POST"])
def preview_stand():
    p = _stand_params(request.get_json() or {})
    sb = _get_bytes("sticker")
    sticker = Image.open(io.BytesIO(sb)).convert("RGBA") if sb else None
    card = generate_stand_card(sticker, p["name"], p["scores"],
                               p["font_style"], p["bg_color"],
                               sticker_scale=p["sticker_scale"],
                               sticker_angle=p["sticker_angle"],
                               sticker_erode=p["sticker_erode"],
                               bg_pattern=p["bg_pattern"])
    return jsonify({"preview": _to_base64(card)})

@app.route("/download_stand", methods=["POST"])
def download_stand():
    p = _stand_params(request.get_json() or {})
    sb = _get_bytes("sticker")
    sticker = Image.open(io.BytesIO(sb)).convert("RGBA") if sb else None
    _chart_cache["key"] = None
    card = generate_stand_card(sticker, p["name"], p["scores"],
                               p["font_style"], p["bg_color"], for_save=True,
                               sticker_scale=p["sticker_scale"],
                               sticker_angle=p["sticker_angle"],
                               sticker_erode=p["sticker_erode"],
                               bg_pattern=p["bg_pattern"])
    _chart_cache["key"] = None
    buf = io.BytesIO(); card.save(buf, "JPEG", quality=95); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", as_attachment=True, download_name="stand_card.jpg")

if __name__ == "__main__":
    import socket
    port = int(os.environ.get("PORT", 5001))
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
    except Exception:
        ip = "localhost"
    print(f"\n🍶 スタンドカード起動中...  http://localhost:{port}  /  http://{ip}:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
