import io
import os
import re
import json
import math
import base64
import socket
import statistics
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
import anthropic

app = Flask(__name__)
_stored_image: bytes | None = None


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


# ────────────────────────────────────────────
#  回転（黒背景なし）
#  画像を少し拡大してから回転→元サイズにクロップ
# ────────────────────────────────────────────
def _rotate_no_black(img: Image.Image, angle: float) -> Image.Image:
    if angle == 0:
        return img
    a_rad = math.radians(abs(angle))
    # 拡大率: 回転後に黒が出ないよう余白を確保
    scale = 1.0 + math.sin(a_rad) * 0.7
    w, h = img.size
    big_w, big_h = int(w * scale), int(h * scale)
    big = img.resize((big_w, big_h), Image.LANCZOS)
    rotated = big.rotate(angle, resample=Image.BICUBIC, expand=False)
    # 中央を元サイズで切り出し
    x = (big_w - w) // 2
    y = (big_h - h) // 2
    return rotated.crop((x, y, x + w, y + h))


# ────────────────────────────────────────────
#  画像処理メイン
# ────────────────────────────────────────────
def process_image(img: Image.Image, params: dict, for_save: bool = False) -> Image.Image:
    angle = float(params.get("rotate", 0))
    if angle != 0:
        img = _rotate_no_black(img, angle)

    # トリミング
    crop = params.get("crop", "4:5")
    if crop != "original":
        target = 4 / 5 if crop == "4:5" else 1.0
        w, h = img.size
        if (w / h) > target:
            new_w = int(h * target)
            img = img.crop(((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h))
        else:
            new_h = int(w / target)
            img = img.crop((0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h))

    # 温かみ
    warmth = int(float(params.get("warmth", 0)))
    if warmth != 0:
        r, g, b = img.split()
        r = r.point(lambda x: min(255, max(0, x + warmth)))
        b = b.point(lambda x: min(255, max(0, x - warmth)))
        img = Image.merge("RGB", (r, g, b))

    img = ImageEnhance.Brightness(img).enhance(float(params.get("brightness", 1.0)))
    img = ImageEnhance.Contrast(img).enhance(float(params.get("contrast", 1.0)))
    img = ImageEnhance.Color(img).enhance(float(params.get("saturation", 1.0)))
    img = ImageEnhance.Sharpness(img).enhance(float(params.get("sharpness", 1.0)))

    # 保存時フルサイズ
    if for_save:
        if crop == "4:5":
            img = img.resize((1080, 1350), Image.LANCZOS)
        elif crop == "1:1":
            img = img.resize((1080, 1080), Image.LANCZOS)
        else:
            w, h = img.size
            sc = 1080 / max(w, h)
            img = img.resize((int(w * sc), int(h * sc)), Image.LANCZOS)

    text = params.get("text", "").strip()
    if text:
        img = _draw_text(img, params)

    return img


# ────────────────────────────────────────────
#  テキスト描画
# ────────────────────────────────────────────
def _draw_text(img: Image.Image, params: dict) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = max(10, int(float(params.get("text_size", 72)) * w / 540))

    font_paths = [
        r"C:\Windows\Fonts\YuGothB.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    x = int(w * float(params.get("text_x", 50)) / 100)
    y = int(h * float(params.get("text_y", 15)) / 100)
    color = params.get("text_color", "#FFFFFF")
    shadow = str(params.get("text_shadow", "true")).lower() == "true"
    lines = params.get("text", "").split("\n")
    line_h = int(font_size * 1.35)
    start_y = y - (line_h * len(lines)) // 2

    for i, line in enumerate(lines):
        ly = start_y + i * line_h
        if shadow:
            off = max(2, font_size // 20)
            draw.text((x + off, ly + off), line, font=font, fill=(0, 0, 0), anchor="mm")
        draw.text((x, ly), line, font=font, fill=color, anchor="mm")

    return img


# ────────────────────────────────────────────
#  自動最適化（画像分析）
#  Instagramにはない機能：画像を解析して最良の設定を返す
# ────────────────────────────────────────────
def _sample_pixels(gray: Image.Image, n: int = 5000) -> list[int]:
    data = list(gray.getdata())
    step = max(1, len(data) // n)
    return data[::step]


def claude_analyze(img: Image.Image, text: str) -> dict:
    """Claude Vision で画像を見て最適パラメータを判断する"""
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 環境変数から自動取得

    # API送信用に縮小
    preview = img.copy()
    preview.thumbnail((1000, 1250), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = f"""あなたはInstagramの写真編集とビジュアルデザインの専門家です。
この写真を見て、インスタ映えする投稿画像にするための最適な編集パラメータをJSONで返してください。

画像に重ねるテキスト: 「{text if text else "（テキストなし）"}」

以下を分析して判断してください：
- 被写体の傾き（瓶・料理・人物が真っ直ぐかどうか）
- 画像の明るさ・コントラスト・色味の状態
- ボケや手ブレの有無
- テキストを置くなら被写体と重ならない空きスペースはどこか
- 背景色に対して文字が読みやすい色は何か

返すJSONのキーと範囲：
{{
  "rotate": -15〜15（傾き補正。小数可。今の傾きを見て判断）,
  "brightness": 0.5〜2.0（1.0が元画像）,
  "contrast": 0.5〜2.0,
  "saturation": 0.5〜2.5（インスタ映えのため通常1.3〜1.6程度）,
  "warmth": -50〜50（正で暖色、負で寒色）,
  "sharpness": 0〜3.0（ボケていれば1.5以上）,
  "crop": "4:5" または "1:1" または "original",
  "text_x": 0〜100（テキスト横位置%、50が中央）,
  "text_y": 0〜100（テキスト縦位置%、上から何%か）,
  "text_color": "#FFFFFF" や "#000000" など（背景に対して読みやすい色）,
  "text_size": 20〜160（文字数と余白のバランスで決定）,
  "text_shadow": "true" または "false",
  "reason": "判断の根拠を日本語で3〜4文。なぜその設定にしたのかを具体的に説明"
}}

JSONのみ返してください。前後の説明文は不要です。"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )

    raw = response.content[0].text.strip()
    # コードブロックがあれば除去
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)


def auto_analyze(img: Image.Image, text: str) -> dict:
    # まず4:5クロップで分析
    w, h = img.size
    if (w / h) > 4 / 5:
        nw = int(h * 4 / 5)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    w, h = img.size

    gray = img.convert("L")
    pixels = _sample_pixels(gray)
    avg_b = sum(pixels) / len(pixels)

    # ──── 明るさ ────
    # 目標輝度130に合わせる（暗すぎ/明るすぎを補正）
    brightness = round(min(1.8, max(0.7, 130 / max(avg_b, 1))), 2)

    # ──── コントラスト ────
    try:
        std = statistics.stdev(pixels)
    except Exception:
        std = 40
    contrast = round(min(1.6, max(1.0, 70 / max(std, 1))), 2)

    # ──── 彩度 ────
    # 常にやや上げる（インスタ映え）
    saturation = 1.45

    # ──── 温かみ ────
    r_ch = list(img.split()[0].getdata())[::10]
    b_ch = list(img.split()[2].getdata())[::10]
    r_avg = sum(r_ch) / len(r_ch)
    b_avg = sum(b_ch) / len(b_ch)
    warmth = 0 if (r_avg - b_avg) > 25 else 8  # 既に温かければ加えない

    # ──── シャープネス ────
    sharpness = 1.5

    # ──── テキスト最適配置 ────
    # 画像を上/下/左上/右上/左下/右下の候補ゾーンに分け
    # 最もぼんやりしているゾーン（分散が低い＝背景が単純）を選ぶ
    candidates = [
        (50, 12), (50, 88),   # 上中央・下中央
        (50, 20), (50, 80),   # 上寄り・下寄り
        (25, 15), (75, 15),   # 左上・右上
        (25, 85), (75, 85),   # 左下・右下
    ]
    best_pos = (50, 15)
    min_var = float("inf")
    zone_w, zone_h = w // 4, h // 8
    for cx_pct, cy_pct in candidates:
        cx = int(w * cx_pct / 100)
        cy = int(h * cy_pct / 100)
        x1 = max(0, cx - zone_w // 2)
        y1 = max(0, cy - zone_h // 2)
        x2 = min(w, x1 + zone_w)
        y2 = min(h, y1 + zone_h)
        region = gray.crop((x1, y1, x2, y2))
        rp = list(region.getdata())
        if len(rp) < 4:
            continue
        try:
            var = statistics.variance(rp)
        except Exception:
            var = 0
        if var < min_var:
            min_var = var
            best_pos = (cx_pct, cy_pct)

    tx, ty = best_pos

    # ──── 文字色（背景輝度から自動選択）────
    sx = int(w * tx / 100)
    sy = int(h * ty / 100)
    zw = min(60, w // 4)
    sample = gray.crop((max(0, sx - zw), max(0, sy - zw // 2),
                         min(w, sx + zw), min(h, sy + zw // 2)))
    sp = list(sample.getdata())
    bg_lum = sum(sp) / max(len(sp), 1)
    text_color = "#000000" if bg_lum > 140 else "#FFFFFF"

    # ──── 文字サイズ（文字数から自動）────
    char_count = len(text.replace("\n", "").replace(" ", ""))
    if char_count <= 8:
        text_size = 96
    elif char_count <= 15:
        text_size = 76
    elif char_count <= 30:
        text_size = 60
    else:
        text_size = 46

    return {
        "rotate": 0,
        "crop": "4:5",
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "warmth": warmth,
        "sharpness": sharpness,
        "text_x": tx,
        "text_y": ty,
        "text_color": text_color,
        "text_size": text_size,
        "text_shadow": "true",
    }


# ────────────────────────────────────────────
#  ユーティリティ
# ────────────────────────────────────────────
def _to_base64(img: Image.Image, max_h: int = 900) -> str:
    img = img.copy()
    if img.height > max_h:
        r = max_h / img.height
        img = img.resize((int(img.width * r), max_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ────────────────────────────────────────────
#  Flask ルート
# ────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global _stored_image
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no file"}), 400
    img = Image.open(f.stream).convert("RGB")
    img = ImageOps.exif_transpose(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    _stored_image = buf.getvalue()
    return jsonify({"preview": _to_base64(img), "w": img.width, "h": img.height})


@app.route("/preview", methods=["POST"])
def preview():
    if _stored_image is None:
        return jsonify({"error": "no image"}), 400
    params = request.get_json() or {}
    img = Image.open(io.BytesIO(_stored_image)).convert("RGB")
    img = process_image(img, params, for_save=False)
    return jsonify({"preview": _to_base64(img)})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Claude Vision で画像を分析して最適パラメータを返す"""
    if _stored_image is None:
        return jsonify({"error": "no image"}), 400
    data = request.get_json() or {}
    text = data.get("text", "")
    img = Image.open(io.BytesIO(_stored_image)).convert("RGB")
    img = ImageOps.exif_transpose(img)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            params = claude_analyze(img, text)
            params["source"] = "claude"
            return jsonify(params)
        except Exception as e:
            print(f"[Claude API error] {e}")
            # フォールバック
    params = auto_analyze(img, text)
    params["source"] = "pixel"
    return jsonify(params)


@app.route("/download", methods=["POST"])
def download():
    if _stored_image is None:
        return jsonify({"error": "no image"}), 400
    params = request.get_json() or {}
    img = Image.open(io.BytesIO(_stored_image)).convert("RGB")
    img = process_image(img, params, for_save=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg",
                     as_attachment=True, download_name="insta_post.jpg")


if __name__ == "__main__":
    ip = get_lan_ip()
    print("\n🍶 インスタ画像エディター 起動中...")
    print(f"   PC     : http://localhost:5000")
    print(f"   iPhone : http://{ip}:5000")
    print("   ※ iPhoneとPCが同じWi-Fiに接続していること\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
