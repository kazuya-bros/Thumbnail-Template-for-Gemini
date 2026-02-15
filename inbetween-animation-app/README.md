# Inbetween Animation App

2枚のアニメ/イラスト画像から中割りフレームを自動生成し、滑らかな口パク・瞬きアニメーションを作成するツール。

## セットアップ

```bash
cd inbetween-animation-app
pip install -r requirements.txt
```

## 使い方

### 1. 単一フレーム生成

```bash
python -m src.cli single \
  --image-a samples/input/eyes_open.png \
  --image-b samples/input/eyes_closed.png \
  --eye-ratio 0.5 --mouth-ratio 0.5 \
  --output samples/output/inbetween.png
```

### 2. バッチ生成（複数の中割り）

```bash
# 4分割（0%, 25%, 50%, 75%, 100%）
python -m src.cli batch \
  --image-a samples/input/eyes_open.png \
  --image-b samples/input/eyes_closed.png \
  --steps 4 \
  --output-dir samples/output/batch
```

### 3. アニメーション生成

```bash
# 瞬き+口パクアニメーション
python -m src.cli animate \
  --image-a samples/input/eyes_open.png \
  --image-b samples/input/eyes_closed.png \
  --fps 24 --format gif --preset blink_talk \
  --output samples/output/animation.gif
```

### 4. セグメンテーション確認

```bash
python -m src.cli masks \
  --image samples/input/eyes_open.png \
  --output samples/output/masks.png
```

## 処理モード

| モード | コマンド | 説明 |
|--------|---------|------|
| `blend` | `--mode blend` | αブレンド（最速・デフォルト） |
| `poisson` | `--mode poisson` | Poisson合成（境界が自然） |
| `rife` | `--mode rife` | RIFE神経補間（高品質・GPU） |
| `gemini` | `--mode gemini` | Gemini API（最高品質・要API） |
| `hybrid` | `--mode hybrid` | Gemini + マスク合成（推奨） |

## 入力画像の要件

- **画像A**: 目が開いていて口が閉じている
- **画像B**: 目が閉じていて口が開いている
- 同一キャラクター、同一ポーズ、同一背景
- PNG/JPEG形式

## アーキテクチャ

```
src/
├── cli.py              # CLIインターフェース
├── pipeline.py         # メインパイプライン
├── segmentation.py     # アニメ顔セグメンテーション
├── inbetween_blend.py  # αブレンド方式
├── inbetween_rife.py   # RIFE補間方式
├── inbetween_gemini.py # Gemini API方式
└── output.py           # 出力（GIF/APNG/MP4/PNG連番）
```
