# 中割りアニメーション自動生成アプリ - 技術要件・実現可能性分析

## 概要

2枚のイラスト/アニメ調キーフレーム画像（目開き口閉じ / 目閉じ口開き）から、任意の比率（1/4, 1/2, 3/4等）で中割り画像を自動生成し、滑らかな口パク・瞬きアニメーションを作成するWindowsデスクトップアプリ。

### 前提条件

- **入力画像**: イラスト・アニメ調（実写ではない）
- **ターゲット環境**: Windows、GPU 24GB VRAM まで利用可
- **軽量性**: 必須ではない（品質優先）

---

## イラスト/アニメ顔における技術的課題

実写顔とアニメ顔では技術要件が根本的に異なる:

| 項目 | 実写 | アニメ/イラスト |
|------|------|----------------|
| ランドマーク検出 | dlib/MediaPipe で高精度 | **標準ツールが使えない** |
| 顔の特徴 | 写実的、3D構造あり | デフォルメ、2D的、スタイル多様 |
| 目の表現 | 瞳孔・虹彩が自然 | 大きな目、ハイライト、特殊な表現 |
| 口の表現 | 唇の形状変化 | 線画ベース、シンプルな形状変化 |
| 既存ツール | 豊富 | **限定的** |

---

## コア技術の分析（アニメ特化・GPU活用前提）

### 1. 顔パーツのセグメンテーション（目・口の領域特定）

#### 選択肢A: Anime-Face-Segmentation（MobileNetV2ベース）★推奨
- **実現性: ◎**
- アニメ顔専用セマンティックセグメンテーション
- 7クラス分類: `background, hair, eye, mouth, face, skin, clothes`
- 入力: 3x512x512 → 出力: 7x512x512 のマスクマップ
- MobileNetV2エンコーダで軽量（GPUなくても動く）
- [GitHub: siyeong0/Anime-Face-Segmentation](https://github.com/siyeong0/Anime-Face-Segmentation)

#### 選択肢B: Anzhc's YOLOモデル（アニメ特化）
- **実現性: ○**
- Adetailer用に訓練されたアニメ目・顔検出モデル
- 強膜領域の精密検出（まつげ・外周を除外）
- mAP 0.72〜0.87
- [Hugging Face: Anzhc/Anzhcs_YOLOs](https://huggingface.co/Anzhc/Anzhcs_YOLOs)

#### 選択肢C: SAM 3（Segment Anything Model 3）
- **実現性: ○**（GPU 24GBなら十分動作）
- 848Mパラメータ、テキストプロンプトで「eye」「mouth」指定可能
- アニメ・イラストでも概念ベースで検出できるのが強み
- 汎用的だが、アニメ特化モデルより精度が劣る可能性
- [GitHub: facebookresearch/sam3](https://github.com/facebookresearch/sam3)

#### 選択肢D: anime-face-detector（28点ランドマーク）
- **実現性: ○**
- mmdet + mmpose ベースのアニメ顔ランドマーク検出
- 28点と少なめだが、アニメ顔に特化
- [GitHub: hysts/anime-face-detector](https://github.com/hysts/anime-face-detector)

### 2. 中割り画像の生成（最重要）

#### アプローチ1: Nano Banana（Gemini API）によるAI中割り生成 ★最有力
- **実現性: ◎**
- **仕組み**: 2枚の画像を入力し、「この2枚の間の中間状態を生成して」とプロンプト
- **最大の強み**: セマンティックな理解
  - 「目を1/4だけ閉じた状態」「口を半分開いた状態」を**意味的に理解**して生成
  - 単純なピクセル補間ではなく、「半開きの目」の**概念**を描画できる
- キャラクター一貫性の維持が得意
- **API**: `gemini-2.5-flash-image-preview` or `gemini-3-pro-image-preview`
- **コスト**: ~$0.039/画像（~1,290トークン/枚）
- **プロンプト例**:
  ```
  I have two illustrations of the same anime character.
  Image A: eyes fully open, mouth closed.
  Image B: eyes closed, mouth open.
  Generate the intermediate state where eyes are 25% closed and mouth is 25% open.
  Keep the exact same art style, colors, and all other details identical.
  ```
- **デメリット**: API依存（オフライン不可）、レイテンシ（数秒/枚）、結果が非決定的

#### アプローチ2: マスクベース合成（ローカル処理）★堅実
- **実現性: ◎**
- パイプライン:
  1. Anime-Face-Segmentation で目・口のマスクを抽出
  2. 目の領域のみ画像AとBをα値でブレンド
  3. 口の領域のみ画像AとBを別のα値でブレンド
  4. Poisson Blending（シームレスクローン）で自然に合成
- **任意の比率サポート**: α値を変えるだけ
- **メリット**: 完全ローカル、高速、決定的、目と口を独立制御
- **デメリット**: 単純なブレンドだと半透明な重なりになりがち（形状変化に弱い）

#### アプローチ3: RIFE + セグメンテーション ★高品質ローカル
- **実現性: ○**
- パイプライン:
  1. セグメンテーションで目・口の領域マスクを取得
  2. 目の領域だけ切り出した2枚の画像をRIFEに入力 → 中間フレーム生成
  3. 口の領域も同様に処理
  4. 元画像に合成
- **メリット**: ニューラルネットによる高品質補間、ローカル処理
- **デメリット**: GPU必要（24GBなら余裕）、形状変化の大きい補間は苦手な場合あり
- [GitHub: hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)

#### アプローチ4: ハイブリッド（Gemini + ローカル後処理）★★★ 最推奨
- **パイプライン**:
  1. **Gemini API** で中間状態の画像を生成（セマンティックに正確な中割り）
  2. **Anime-Face-Segmentation** で目・口マスクを取得
  3. **OpenCV** で元画像とGemini生成画像をマスクベースで合成
     → 目・口以外は元画像を完全保持（背景・服装のブレ防止）
  4. **[Optional] RIFE** でフレーム間をさらに滑らかに補間
- **メリット**:
  - Geminiの意味的理解 + ローカル処理の精密さを両立
  - 背景や服装がブレない（マスク合成で保護）
  - 目・口の独立制御が自然にできる
- **デメリット**: Gemini API依存（ただしフォールバックとしてアプローチ2/3を用意可能）

---

## 推奨アーキテクチャ

```
┌──────────────────────────────────────────────────────┐
│  Tauri 2.0 (Rust)                                    │
│  ┌────────────────────────────────────────────────┐  │
│  │  WebView (React / Svelte)                      │  │
│  │  - 画像A/B アップロード                         │  │
│  │  - 比率スライダー（目・口独立、1/4刻み〜自由）    │  │
│  │  - プレビュー・比較表示                         │  │
│  │  - アニメーション再生・タイムライン              │  │
│  │  - 処理モード選択（AI / Local / Hybrid）        │  │
│  └────────────────────┬───────────────────────────┘  │
│                       │ IPC (invoke)                  │
│  ┌────────────────────▼───────────────────────────┐  │
│  │  Rust Backend                                  │  │
│  │  - ファイルI/O管理                              │  │
│  │  - Python Sidecar プロセス管理                  │  │
│  │  - 画像キャッシュ                               │  │
│  └────────────────────┬───────────────────────────┘  │
│                       │ stdin/stdout JSON-RPC         │
│  ┌────────────────────▼───────────────────────────┐  │
│  │  Python Backend (Sidecar / FastAPI)            │  │
│  │                                                │  │
│  │  [セグメンテーション]                            │  │
│  │  ├── Anime-Face-Segmentation (MobileNetV2)     │  │
│  │  ├── anime-face-detector (28点ランドマーク)      │  │
│  │  └── SAM 3 (フォールバック/高精度モード)         │  │
│  │                                                │  │
│  │  [中割り生成]                                   │  │
│  │  ├── Gemini API (Nano Banana) ← Primary        │  │
│  │  ├── RIFE (ローカル高品質補間)                   │  │
│  │  └── OpenCV Alpha Blend (ローカル高速)           │  │
│  │                                                │  │
│  │  [合成・後処理]                                  │  │
│  │  ├── OpenCV (マスク合成, Poisson Blending)       │  │
│  │  ├── NumPy/SciPy (数値計算)                     │  │
│  │  └── Pillow/FFmpeg (出力生成)                   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## GPU利用計画（24GB VRAM）

| モデル/処理 | VRAM使用量 | 同時利用 |
|------------|-----------|---------|
| Anime-Face-Segmentation | ~200MB | ◎ 常駐可能 |
| anime-face-detector | ~500MB | ◎ 常駐可能 |
| SAM 3 | ~4-6GB | ○ 必要時のみロード |
| RIFE v4.25 | ~1-2GB | ◎ 常駐可能 |
| **合計（通常モード）** | **~2-3GB** | セグメンテーション + RIFE |
| **合計（フルモード）** | **~8-10GB** | 全モデル同時ロード |

24GB VRAMなら全モデル同時ロードでも余裕。

---

## 処理フロー詳細（ハイブリッドモード）

### 入力
```
画像A: 目開き・口閉じ（例: default.png）
画像B: 目閉じ・口開き（例: blink_talk.png）
パラメータ:
  - 目の比率: [0.0, 0.25, 0.5, 0.75, 1.0]  (0=全開, 1=全閉)
  - 口の比率: [0.0, 0.25, 0.5, 0.75, 1.0]  (0=全閉, 1=全開)
```

### Step 1: セグメンテーション
```python
# Anime-Face-Segmentationで目・口マスク取得
mask_eye_A = segment(imageA, class="eye")   # 画像Aの目の領域
mask_mouth_A = segment(imageA, class="mouth") # 画像Aの口の領域
mask_eye_B = segment(imageB, class="eye")
mask_mouth_B = segment(imageB, class="mouth")
# マスクを合成（両画像のunion）+ 余裕を持たせてdilate
mask_eye = dilate(union(mask_eye_A, mask_eye_B), kernel=5)
mask_mouth = dilate(union(mask_mouth_A, mask_mouth_B), kernel=5)
```

### Step 2: 中割り生成（3つのモード）

#### Mode A: Gemini API（高品質・オンライン）
```python
# Gemini APIで中間状態を直接生成
prompt = f"""
Generate an intermediate anime illustration between these two images.
Eye openness: {eye_ratio*100}% closed (0%=fully open, 100%=fully closed)
Mouth openness: {mouth_ratio*100}% open (0%=fully closed, 100%=fully open)
Keep EVERYTHING else identical: art style, colors, hair, clothing, background.
"""
inbetween = gemini.generate(images=[imageA, imageB], prompt=prompt)
```

#### Mode B: RIFE（高品質・ローカル）
```python
# 目の領域だけRIFEで補間
eye_crop_A = crop(imageA, mask_eye)
eye_crop_B = crop(imageB, mask_eye)
eye_inbetween = rife.interpolate(eye_crop_A, eye_crop_B, timestep=eye_ratio)
# 口も同様
mouth_crop_A = crop(imageA, mask_mouth)
mouth_crop_B = crop(imageB, mask_mouth)
mouth_inbetween = rife.interpolate(mouth_crop_A, mouth_crop_B, timestep=mouth_ratio)
```

#### Mode C: Alpha Blend（高速・ローカル）
```python
eye_inbetween = cv2.addWeighted(eye_crop_A, 1-eye_ratio, eye_crop_B, eye_ratio, 0)
mouth_inbetween = cv2.addWeighted(mouth_crop_A, 1-mouth_ratio, mouth_crop_B, mouth_ratio, 0)
```

### Step 3: 合成
```python
# ベース画像（画像A）に目・口の中割りを合成
result = imageA.copy()
# Poisson Blending（シームレスクローン）で自然に合成
result = cv2.seamlessClone(eye_inbetween, result, mask_eye, center_eye, cv2.NORMAL_CLONE)
result = cv2.seamlessClone(mouth_inbetween, result, mask_mouth, center_mouth, cv2.NORMAL_CLONE)
```

### Step 4: アニメーション生成
```python
# 口パク: 閉→開→閉 のサイクル
# 瞬き: 開→閉→開 のサイクル
# これらを組み合わせてフレームシーケンスを生成
frames = []
for eye_r, mouth_r in animation_timeline:
    frame = generate_inbetween(imageA, imageB, eye_ratio=eye_r, mouth_ratio=mouth_r)
    frames.append(frame)
# GIF/APNG/MP4に書き出し
```

---

## 実現可能性の総合評価（改訂版）

### Tier 1: 確実に実現可能（MVP）
| 機能 | 技術 | 難易度 | 備考 |
|------|------|--------|------|
| アニメ顔セグメンテーション | Anime-Face-Segmentation | ★☆☆ | 既存モデルそのまま利用 |
| マスクベースα合成 | OpenCV | ★☆☆ | 最もシンプルな中割り |
| 任意比率の中割り | α値パラメータ化 | ★☆☆ | スライダーUIと連動 |
| デスクトップUI | Tauri 2.0 | ★★☆ | 画像入出力+スライダー |

### Tier 2: 十分に実現可能
| 機能 | 技術 | 難易度 | 備考 |
|------|------|--------|------|
| AI中割り生成 | Gemini API (Nano Banana) | ★★☆ | APIキー設定のみ |
| 高品質ローカル補間 | RIFE v4.25 | ★★☆ | GPU 24GBで余裕 |
| 目・口の独立制御 | マスクベース合成 | ★★☆ | セグメンテーション+合成 |
| Poisson Blending | OpenCV seamlessClone | ★★☆ | 合成の自然さ向上 |

### Tier 3: 追加で対応可能
| 機能 | 技術 | 難易度 | 備考 |
|------|------|--------|------|
| SAM 3 高精度セグメンテーション | SAM 3 | ★★☆ | GPU 24GBで動作 |
| アニメーション書き出し | FFmpeg / Pillow | ★★☆ | GIF/APNG/MP4 |
| タイムラインエディタ | Canvas/WebGL | ★★★ | 口パク・瞬きパターン編集 |
| リアルタイムプレビュー | WebSocket + GPU処理 | ★★★ | スライダー操作に即応 |

---

## リスクと対策

### リスク1: Gemini APIの一貫性
- **問題**: 同じプロンプトでも毎回微妙に異なる画像が生成される
- **対策**: マスクベース合成で目・口以外は元画像を保持。AI生成は目・口領域のみ使用

### リスク2: アニメ画風の多様性
- **問題**: Anime-Face-Segmentationの学習データと異なるスタイルでは精度低下
- **対策1**: SAM 3 をフォールバックとして用意（汎用的な概念理解）
- **対策2**: ユーザーが手動でマスクを調整できるUI

### リスク3: 目の開閉モーフィングの不自然さ
- **問題**: 目を閉じる動作は形状変化が大きく、αブレンドだけでは「透明な重なり」になる
- **対策1**: Gemini APIモードを推奨（意味的に正しい半開き目を生成）
- **対策2**: RIFEモードでオプティカルフロー的な補間
- **対策3**: ユーザーに中間状態の画像を追加で用意してもらうオプション

### リスク4: Gemini APIのコスト・レート制限
- **問題**: 大量の中割り生成でコストが嵩む
- **対策**: ローカルモード（RIFE / αブレンド）をデフォルトに。Gemini はキーフレームのみに使用し、残りはRIFEで補間

---

## 依存パッケージ（Python側）

```
# Core
opencv-python>=4.8
numpy>=1.24
scipy>=1.10
Pillow>=10.0

# Anime Face Segmentation
torch>=2.0
torchvision>=0.15

# Anime Face Detection (optional)
mmdet>=3.0
mmpose>=1.0
anime-face-detector>=0.0.1

# RIFE Frame Interpolation
# (Practical-RIFE をサブモジュールとして組み込み)

# SAM 3 (optional, high-precision mode)
# segment-anything-3 (pip install when needed)

# Gemini API
google-genai>=1.0

# Output
imageio>=2.31  # GIF/APNG
ffmpeg-python>=0.2  # MP4
```

---

## 開発ロードマップ

### Phase 1: PoC（概念実証）— まずここから
1. Python CLIで基本パイプラインを実装
2. Anime-Face-Segmentation で目・口マスク抽出
3. αブレンドによる最もシンプルな中割り生成
4. 結果を目視確認、品質評価

### Phase 2: 品質向上
1. RIFE による高品質ローカル補間
2. Gemini API による AI中割り生成
3. ハイブリッドモード（Gemini + マスク合成）
4. Poisson Blending による自然な合成

### Phase 3: デスクトップアプリ化
1. Tauri 2.0 + React/Svelte でUI構築
2. Python Sidecar連携
3. 目・口独立スライダーUI
4. アニメーションプレビュー

### Phase 4: 完成度向上
1. アニメーション書き出し（GIF/APNG/MP4）
2. タイムラインエディタ
3. プリセットパターン（口パク・瞬き）
4. マスク手動調整UI
