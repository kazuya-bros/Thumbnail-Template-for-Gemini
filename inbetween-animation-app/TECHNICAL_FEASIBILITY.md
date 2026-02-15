# 中割りアニメーション自動生成アプリ - 技術要件・実現可能性分析

## 概要

2枚のイラスト/アニメ調キーフレーム画像（目開き口閉じ / 目閉じ口開き）から、滑らかな口パク・瞬きアニメーションを自動生成するWindowsデスクトップアプリ。

**2つのアプローチを並行開発**:
- **Track A（画像生成）**: 中割り画像を1枚ずつ精密に生成 → フレームを連結してアニメーション化
- **Track B（動画生成）**: 2枚の画像から直接アニメーション動画を生成

### 前提条件

- **入力画像**: イラスト・アニメ調（実写ではない）
- **ターゲット環境**: Windows、GPU 24GB VRAM まで利用可
- **軽量性**: 必須ではない（品質優先）

---

## Track A: 画像生成アプローチ

### コンセプト

2枚の入力画像間の **任意の比率（1/4, 1/2, 3/4等）で中割り画像を1枚ずつ生成** し、フレーム列としてアニメーション化する。

**利点**: 目・口を独立した比率で精密制御可能
**欠点**: フレーム間の一貫性を自分で管理する必要がある

### A-1. セグメンテーション（目・口の領域特定）

全モードで共通の前処理。目と口を独立制御するためにマスクが必要。

| 技術 | モデルサイズ | VRAM | アニメ対応 | 精度 | 備考 |
|------|-----------|------|----------|------|------|
| **Anime-Face-Segmentation** | ~15MB | ~200MB | ◎ 専用 | ○ | 7クラス(eye,mouth等)、MobileNetV2。[GitHub](https://github.com/siyeong0/Anime-Face-Segmentation) |
| Anzhc's YOLOs | ~20MB | ~300MB | ◎ 専用 | ○ | Adetailer用、強膜精密検出。[HuggingFace](https://huggingface.co/Anzhc/Anzhcs_YOLOs) |
| SAM 3 | ~3.4GB | 4-6GB | ○ 汎用 | ◎ | テキストプロンプト"eye","mouth"。[GitHub](https://github.com/facebookresearch/sam3) |
| anime-face-detector | ~150MB | ~500MB | ◎ 専用 | ○ | 28点ランドマーク。[GitHub](https://github.com/hysts/anime-face-detector) |

**推奨**: Anime-Face-Segmentation をメイン、SAM 3 をフォールバック

### A-2. 中割り画像の生成エンジン

#### A-2a. ローカル従来手法

| 技術 | 方式 | VRAM | 速度 | 比率制御 | 品質 |
|------|------|------|------|---------|------|
| **αブレンド+マスク** | ピクセル補間 | 0 | <0.1秒 | ◎ 精密 | △ 半透明になりがち |
| **Poisson Blending** | 勾配領域合成 | 0 | ~0.5秒 | ◎ 精密 | ○ 境界が自然 |
| **RIFE v4.25** | ニューラル補間 | 1-2GB | ~1秒 | ◎ 精密 | ○ 光学フロー的 |

#### A-2b. 自然言語AI画像編集（ローカル実行可能）

| 技術 | パラメータ | VRAM (24GB) | 速度 | 品質 | ライセンス |
|------|----------|------------|------|------|----------|
| **Qwen Image Edit** | 20B | 24GB (CPU offload必要) | ~14秒 (Lightning 4step) | ◎ | Apache-2.0 |
| Qwen Image Edit (Q4量子化) | 20B→~13GB | ~16GB | ~20秒 | ○ やや劣化 | Apache-2.0 |

**Qwen Image Edit 詳細**:
- [GitHub](https://github.com/QwenLM/Qwen-Image) / [HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Edit)
- Qwen2.5-VLで入力画像のセマンティック理解 + VAEで見た目保持を同時実現
- **アニメ特化LoRA**あり: [Qwen-Image-Edit-2511-Anime](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-2511-Anime)
- 24GB GPU: `enable_model_cpu_offload()` + Lightning LoRA (4step) で動作。CPU RAM 50GB推奨
- DFloat11圧縮（ロスレス32%縮小）: [DFloat11/Qwen-Image-Edit-DF11](https://huggingface.co/DFloat11/Qwen-Image-Edit-DF11)
- 4bit GGUF版もあり: 16GB以下でも動作（品質やや低下）
- ComfyUI / DiffSynth-Studio 統合済み
- **プロンプト例**:
  ```
  この画像のアニメキャラクターの目を25%だけ閉じ、口を25%だけ開いた状態にしてください。
  それ以外は一切変更しないでください。
  ```

#### A-2c. 自然言語AI画像編集（API）

| 技術 | 提供 | コスト | 速度 | 品質 | オフライン |
|------|------|-------|------|------|----------|
| **Nano Banana** | Google Gemini API | ~$0.039/枚 | 3-5秒 | ◎ | × |
| **Nano Banana Pro** | Google Gemini API | ~$0.06/枚 | 5-8秒 | ◎◎ | × |
| **Seedream 4.5** | ByteDance API | 要問合せ | ~2秒 | ◎◎ | × |

- [Nano Banana 公式ドキュメント](https://ai.google.dev/gemini-api/docs/image-generation)
- [Seedream 公式](https://seed.bytedance.com/en/seedream4_5)
- Seedream 4.5: ELOスコアでGemini超え。ただし**重みは非公開**、API経由のみ

### A-3. 合成パイプライン

```
入力: 画像A(目開・口閉) + 画像B(目閉・口開) + 比率パラメータ

Step 1: セグメンテーション → eye_mask, mouth_mask
Step 2: AI画像編集 or 従来手法 → 中割り画像生成
Step 3: マスクベース合成 → 目・口以外は元画像を保持
Step 4: フレーム列をアニメーション化
```

### A-4. Track A 全体比較

| 構成 | オフライン | 品質 | 速度 | 精密制御 | 導入難易度 |
|------|----------|------|------|---------|----------|
| αブレンド + マスク | ○ | △ | ◎ 最速 | ◎ | ★☆☆ |
| RIFE + マスク | ○ | ○ | ○ | ◎ | ★★☆ |
| **Qwen Image Edit + マスク** | **○** | **◎** | **○ 14秒/枚** | **◎** | **★★☆** |
| Nano Banana + マスク | × | ◎ | ○ | ◎ | ★★☆ |
| Seedream 4.5 + マスク | × | ◎◎ | ◎ | ◎ | ★★☆ |

**Track A 推奨構成**: Qwen Image Edit（ローカル） + Anime-Face-Segmentation + マスク合成
フォールバック: Nano Banana API / RIFE / αブレンド

---

## Track B: 動画生成アプローチ

### コンセプト

入力画像からプロンプトで **「瞬きして口を開ける」動画を直接生成**。比率の精密制御はしない代わりに、AI が自然なタイミング・動きの動画を一発生成する。

**利点**: 自然なモーション生成、フレーム間の一貫性が保証される
**欠点**: 比率の精密制御は困難、背景が変わるリスクあり

### B-1. ローカル動画生成モデル

#### FramePack（★最も手軽）

- [GitHub](https://github.com/lllyasviel/FramePack) / Apache-2.0
- **6GB VRAMで動作** — 24GBなら余裕すぎる
- 内部はHunyuanベース（~13B）、フレーム順次生成
- RTX 4090: 2.5秒/フレーム（未最適化）、1.5秒/フレーム（TeaCache）
- 5秒動画: ~10分（RTX 4090）
- 逆順生成（inverted anti-drifting）: 入力画像が高品質なアンカーになる
- F1 / P1 バージョンで反ドリフト精度が向上
- **FramePack Studio**: マルチ機能拡張版（キュー、エンドフレーム制御等）

**中割りへの応用**:
```
入力: 画像A（目開・口閉）
プロンプト: "The anime character slowly blinks their eyes and opens their mouth"
→ 数秒の動画が生成される
→ 各フレームを中割り画像として抽出
```

**限界**: 変化が最初の1-2秒に集中し残りが静止画になりがち

#### Wan 2.2（★最高品質ローカル）

- [GitHub](https://github.com/Wan-Video/Wan2.2) / Apache-2.0
- MoE構造: 27B総パラメータ（14Bアクティブ）
- Image-to-Video (I2V): `Wan2.2-I2V-A14B`
- 480P/720P対応
- **24GB GPU**: `--offload_model True --t5_cpu` で動作（推論は遅くなる）
  - 軽量版 `Wan2.2-TI2V-5B` なら24GBで比較的快適
- ComfyUI / Diffusers / DiffSynth-Studio 統合済み
- LightX2V で 8GB VRAM (RTX 4060) でも動作可能

**中割りへの応用**:
```
入力: 画像A + プロンプト
python generate.py --task i2v-A14B --image eyes_open.png \
  --prompt "The anime character gently blinks and opens mouth to speak"
→ 高品質な数秒動画 → フレーム抽出
```

#### LTX-2（★高速・高解像度）

- [GitHub](https://github.com/Lightricks/LTX-2) / Apache-2.0
- DiTベース: 14B(映像) + 5B(音声)
- 4K/50FPS、20秒連続生成
- Wan 2.2の**約18倍高速**
- **24GB GPU**: FP8 + NVFP8最適化で動作
- モデルDL: ~30GB
- NVIDIA RTXとの統合最適化あり（CES 2026発表）

**中割りへの応用**:
```
入力: 画像A → Image-to-Video
高速生成だが、24GBではFP8必須
Wan 2.2ほどの品質は出にくい（トレードオフ）
```

### B-2. Track B 全体比較

| モデル | VRAM | 速度(5秒動画) | 品質 | 制御性 | 導入容易性 |
|--------|------|-------------|------|--------|----------|
| **FramePack** | **6GB** | ~10分 | ○ | △ プロンプト | **★☆☆ 最も簡単** |
| **Wan 2.2 (TI2V-5B)** | ~12GB | ~15分 | ◎ | △ プロンプト | ★★☆ |
| Wan 2.2 (I2V-A14B) | ~24GB* | ~20分 | ◎◎ | △ プロンプト | ★★★ |
| LTX-2 (FP8) | ~24GB | ~5分 | ○ | △ プロンプト | ★★★ |

*offload使用時

**Track B 推奨構成**: FramePack（手軽・6GB） + Wan 2.2 TI2V-5B（高品質）

### B-3. 動画生成後の後処理

動画をそのまま使うのではなく、Track A の技術と組み合わせて品質を上げる:

```
Step 1: 動画生成AI → 粗い動画（背景がブレる可能性あり）
Step 2: 各フレームから Anime-Face-Segmentation でマスク抽出
Step 3: 目・口領域のみ生成動画から採用、背景は元画像で差し替え
Step 4: [Optional] RIFE でフレーム間をさらに滑らかに補間
```

---

## Track A と Track B の関係

```
┌─────────────────────────────────────────────────────────────┐
│                    入力: 画像A + 画像B                        │
│                 (目開・口閉)  (目閉・口開)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐        ┌───────────────┐
  │   Track A     │        │   Track B     │
  │  画像生成      │        │  動画生成      │
  │               │        │               │
  │ Qwen Image    │        │ FramePack     │
  │ Edit          │        │ Wan 2.2       │
  │ Nano Banana   │        │ LTX-2         │
  │ RIFE          │        │               │
  │ αブレンド      │        │               │
  └───────┬───────┘        └───────┬───────┘
          │                         │
          │  中割り画像列             │  動画フレーム列
          │                         │
          └────────────┬────────────┘
                       ▼
            ┌─────────────────────┐
            │    共通後処理         │
            │                     │
            │ セグメンテーション     │
            │ マスクベース合成       │
            │ 背景保護             │
            │ RIFE追加補間          │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │    出力              │
            │ GIF / APNG / MP4    │
            │ 連番PNG              │
            └─────────────────────┘
```

### 使い分けの指針

| ユースケース | 推奨Track |
|-------------|----------|
| 目・口の比率を細かく指定したい | **Track A** (Qwen / Nano Banana) |
| とにかく自然な動きが欲しい | **Track B** (FramePack / Wan) |
| オフライン・ローカル完結 | **Track A** (Qwen) or **Track B** (FramePack) |
| 最速でプロトタイプ | **Track A** (αブレンド) |
| 最高品質 | **Track A** (Qwen) + **Track B** (Wan) の良いとこ取り |

---

## GPU利用計画（24GB VRAM）

### Track A 実行時
| モデル | VRAM | 備考 |
|--------|------|------|
| Anime-Face-Segmentation | ~200MB | 常駐 |
| RIFE v4.25 | ~1-2GB | 常駐可 |
| Qwen Image Edit (CPU offload) | ~20GB | メイン処理中のみ |
| **合計** | **~22GB** | ギリギリだが収まる |

### Track B 実行時
| モデル | VRAM | 備考 |
|--------|------|------|
| FramePack | ~6GB | 最も軽い |
| Wan 2.2 TI2V-5B | ~12GB | 中品質 |
| Wan 2.2 I2V-A14B (offload) | ~20GB | 高品質 |
| LTX-2 (FP8) | ~20GB | 高速 |

**注意**: Track A (Qwen) と Track B (Wan/LTX) は同時実行できない。切り替えて使う。
FramePack (6GB) なら Track A のセグメンテーション + RIFE と併用可能。

---

## リスクと対策

### 共通リスク

| リスク | 対策 |
|--------|------|
| アニメ画風の多様性でセグメンテーション精度低下 | SAM 3 フォールバック / 手動マスク調整UI |
| 目の開閉は形状変化が大きく不自然になりやすい | AI画像編集 or 動画生成で意味的に解決 |
| 24GB VRAM不足 | 量子化 / CPU offload / 軽量モデル選択 |

### Track A 固有リスク

| リスク | 対策 |
|--------|------|
| Qwen Image Edit の出力が元画像とスタイル乖離 | マスク合成で目・口以外は元画像保持 |
| AI生成画像の非決定性 | seed固定 / 複数生成から選択UI |
| Gemini/Seedream API依存 | Qwen をローカルプライマリに |

### Track B 固有リスク

| リスク | 対策 |
|--------|------|
| 動画の背景・服装が変わる | セグメンテーション + マスク合成で背景差し替え |
| 精密な比率制御ができない | Track A と併用 / フレーム選択UI |
| 変化が最初に集中し残りが静止画 | プロンプト工夫 / 複数クリップ結合 |

---

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│  Tauri 2.0 (Rust)                                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  WebView (React / Svelte)                                 │  │
│  │  - 画像A/B アップロード                                    │  │
│  │  - Track A: 比率スライダー（目・口独立、連続値）             │  │
│  │  - Track B: プロンプト入力 + 動画プレビュー                 │  │
│  │  - Track A+B 統合: タイムライン + フレーム選択              │  │
│  │  - モード切替（Track A / Track B / ハイブリッド）           │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                           │ IPC                                  │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │  Rust Backend                                             │  │
│  │  - ファイルI/O + 画像キャッシュ                             │  │
│  │  - Python Sidecar プロセス管理                             │  │
│  │  - GPUリソース管理（Track A/B モデルの排他制御）             │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                           │ JSON-RPC                             │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │  Python Backend                                           │  │
│  │                                                           │  │
│  │  [共通] セグメンテーション                                  │  │
│  │  ├── Anime-Face-Segmentation (MobileNetV2, ~200MB)        │  │
│  │  └── SAM 3 (フォールバック, ~4-6GB)                        │  │
│  │                                                           │  │
│  │  [Track A] 画像生成エンジン                                │  │
│  │  ├── Qwen Image Edit (ローカル, 20B, ~20GB) ← Primary     │  │
│  │  ├── Nano Banana / Gemini API (オンライン)                 │  │
│  │  ├── Seedream 4.5 API (オンライン、最高品質)               │  │
│  │  ├── RIFE v4.25 (ニューラル補間, ~1-2GB)                  │  │
│  │  └── OpenCV (αブレンド / Poisson, CPU)                    │  │
│  │                                                           │  │
│  │  [Track B] 動画生成エンジン                                │  │
│  │  ├── FramePack (Hunyuan, ~6GB) ← 手軽                    │  │
│  │  ├── Wan 2.2 TI2V-5B (~12GB) ← バランス                  │  │
│  │  ├── Wan 2.2 I2V-A14B (~20GB) ← 高品質                   │  │
│  │  └── LTX-2 FP8 (~20GB) ← 高速                            │  │
│  │                                                           │  │
│  │  [共通] 合成・後処理                                       │  │
│  │  ├── マスクベース合成 (OpenCV)                              │  │
│  │  ├── RIFE 追加補間                                         │  │
│  │  └── 出力 (GIF/APNG/MP4/PNG連番)                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 開発ロードマップ

### Phase 1: PoC（両Track並行で概念実証）

**Track A PoC**:
1. Anime-Face-Segmentation で目・口マスク抽出テスト
2. αブレンド + Poisson Blending で最もシンプルな中割り確認
3. RIFE でローカル補間の品質確認
4. Qwen Image Edit ローカル推論テスト（CPU offload + Lightning）
5. Nano Banana API テスト

**Track B PoC**:
1. FramePack で画像→動画生成テスト（最も手軽、6GB）
2. 生成動画からフレーム抽出 + 品質確認
3. Wan 2.2 TI2V-5B テスト
4. セグメンテーション + マスク合成で背景差し替えテスト

**品質比較**: 同じ入力画像で全手法の結果を並べて比較

### Phase 2: エンジン統合

1. Track A / B の最良手法を選定
2. 共通の後処理パイプライン構築
3. GPUリソース管理（モデルのロード/アンロード切り替え）
4. ハイブリッドモード（Track A の精密制御 + Track B の自然なモーション）

### Phase 3: デスクトップアプリ化

1. Tauri 2.0 + React/Svelte でUI構築
2. Track A: 比率スライダーUI（目・口独立）
3. Track B: プロンプト入力 + 動画プレビューUI
4. 統合タイムラインエディタ
5. Python Sidecar連携 + GPUリソース管理

### Phase 4: 完成度向上

1. アニメーション書き出し（GIF/APNG/MP4）
2. プリセットパターン（口パク・瞬きの自然なタイミング）
3. マスク手動調整UI
4. バッチ処理（複数表情パターン一括生成）
5. LoRA微調整（特定キャラスタイルへの最適化）
