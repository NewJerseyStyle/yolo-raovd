# YOLO-RAOVD Zero-Shot Object Detection Framework Design

## 1. 目標

建立一個可發布於 GitHub 的 zero-shot 物件偵測框架，具備：
- 使用者輸入文字 prompt 或參考影像定義目標物件
- 回傳方框結果、metadata、top-k 相似證據
- 支援 `confidence`/`score` 輸出與 NMS/Threshold
- 提供 COCO benchmark 腳本與評估報表

註：文件刻意避免與現有公司知識產權重疊，僅使用公開模型、公開資料與可重現流程。

## 2. 核心設計

### 2.1 架構總覽

```text
Input Image
  -> YOLO proposal backbone+neck
  -> Region proposals + ROI feature (視覺向量)
  -> Query embedding (text / image reference)
  -> FAISS ANN search (top-k) per region
  -> 聚合 top-k 證據 => retrieval_score
  -> 結合 YOLO objectness 與 class 間 margin
  -> NMS 與阈值過濾
  -> output boxes + metadata + confidence
```

### 2.2 模型選擇（關鍵）

`FAISS` 僅在同一 embedding 空間可做 cosine / inner-product 比對，故文件採兩條索引路徑：

- CLIP path（文字查詢）
  - text query: `CLIP text encoder` -> `text vector`
  - image region: `CLIP image encoder` -> `region vector`
  - 用於「找紅杯子」、「找車牌」等可語意描述目標
- DINOv3 path（影像參考查詢）
  - reference image/query image: `DINOv3 image encoder` -> `image vector`
  - image region: `DINOv3 region visual head` -> `region vector`
  - 適合材質、紋理、特徵細節辨識

工程上提供：
- 雙索引（`clip_index`, `dinov3_index`）或
- 統一向量空間的投影頭（如果已訓練對齊網路才啟用）

初版先實作雙索引，先求可行、可復現、可調參。

## 3. 資料結構

### 3.1 Reference entry

每筆參考資料必須至少包含：
- `id`
- `label`（目標名稱）
- `modality`（`text` 或 `image`）
- `embedding`（float32）
- `metadata`（來源、拍攝條件、版本）

### 3.2 索引元資料

- `namespace`: 可分 `class_id`、`user_id`、`project_id`
- `k` support: 1~N 個同物件樣本（角度/遮擋/尺寸）可共存
- `index_version`: `uuid`，便於回溯

### 3.3 回傳記錄

- `box: [x1,y1,x2,y2]`
- `label`
- `score`（可輸出未校準值）
- `confidence`（建議經校準後）
- `objectness`（YOLO raw）
- `top_k_scores`: `[(ref_id, sim)]`
- `aggregation`: `{mean, max, weighted_mean, lse}`
- `margin`（第一/第二類）
- `support_ratio`

## 4. Top-k 置信度設計（核心）

FAISS top-k 提供的是「相似度證據」，可構成可解釋置信度：

1. 對 region `r` 擷取候選向量 `v_r`
2. 於對應 index 搜尋 `K`：得到 `(ref_id, score_i, class_i)`
3. 每個類別/目標 `c` 聚合其前 `M` 筆分數 `s_i`
4. 聚合策略之一（建議）：
   - `w_i = softmax(s_i / T)` 
   - `S_c = sum(w_i * s_i)`
5. 計算對比項：
   - `margin = S_best - S_second`
   - `consistency = std(scores_top_M)`（越低越穩）
   - `support = mean(scores_top_M > tau)`
6. 融合 YOLO objectness：
   - `raw = sigmoid(a * S_best + b * margin - c * consistency + d * support) * (objectness ^ γ)`
7. 若要當機率輸出，需校準：
   - `confidence = temp_scale(raw)`（溫度縮放 / isotonic / Platt）
   - 以標註驗證集回歸（必要）

注意：
- 這個值可當 `confidence` 使用；但若未校準，文件中應命名為 `retrieval_score` 或 `raw_confidence`。

## 5. 演算法流程

### 5.1 建檔（Index）

1. 輸入 text / image references
2. 分別以 encoder 產生向量
3. 向量正規化（若採 cosine）
4. 入庫：
   - `index.add(embeddings, ids, metadata)`
   - 寫入 `manifest`（schema version、encoder、index type）

### 5.2 偵測

1. YOLO forward，取得 proposals 與 objectness
2. ROIAlign 提取視覺向量
3. query embedding (text/image)
4. top-k ANN search
5. 聚類/分組每類 top-k
6. 計算 `retrieval_score`、`confidence`
7. 依阈值與 NMS 產生最終結果

## 6. NMS/Threshold 策略

- `score` 排序依 `confidence`（或 `retrieval_score`）
- NMS IoU：
  - 文字查詢：`0.5`
  - 文字+細粒度件：`0.4~0.6` 可調
- 最小 proposal 數保底：過濾極小 bbox

## 7. COCO benchmark 規劃

- 支援輸入 COCO annotation JSON（`instances_train2017.json` / `instances_val2017.json`）
- 支援固定 class 名單映射到 prompt
- 指標：
  - `AP`, `AP50`, `AP75`, `AR`，與 IoU 曲線
  - 每類別 AP
  - 速度：`FPS`, `latency per image`
- 輸出：
  - JSON 報表
  - CSV summary
  - `matplotlib` 可視化（可選）

## 8. 目錄規劃（實作階段 1）

```
project/
  src/
  data/
  scripts/
  tests/
  docs/
  README.md
  design.md
```

## 9. 實作里程碑（第一版）

1. 先完成文件與配置 schema（本版）
2. 建立 `ReferenceDB` 與 `FAISS` build/load
3. 完成 `detect` pipeline（YOLO proposal + embedding + retrieval + NMS）
4. 完成 `benchmark/coco.py`
5. 加上可視化輸出（box overlay + confidence metadata）
6. 補齊 license/ethics 說明

## 10. 後續可擴展

- 以 class-wise hard negative 降低誤報
- 加 query expansion（同義詞、短語模板）
- 加 language grounding 的 score calibration（同一圖片多 query cache）
- 導入 vector quantization / OPQ 提升 FAISS 推論吞吐


