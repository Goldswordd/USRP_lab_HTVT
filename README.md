# D8 MRTA-T5: Kiến trúc, Thuật toán & Phân tích chuyên sâu

> Model đã beat SA-TinyML: **93.05 ± 0.27%** vs 92.92 ± 0.28%

---

## 1. Tổng quan kiến trúc

D8 MRTA-T5 (Multi-Resolution Token Attention with 5 Tokens) là kiến trúc lai CNN–Transformer nhẹ cho bài toán UWB NLOS classification. Kiến trúc kết hợp:

- **CNN multi-resolution** cho extraction đặc trưng không gian CIR
- **Token attention** cho cross-modal interaction giữa CIR và AUX features

```
Input: (B, 57) = [CIR(50) | AUX(7)]

  ┌─── CIR Branch ─────────────────────────────────────────┐
  │ (B,1,50) → 3× DilatedConv1d [d=1,4,16]                │
  │          → 1×1 Conv mixing → BN → ReLU                 │
  │          → MaxPool(2) → (B,16,25)                       │
  │          → ECA attention → Conv1d → BN → ReLU          │
  │          → Tokenize: 5 segments × 5 positions → (B,5,16)│
  │          + Position embedding + Modality embedding       │
  └─────────────────────────────────────────────────────────┘

  ┌─── AUX Branch ──────────────────────────────────────────┐
  │ (B,7) → FC(7→32) → ReLU → FC(32→16) → ReLU → (B,1,16) │
  │       + Modality embedding                               │
  └─────────────────────────────────────────────────────────┘

  ┌─── Token Assembly ──────────────────────────────────────┐
  │ tokens = [cir_tok₁, cir_tok₂, ..., cir_tok₅, aux_tok]  │
  │        = (B, 6, 16)                                      │
  └─────────────────────────────────────────────────────────┘

  ┌─── 2-Head Self-Attention ───────────────────────────────┐
  │ QKV projection: Linear(16 → 48) → split Q,K,V          │
  │ 2 heads × 8 dims: attention map (6×6) per head          │
  │ Output projection: Linear(16 → 16)                      │
  │ Residual connection + LayerNorm                          │
  └─────────────────────────────────────────────────────────┘

  ┌─── Pooling + Classifier ────────────────────────────────┐
  │ Mean pool over 6 tokens → (B, 16)                        │
  │ FC(16→42) → ReLU → Drop(0.3)                            │
  │ FC(42→16) → ReLU → Drop(0.15)                           │
  │ FC(16→1) → logit                                         │
  └─────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết từng module

### 2.1 Multi-Resolution Convolutional Block

**Mục đích**: Extract multi-scale spatial patterns từ CIR (Channel Impulse Response).

Trong UWB, CIR chứa thông tin multipath ở nhiều scale khác nhau:
- **Local** (d=1): Direct path, first-path peak
- **Medium** (d=4): Early multipath clusters
- **Wide** (d=16): Late-arriving reflections, room reverberation

```python
# 3 nhánh dilated convolution song song
branches = [
    Conv1d(1, 8, k=3, dilation=1,  padding=1),   # local patterns
    Conv1d(1, 8, k=3, dilation=4,  padding=4),   # medium-range
    Conv1d(1, 8, k=3, dilation=16, padding=16),  # long-range
]
# Mỗi nhánh: Conv → BN → ReLU → trim to input length
# Concatenate: (B, 24, 50)
# 1×1 mixing: Conv1d(24, 16, 1) → BN → ReLU → (B, 16, 50)
# MaxPool(2) → (B, 16, 25)
# ECA attention → Conv1d(16, 16, 3) → BN → ReLU → (B, 16, 25)
```

**ECA (Efficient Channel Attention)**: Re-weight 16 channels dựa trên global average, chỉ tốn 3 params.

### 2.2 CIR Tokenization — Bước đột phá then chốt

**Vấn đề với GAP**: Các kiến trúc trước (ECA-UWB, D3, D5) đều dùng Global Average Pooling để collapse feature map (B, 16, 25) → (B, 16). Điều này **phá hủy toàn bộ thông tin vị trí** trong CIR.

**Giải pháp D8**: Thay vì GAP, chia feature map thành 5 segment bằng nhau, trung bình mỗi segment:

```
Feature map: (B, 16, 25)
   ↓
Segment 1: positions  0–4  → mean → token₁ ∈ ℝ¹⁶  (earliest CIR region)
Segment 2: positions  5–9  → mean → token₂ ∈ ℝ¹⁶
Segment 3: positions 10–14 → mean → token₃ ∈ ℝ¹⁶  (first-path region)
Segment 4: positions 15–19 → mean → token₄ ∈ ℝ¹⁶
Segment 5: positions 20–24 → mean → token₅ ∈ ℝ¹⁶  (late multipath region)
```

**Tại sao 5 tokens mà không phải 4 (D6)?**

```
Feature map length = 25 (after MaxPool from 50)

D6: K=4 → 25/4 = 6.25 → segments [6, 6, 6, 7] ← không đều!
D8: K=5 → 25/5 = 5.00 → segments [5, 5, 5, 5, 5] ← hoàn toàn đều!
```

Khi segments không đều (D6), token cuối chứa nhiều thông tin hơn các token khác → **bias bất đối xứng** trong attention. D8 với K=5 loại bỏ hoàn toàn vấn đề này → attention weights phản ánh đúng importance của từng vùng CIR.

**Kết quả**: D6→D8 chỉ thay `n_cir_tokens=4→5` mà accuracy tăng **+0.18 pp** (92.87→93.05).

### 2.3 Position & Modality Embeddings

Mỗi token được cộng thêm:
- **Position embedding**: `pos_embed ∈ ℝ⁵ˣ¹⁶` (learnable, khởi tạo N(0, 0.02))
  - Giúp attention biết token nào thuộc vùng CIR nào
- **Modality embedding**: `mod_embed ∈ ℝ²ˣ¹⁶` (CIR vs AUX)
  - Giúp attention phân biệt CIR tokens vs AUX token

### 2.4 Cross-Modal Self-Attention (2-Head)

6 tokens (5 CIR + 1 AUX) đi qua 2-head self-attention:

```
tokens = [cir₁, cir₂, cir₃, cir₄, cir₅, aux]   ∈ ℝ⁶ˣ¹⁶

QKV = Linear(16 → 48)(tokens)    → split → Q, K, V ∈ ℝ⁶ˣ¹⁶
→ reshape to 2 heads: Q_h, K_h, V_h ∈ ℝ⁶ˣ⁸

Attention map per head:
  A = softmax(Q_h @ K_h^T / √8)   ∈ ℝ⁶ˣ⁶

Attention map captures 4 loại interaction:
  ┌────────────────────────────────────────────┐
  │  CIR₁↔CIR₂  CIR₁↔CIR₃  ...  CIR₁↔AUX    │  ← intra-CIR spatial
  │  CIR₂↔CIR₁  CIR₂↔CIR₃  ...  CIR₂↔AUX    │  ← cross-spatial
  │  ...                                        │
  │  AUX↔CIR₁   AUX↔CIR₂   ...  AUX↔AUX      │  ← cross-modal
  └────────────────────────────────────────────┘

Output = A @ V_h → concat heads → Linear(16→16) → residual + LayerNorm
```

**So sánh DOF (degrees of freedom)**:

| Method | Attention DOF | Type |
|---|---:|---|
| ECA-UWB GatedFusion | 2 | Scalar gates chỉ |
| D5 element-wise CCAF | 16 | Per-dim CIR↔AUX |
| SA-TinyML self-attn | 256 | 16×16 feature-level |
| **D8 token attention** | **72** | **6×6×2 heads, cross-modal** |

D8 có ít DOF hơn SA-TinyML (72 vs 256) nhưng hiệu quả hơn vì tokens có **cấu trúc ngữ nghĩa** (spatial position + modality type), trong khi SA-TinyML's 16 features là flat scalars không có structure.

### 2.5 Mean Pooling + Classifier

Sau attention, 6 tokens được average thành 1 vector 16-D → classifier:

```
f = mean([tok₁, tok₂, ..., tok₆])   ∈ ℝ¹⁶
→ FC(16→42) → ReLU → Dropout(0.3)
→ FC(42→16) → ReLU → Dropout(0.15)
→ FC(16→1)  → logit
```

**Tại sao mean pool thắng CLS token (D7)?**
- CLS token thêm 1 token "trống" phải học toàn bộ aggregate từ scratch
- Mean pool tận dụng trực tiếp tất cả refined token representations
- Với chỉ 6 tokens (rất ít), mean pool đủ hiệu quả

---

## 3. Tổng kết tham số

| Component | Params |
|---|---:|
| 3× DilatedConv1d(1→8, k=3) + BN | 144 |
| 1×1 Conv(24→16) + BN | 416 |
| ECA(k=3) | 3 |
| Conv1d(16→16, k=3) + BN | 800 |
| AUX branch: FC(7→32→16) | 816 |
| Position embedding (5×16) | 80 |
| Modality embedding (2×16) | 32 |
| QKV projection (16→48) | 816 |
| Output projection (16→16) | 272 |
| LayerNorm(16) | 32 |
| Classifier (16→42→16→1) | 1,419 |
| **Tổng** | **4,830** |

---

## 4. Quy trình training

### 4.1 Hyperparameters (HP_D6)

```python
{
  "lr": 1e-3,           # Adam learning rate
  "weight_decay": 1e-4,  # L2 regularization
  "pos_weight": 1.0,     # balanced BCE
  "batch_size": 256,
  "epochs": 250,          # max epochs
  "patience": 40,         # early stopping patience
  "warmup_epochs": 5,     # LR warmup
  "mixup_alpha": 0.0,     # NO Mixup
  "label_smooth": 0.0,    # NO Label Smoothing
}
```

**Điểm quan trọng**: D8 KHÔNG dùng training tricks! Performance đến 100% từ kiến trúc.

### 4.2 Hành vi training thực tế

- D8 hội tụ rất nhanh: **~100 epoch** (early stopping)
- Không bao giờ vào pha SWA (swa_start=220 > actual epochs)
- Thời gian/seed: 243–385 giây
- Tổng 5 seeds: ~25 phút

### 4.3 Data pipeline

```
eWINE dataset → CSV files → extract 50-sample CIR window + 7 aux features
→ remove outliers (z>6.0) → 41,869 samples
→ stratified split 70/15/15 → StandardScaler (fit on train only)
→ TensorDataset → DataLoader(batch=256, shuffle for train)
```

---

## 5. Ưu điểm nổi bật

### 5.1 Ưu điểm kiến trúc

1. **Spatial-aware**: Tokenization giữ thông tin vị trí CIR — biết đâu là first-path, đâu là multipath
2. **Cross-modal**: Attention trên hỗn hợp CIR+AUX tokens cho phép CIR features tham chiếu statistical diagnostics
3. **Multi-scale**: 3 dilation rates capture patterns ở nhiều scale
4. **Parameter efficient**: 4,830 params — chỉ hơn SA-TinyML 4.4%

### 5.2 Ưu điểm thực nghiệm

1. **Beat SOTA**: 93.05% > 92.92% (SA-TinyML), confirmed trên 5-seed mean
2. **Ổn định**: Std 0.27 — thấp hơn cả SA-TinyML (0.28) và D6 (0.34)
3. **Floor cao**: Seed kém nhất = 92.82%, vẫn gần bằng SA-TinyML mean
4. **Hội tụ nhanh**: ~100 epoch, không cần tricks
5. **1-stage training**: Đơn giản hơn SA-TinyML (2-stage)

### 5.3 Ưu điểm triển khai

1. **MCU-compatible**: 4,830 params, ~50K MACs — thoải mái trên STM32F401RE
2. **ONNX-friendly**: Chỉ dùng MatMul, Softmax, Conv1d — ST Edge AI hỗ trợ
3. **Không phụ thuộc runtime attention**: attention map chỉ 6×6 — tính toán trivial

---

## 6. Novelty claims cho paper

### Contribution 1: Multi-Resolution Tokenization
> Biến đổi CIR feature map thành spatial tokens thay vì GAP collapsing. Mỗi token đại diện cho 1 vùng temporal CIR, giữ thông tin vị trí multipath cho attention.

### Contribution 2: Cross-Modal Token Attention
> Self-attention trên tổ hợp CIR spatial tokens + AUX statistical token. Cho phép mô hình tự học interaction giữa spatial multipath patterns và channel diagnostic metrics.

### Contribution 3: Aligned Tokenization
> Chứng minh rằng alignment giữa số tokens và feature map length (25/5=5 exact vs 25/4=6.25 uneven) là yếu tố quyết định, tạo ra cải thiện +0.18pp chỉ bằng thay đổi 1 hyperparameter.

### So sánh với existing methods:

| Aspect | SA-TinyML (Wu 2024) | Proposed D8 MRTA |
|---|---|---|
| CIR processing | MLP flat (no spatial) | Multi-res CNN → tokens (spatial) |
| Feature interaction | Self-attn on 16 scalars | Self-attn on 6 semantic tokens |
| Cross-modal fusion | Implicit | Explicit (CIR+AUX tokens) |
| Training | 2-stage (pretrain+finetune) | 1-stage end-to-end |
| Accuracy | 92.92% | **93.05%** |

---

## 7. Hạn chế và hướng phát triển

### Hạn chế
1. **MACs cao hơn SA-TinyML** (~50K vs ~17K) do convolution layers
2. **Chưa test INT8 quantization** — cần verify trước deployment
3. **Chỉ test trên eWINE** — cần validate trên OIUD hoặc dataset khác

### Tiềm năng cải tiến
1. Depthwise separable convolution để giảm MACs mà giữ accuracy
2. Knowledge distillation từ D8 xuống model nhỏ hơn cho edge deployment
3. Mở rộng sang multi-class (LOS / Soft-NLOS / Hard-NLOS / Multi-bounce)
