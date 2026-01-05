# GPU và LLM

## 1. LLM tương tác với GPU như nào?

### Hiểu đơn giản

GPU (Graphics Processing Unit) giống như một "siêu máy tính nhỏ" chuyên xử lý song song:
- **CPU**: Giống như 1 người thợ giỏi, làm từng việc một nhưng rất nhanh
- **GPU**: Giống như 1000 người thợ, làm nhiều việc cùng lúc (song song)

LLM cần GPU vì:
- Có hàng tỷ phép tính cần làm cùng lúc
- GPU có thể tính hàng nghìn phép tính song song
- Nhanh hơn CPU hàng chục đến hàng trăm lần

### 1.1. Tại sao GPU nhanh hơn CPU cho LLM?

**Giống như:** So sánh 1 đầu bếp vs 100 đầu bếp

- **CPU (Central Processing Unit)**:
  - Có ít lõi (4-16 cores)
  - Mỗi lõi rất mạnh, làm việc tuần tự
  - Phù hợp: Logic phức tạp, điều khiển
  - Ví dụ: Tính 1 triệu phép nhân → làm từng cái một → chậm

- **GPU (Graphics Processing Unit)**:
  - Có rất nhiều lõi (hàng nghìn cores)
  - Mỗi lõi đơn giản, nhưng làm song song
  - Phù hợp: Tính toán đơn giản lặp lại nhiều lần
  - Ví dụ: Tính 1 triệu phép nhân → làm cùng lúc → nhanh gấp 100 lần

**Ví dụ cụ thể:**
- CPU: Tính ma trận 1000×1000 → ~0.1 giây
- GPU: Tính ma trận 1000×1000 → ~0.001 giây (nhanh 100x!)

### 1.2. Quá trình LLM chạy trên GPU

**Giống như:** Dây chuyền sản xuất trong nhà máy

**Bước 1: Tải model vào GPU (Loading)**
```
CPU RAM → GPU VRAM
- Model weights được copy từ RAM sang VRAM
- Ví dụ: Model 14 GB → mất vài giây để copy
- Sau đó model "sống" trong GPU VRAM
```

**Bước 2: Xử lý input (Inference)**
```
1. Input (CPU RAM) → GPU VRAM
   - Câu hỏi được chuyển thành số
   - Copy vào GPU để xử lý

2. Tính toán trong GPU
   - GPU thực hiện hàng nghìn phép tính song song
   - Mỗi layer tính toán → kết quả lưu trong VRAM
   - Qua 32 layers → tạo ra câu trả lời

3. Output (GPU VRAM) → CPU RAM
   - Kết quả được copy về RAM
   - Chuyển thành text để hiển thị
```

**Bước 3: Training (nếu có)**
```
- Forward pass: Tính toán như inference
- Backward pass: Tính gradients (ngược lại)
- Update weights: Cập nhật model weights
- Tất cả xảy ra trong GPU VRAM
```

### 1.3. Các thành phần GPU quan trọng

#### 1.3.1. VRAM (Video RAM) - Bộ nhớ GPU

**Giống như:** Kho chứa đồ của GPU

- **Chức năng**: Lưu model weights, activations, gradients
- **Đặc điểm**: 
  - Rất nhanh (nhanh hơn CPU RAM 10-20x)
  - Ít hơn (8GB, 16GB, 24GB, 40GB, 80GB...)
  - Kết nối trực tiếp với GPU cores → truy cập cực nhanh

- **Ví dụ VRAM cần thiết:**
  - Model 7B inference: ~15 GB VRAM
  - Model 7B training: ~60 GB VRAM
  - Model 13B inference: ~27 GB VRAM
  - Model 70B inference: ~140 GB VRAM (cần nhiều GPU!)

**Memory Bandwidth (Băng thông bộ nhớ):**
- Tốc độ đọc/ghi dữ liệu từ VRAM
- Ví dụ: RTX 4090 có 1008 GB/s → đọc 1 GB chỉ mất 0.001 giây
- Quan trọng: Bandwidth cao → GPU nhanh hơn

#### 1.3.2. CUDA Cores - Lõi tính toán

**Giống như:** Công nhân trong nhà máy

- **Chức năng**: Thực hiện các phép tính cơ bản (+, -, ×, ÷)
- **Số lượng**: 
  - RTX 4090: 16,384 cores
  - A100: 6,912 cores
  - RTX 3090: 10,496 cores
- **Càng nhiều cores → càng nhanh**

**Ví dụ:**
- 1 core: Tính 1 phép nhân/giây
- 16,384 cores: Tính 16,384 phép nhân cùng lúc → nhanh gấp 16,384 lần!

#### 1.3.3. Tensor Cores - Lõi tính ma trận chuyên dụng

**Giống như:** Máy chuyên dụng tính ma trận

- **Chức năng**: Tính toán ma trận cực nhanh (nhân ma trận)
- **Đặc biệt**: 
  - Nhanh hơn CUDA cores 4-8 lần cho ma trận
  - Hỗ trợ mixed precision (FP16, BF16, INT8)
  - LLM chủ yếu tính ma trận → Tensor Cores rất quan trọng!

**Ví dụ:**
- CUDA cores: Nhân ma trận 1000×1000 → 0.001 giây
- Tensor Cores: Nhân ma trận 1000×1000 → 0.0001 giây (nhanh 10x!)

#### 1.3.4. Memory Bandwidth - Băng thông bộ nhớ

**Giống như:** Đường cao tốc dẫn vào kho

- **Chức năng**: Tốc độ đọc/ghi dữ liệu từ VRAM
- **Đơn vị**: GB/s (Gigabytes per second)
- **Ví dụ:**
  - RTX 4090: 1008 GB/s
  - A100: 1935 GB/s
  - RTX 3090: 936 GB/s

**Tại sao quan trọng?**
- GPU tính toán rất nhanh → cần đọc dữ liệu nhanh
- Nếu bandwidth thấp → GPU phải chờ dữ liệu → chậm
- Giống như có xe nhanh nhưng đường tắc → không đi nhanh được

## 2. So sánh GPU vs CPU cho LLM

### 2.1. Tốc độ xử lý

**Ví dụ thực tế: Inference Llama 2 7B**

| Thiết bị | Tốc độ (tokens/giây) | Thời gian trả lời |
|----------|---------------------|-------------------|
| CPU (16 cores) | 2-5 tokens/s | ~20-50 giây |
| RTX 3090 (24GB) | 50-100 tokens/s | ~1-2 giây |
| RTX 4090 (24GB) | 80-150 tokens/s | ~0.7-1.3 giây |
| A100 (40GB) | 100-200 tokens/s | ~0.5-1 giây |

**Kết luận:** GPU nhanh hơn CPU **20-100 lần**!

### 2.2. Bộ nhớ

| Thiết bị | Loại bộ nhớ | Dung lượng | Tốc độ |
|----------|-------------|------------|--------|
| CPU | RAM (DDR4/DDR5) | 16-128 GB | 50-100 GB/s |
| GPU | VRAM (GDDR6X/HBM) | 8-80 GB | 500-2000 GB/s |

**Kết luận:** 
- CPU RAM: Nhiều hơn nhưng chậm hơn
- GPU VRAM: Ít hơn nhưng nhanh hơn 10-20 lần

### 2.3. Khi nào dùng CPU? Khi nào dùng GPU?

**Dùng CPU khi:**
- Model nhỏ (< 1B parameters)
- Không có GPU hoặc GPU quá yếu
- Chỉ cần test/demo đơn giản
- Không quan trọng tốc độ

**Dùng GPU khi:**
- Model lớn (> 3B parameters)
- Cần tốc độ cao (production)
- Training model
- Batch inference nhiều requests

## 3. Các loại GPU phù hợp cho LLM

### 3.1. GPU Consumer (Gaming) - Giá rẻ

**RTX 3090 / RTX 4090:**
- VRAM: 24 GB
- Phù hợp: Model 7B-13B inference, training nhỏ
- Giá: $1,500 - $2,000
- Ưu điểm: Giá tốt, đủ cho hầu hết use cases
- Nhược điểm: Không có ECC memory, không tối ưu cho training lớn

**RTX 3060 / RTX 4060:**
- VRAM: 12 GB (RTX 3060) / 16 GB (RTX 4060 Ti 16GB)
- Phù hợp: Model 3B-7B inference (với quantization có thể chạy 7B)
- Giá: $300 - $600
- Ưu điểm: Rất rẻ, đủ cho model nhỏ, tiết kiệm điện
- Nhược điểm: Chậm hơn, VRAM ít, không phù hợp training lớn

### 3.2. GPU Professional (Data Center) - Chuyên nghiệp

**NVIDIA A100:**
- VRAM: 40 GB hoặc 80 GB
- Phù hợp: Model lớn (13B-70B), training chuyên nghiệp
- Giá: $10,000 - $15,000
- Ưu điểm: VRAM lớn, ECC memory, tối ưu cho training
- Nhược điểm: Đắt, cần server chuyên dụng

**NVIDIA H100:**
- VRAM: 80 GB (H100 PCIe) hoặc 80 GB (H100 SXM)
- Phù hợp: Model rất lớn (70B+), training production, inference high-throughput
- Giá: $30,000 - $40,000 (có thể cao hơn tùy thị trường)
- Ưu điểm: Mạnh nhất hiện tại, Transformer Engine, FP8 support, nhanh hơn A100 2-3x
- Nhược điểm: Rất đắt, khó mua, cần server chuyên dụng

### 3.3. Bảng so sánh GPU

| GPU | VRAM | Tốc độ Inference | Phù hợp Model | Giá |
|-----|------|------------------|--------------|-----|
| RTX 3060 | 12 GB | 20-40 tok/s | 3B-7B | $300 |
| RTX 3090 | 24 GB | 50-100 tok/s | 7B-13B | $1,500 |
| RTX 4090 | 24 GB | 80-150 tok/s | 7B-13B | $2,000 |
| A100 | 40 GB | 100-200 tok/s | 13B-70B | $10,000 |
| H100 | 80 GB | 200-400 tok/s | 70B+ | $30,000 |

## 4. Công thức tính toán liên quan đến GPU

### 4.1. Tính tốc độ inference

**Công thức đơn giản:**
```
Tokens/giây = (GPU Compute Power) / (Model Size × Sequence Length)

Với:
- GPU Compute Power: TFLOPs (Tera Floating Point Operations)
- Model Size: Số tham số
- Sequence Length: Độ dài input
```

**Ví dụ:**
- RTX 4090: ~83 TFLOPs (FP16), ~330 TFLOPs (INT8)
- Model 7B, seq=2048
- Tokens/giây (lý thuyết) ≈ GPU TFLOPs / (Model FLOPs per token)
- Thực tế: ~80-150 tokens/s (do tối ưu hóa, KV cache, batching)
- Lưu ý: Tốc độ thực tế phụ thuộc vào nhiều yếu tố (sequence length, batch size, quantization)

### 4.2. Tính VRAM cần thiết

**Công thức inference:**
```
VRAM = Model Size + KV Cache + Activations

Với:
- Model Size = Parameters × 2 bytes (FP16)
- KV Cache = Batch × Seq × Hidden × Layers × 2 bytes
- Activations = Batch × Seq × Hidden × Layers × 2 bytes
```

**Ví dụ Model 7B:**
- Model: 7B × 2 bytes = 14 GB
- KV Cache (seq=2048): 1 × 2048 × 4096 × 32 × 2 = 0.5 GB
- Activations: ~0.3 GB
- **Total: ~15 GB VRAM**

### 4.3. Tính thời gian training

**Công thức:**
```
Thời gian/epoch = (Số samples × Thời gian/sample) / (Batch size × Số GPU)

Với:
- Thời gian/sample ≈ Model Size / GPU Speed
```

**Ví dụ:**
- Dataset: 10,000 samples
- Model 7B trên RTX 4090: ~0.1 giây/sample
- Batch size: 4
- Thời gian/epoch = (10,000 × 0.1) / 4 = 250 giây ≈ 4 phút

## 5. Tối ưu hóa GPU usage

### 5.1. Mixed Precision Training

**Giống như:** Dùng đơn vị nhỏ hơn để tiết kiệm

- **FP32 → FP16/BF16**: Giảm memory 50%, tăng tốc 2x
- **FP16 → INT8**: Giảm memory 50% nữa, tăng tốc 2x
- **Trade-off**: Có thể giảm accuracy nhẹ

**Ví dụ:**
- FP32: Model 7B = 28 GB VRAM
- FP16: Model 7B = 14 GB VRAM (giảm 50%)
- INT8: Model 7B = 7 GB VRAM (giảm 75%)

### 5.2. Gradient Checkpointing

**Giống như:** Chỉ lưu điểm quan trọng, tính lại phần còn lại

- **Cách hoạt động**: Không lưu tất cả activations, chỉ lưu một số điểm
- **Khi backward**: Tính lại activations từ các điểm đã lưu
- **Kết quả**: Giảm memory 50-70%, tăng thời gian 20-30%

**Ví dụ:**
- Không checkpoint: 60 GB VRAM, 4 phút/epoch
- Có checkpoint: 20 GB VRAM, 5 phút/epoch (chậm hơn 25% nhưng tiết kiệm 67% memory)

### 5.3. Batch Size Optimization

**Giống như:** Điều chỉnh số lượng công việc cùng lúc

- **Batch size lớn**: 
  - Tận dụng GPU tốt hơn
  - Nhưng cần nhiều VRAM hơn
- **Batch size nhỏ**:
  - Ít VRAM hơn
  - Nhưng GPU không được tận dụng tối đa

**Công thức tìm batch size tối ưu:**
```
Batch size tối ưu = (VRAM available - Model size) / (Activation size per sample)
```

**Ví dụ:**
- VRAM: 24 GB
- Model: 14 GB
- Activation/sample: 0.5 GB
- Batch size tối ưu = (24 - 14) / 0.5 = 20

### 5.4. Multi-GPU Training

**Giống như:** Nhiều GPU cùng làm việc

**Data Parallelism:**
- Chia batch cho nhiều GPU
- Mỗi GPU tính một phần
- Ví dụ: 4 GPU, batch 16 → mỗi GPU batch 4
- Tốc độ tăng gần tuyến tính (3.5-3.8x với 4 GPU)

**Model Parallelism:**
- Chia model cho nhiều GPU
- Mỗi GPU giữ một phần model
- Ví dụ: Model 70B → 4 GPU, mỗi GPU 17.5B
- Cần cho model quá lớn

**Pipeline Parallelism:**
- Chia layers cho nhiều GPU
- Mỗi GPU xử lý một số layers
- Ví dụ: 32 layers → 4 GPU, mỗi GPU 8 layers

### 5.5. Quantization

**Giống như:** Nén dữ liệu để tiết kiệm không gian

- **FP16 → INT8**: Giảm memory 50%, tăng tốc 2x
- **FP16 → INT4**: Giảm memory 75%, tăng tốc 4x
- **Trade-off**: Giảm accuracy (thường 1-5%)

**Ví dụ:**
- Model 7B FP16: 14 GB VRAM
- Model 7B INT8: 7 GB VRAM (chạy được trên RTX 3060!)
- Model 7B INT4: 3.5 GB VRAM (chạy được trên GPU nhỏ!)

## 6. Monitoring GPU Usage

### 6.1. Các chỉ số quan trọng

**GPU Utilization (%):**
- Phần trăm GPU đang hoạt động
- Lý tưởng: 90-100%
- Thấp (< 50%): GPU không được tận dụng → tăng batch size

**VRAM Usage (GB):**
- Bộ nhớ GPU đang dùng
- Lý tưởng: 80-95% (để lại buffer)
- 100%: Out of memory → giảm batch size hoặc dùng gradient checkpointing

**Temperature (°C):**
- Nhiệt độ GPU
- Lý tưởng: < 80°C
- > 85°C: Quá nóng → giảm tốc độ hoặc tăng cooling

**Power Consumption (W):**
- Điện năng tiêu thụ
- RTX 4090: ~450W khi full load
- A100: ~400W khi full load

### 6.2. Công cụ monitoring

**nvidia-smi:**
```bash
# Xem thông tin GPU real-time
nvidia-smi

# Xem liên tục (mỗi 1 giây)
watch -n 1 nvidia-smi
```

**Output mẫu:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx       Driver Version: 525.xx       CUDA Version: 12.0  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+=======================|
|   0  RTX 4090    Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    50W / 450W |   15234MiB / 24576MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**Giải thích:**
- Memory-Usage: 15234 MB / 24576 MB (62% VRAM đang dùng)
- GPU-Util: 0% (GPU đang idle)
- Temp: 45°C (nhiệt độ)
- Pwr: 50W / 450W (điện năng)

## 7. Kết luận

### 7.1. Tóm tắt

- **GPU nhanh hơn CPU 20-100 lần** cho LLM vì tính toán song song
- **VRAM là yếu tố quan trọng nhất** - quyết định model nào chạy được
- **Tensor Cores** làm cho GPU cực nhanh với ma trận (LLM chủ yếu tính ma trận)
- **Memory bandwidth** ảnh hưởng lớn đến tốc độ thực tế

### 7.2. Lựa chọn GPU

**Cho inference:**
- Model 3B-7B: RTX 3060/4060 (12-16 GB) - $300-600
- Model 7B-13B: RTX 3090/4090 (24 GB) - $1,500-2,000
- Model 13B-70B: A100 (40-80 GB) - $10,000-15,000

**Cho training:**
- Model nhỏ (< 7B): RTX 3090/4090 (24 GB) - có thể dùng
- Model trung bình (7B-13B): A100 (40 GB) - khuyến nghị
- Model lớn (13B+): A100/H100 (80 GB) hoặc multi-GPU - bắt buộc

### 7.3. Tối ưu hóa

1. **Mixed precision** (FP16/BF16): Giảm memory 50%, tăng tốc 2x
2. **Gradient checkpointing**: Giảm memory 50-70% cho training
3. **Quantization** (INT8/INT4): Giảm memory 50-75% cho inference
4. **Batch size optimization**: Tận dụng GPU tối đa
5. **Multi-GPU**: Tăng tốc gần tuyến tính

### 7.4. Best Practices

- **Monitor GPU usage**: Đảm bảo GPU utilization > 80%
- **Quản lý VRAM**: Không để VRAM > 95% (dễ out of memory)
- **Nhiệt độ**: Giữ GPU < 80°C để tránh thermal throttling
- **Power limit**: Có thể giảm power limit để tiết kiệm điện (giảm tốc độ nhẹ)
- **Driver/CUDA**: Luôn cập nhật driver và CUDA mới nhất

---

**Ghi chú xác minh (2024):**
- Thông số GPU (VRAM, CUDA cores, Tensor cores) đã được xác nhận từ NVIDIA official specs
- Tốc độ inference thực tế có thể thay đổi tùy vào model architecture, quantization, và optimization
- Giá GPU có thể dao động đáng kể tùy thị trường và thời điểm
- H100 có Transformer Engine và FP8 support, nhanh hơn A100 đáng kể trong training

