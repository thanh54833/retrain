# CPU và LLM

## 1. LLM tương tác với CPU như nào?

### Hiểu đơn giản

CPU (Central Processing Unit) là "bộ não chính" của máy tính:
- **CPU**: Giống như 1 người thợ giỏi, làm từng việc một nhưng rất nhanh và chính xác
- **GPU**: Giống như 1000 người thợ, làm nhiều việc cùng lúc nhưng mỗi người đơn giản hơn

LLM có thể chạy trên CPU khi:
- Không có GPU hoặc GPU không đủ mạnh
- Model nhỏ (< 3B parameters)
- Chỉ cần test/demo, không cần tốc độ cao
- Cần tiết kiệm chi phí

### 1.1. Tại sao CPU chậm hơn GPU cho LLM?

**Giống như:** So sánh 1 đầu bếp giỏi vs 100 đầu bếp

- **CPU (Central Processing Unit)**:
  - Có ít lõi (4-16 cores thường thấy)
  - Mỗi lõi rất mạnh, xử lý logic phức tạp tốt
  - Làm việc tuần tự (từng việc một)
  - Phù hợp: Logic phức tạp, điều khiển, xử lý tuần tự
  - Ví dụ: Tính 1 triệu phép nhân → làm từng cái một → chậm

- **GPU (Graphics Processing Unit)**:
  - Có rất nhiều lõi (hàng nghìn cores)
  - Mỗi lõi đơn giản, nhưng làm song song
  - Phù hợp: Tính toán đơn giản lặp lại nhiều lần (ma trận)
  - Ví dụ: Tính 1 triệu phép nhân → làm cùng lúc → nhanh gấp 100 lần

**Ví dụ cụ thể:**
- CPU (16 cores): Inference Llama 2 7B → 2-5 tokens/giây
- GPU (RTX 4090): Inference Llama 2 7B → 80-150 tokens/giây
- **GPU nhanh hơn CPU 20-75 lần!**

### 1.2. Quá trình LLM chạy trên CPU

**Giống như:** Một người thợ làm tất cả các công việc một cách tuần tự

**Bước 1: Tải model vào RAM (Loading)**
```
Model weights → CPU RAM
- Model được load từ disk vào RAM
- Ví dụ: Model 7B (14 GB) → mất vài giây để load
- Model "sống" trong CPU RAM
```

**Bước 2: Xử lý input (Inference)**
```
1. Input được đọc vào RAM
   - Câu hỏi được chuyển thành số
   - Lưu trong RAM

2. CPU tính toán tuần tự
   - CPU đi qua từng layer một
   - Mỗi layer: Đọc từ RAM → Tính toán → Ghi lại RAM
   - Qua 32 layers → tạo ra câu trả lời
   - Vấn đề: CPU chỉ có vài cores → không thể song song nhiều

3. Output được trả về
   - Kết quả trong RAM
   - Chuyển thành text để hiển thị
```

**Bước 3: Training (nếu có)**
```
- Forward pass: Tính toán như inference (chậm)
- Backward pass: Tính gradients (chậm hơn nữa)
- Update weights: Cập nhật model weights
- Tất cả xảy ra trong CPU RAM
- Rất chậm so với GPU!
```

### 1.3. Các thành phần CPU quan trọng

#### 1.3.1. CPU Cores - Lõi xử lý

**Giống như:** Số lượng công nhân

- **Số cores**: 
  - CPU thông thường: 4-16 cores
  - CPU server: 16-64 cores
  - CPU workstation: 8-32 cores
- **Càng nhiều cores → càng nhanh** (nhưng vẫn chậm hơn GPU nhiều)

**Ví dụ:**
- CPU 4 cores: Inference 1 token/giây
- CPU 8 cores: Inference 2 tokens/giây (gấp đôi)
- CPU 16 cores: Inference 4 tokens/giây (gấp 4 lần)
- GPU RTX 4090: Inference 100 tokens/giây (gấp 25 lần CPU 16 cores!)

#### 1.3.2. CPU Clock Speed - Tốc độ xung nhịp

**Giống như:** Tốc độ làm việc của mỗi công nhân

- **Đơn vị**: GHz (Gigahertz)
- **Ví dụ**: 
  - CPU 3.0 GHz: 3 tỷ chu kỳ/giây
  - CPU 4.0 GHz: 4 tỷ chu kỳ/giây (nhanh hơn 33%)
- **Quan trọng**: Clock speed cao → CPU nhanh hơn

**Lưu ý:**
- Clock speed không phải tất cả
- Số cores và kiến trúc cũng quan trọng
- CPU 4.0 GHz 4 cores < CPU 3.5 GHz 16 cores (cho LLM)

#### 1.3.3. CPU Cache - Bộ nhớ đệm

**Giống như:** Bàn làm việc nhỏ gần công nhân

- **L1 Cache**: Rất nhanh, rất nhỏ (32-64 KB/core)
- **L2 Cache**: Nhanh, nhỏ (256 KB - 1 MB/core)
- **L3 Cache**: Trung bình, lớn hơn (8-64 MB chung)
- **Chức năng**: Lưu dữ liệu thường dùng → truy cập nhanh hơn RAM

**Tại sao quan trọng?**
- RAM chậm hơn cache 10-100 lần
- Cache lớn → ít phải đọc RAM → nhanh hơn
- LLM cần đọc nhiều dữ liệu → cache lớn giúp đáng kể

#### 1.3.4. RAM Support - Hỗ trợ bộ nhớ

**Giống như:** Kho chứa đồ lớn

- **DDR4**: Tốc độ 2400-3200 MT/s
- **DDR5**: Tốc độ 4800-6400 MT/s (nhanh hơn 2x)
- **Bandwidth**: 
  - DDR4: ~50 GB/s
  - DDR5: ~100 GB/s
- **Quan trọng**: LLM cần đọc/ghi RAM liên tục → bandwidth cao giúp đáng kể

#### 1.3.5. AVX/AVX2/AVX-512 - SIMD Instructions

**Giống như:** Công cụ đặc biệt để làm nhiều việc cùng lúc

- **SIMD**: Single Instruction, Multiple Data
- **Cách hoạt động**: 1 lệnh tính nhiều số cùng lúc
- **Ví dụ**: 
  - Bình thường: Tính 8 số → 8 lệnh
  - AVX-512: Tính 8 số → 1 lệnh (nhanh 8x!)
- **Quan trọng**: LLM tính nhiều ma trận → SIMD giúp tăng tốc đáng kể

**Hỗ trợ:**
- AVX: Hầu hết CPU hiện đại
- AVX2: CPU từ 2013+
- AVX-512: Chỉ một số CPU (Intel Xeon, một số Core i9)

## 2. So sánh CPU vs GPU cho LLM

### 2.1. Tốc độ xử lý

**Ví dụ thực tế: Inference Llama 2 7B**

| Thiết bị | Tốc độ (tokens/giây) | Thời gian trả lời (100 tokens) |
|----------|---------------------|--------------------------------|
| CPU 4 cores | 0.5-1 tok/s | 100-200 giây |
| CPU 8 cores | 1-2 tok/s | 50-100 giây |
| CPU 16 cores | 2-5 tok/s | 20-50 giây |
| RTX 3090 (24GB) | 50-100 tok/s | 1-2 giây |
| RTX 4090 (24GB) | 80-150 tok/s | 0.7-1.3 giây |

**Kết luận:** GPU nhanh hơn CPU **20-300 lần**!

### 2.2. Bộ nhớ

| Thiết bị | Loại bộ nhớ | Dung lượng | Tốc độ | Chi phí |
|----------|-------------|------------|--------|---------|
| CPU | RAM (DDR4/DDR5) | 16-128 GB | 50-100 GB/s | Rẻ |
| GPU | VRAM (GDDR6X/HBM) | 8-80 GB | 500-2000 GB/s | Đắt |

**Kết luận:** 
- CPU RAM: Nhiều hơn, rẻ hơn, nhưng chậm hơn 10-20 lần
- GPU VRAM: Ít hơn, đắt hơn, nhưng nhanh hơn nhiều

### 2.3. Khi nào nên dùng CPU?

**Dùng CPU khi:**

1. **Không có GPU hoặc GPU quá yếu**
   - Không đủ tiền mua GPU
   - GPU cũ không hỗ trợ CUDA tốt
   - Chỉ có integrated graphics

2. **Model nhỏ (< 3B parameters)**
   - Model nhỏ chạy được trên CPU
   - Tốc độ chấp nhận được (5-10 tokens/giây)
   - Ví dụ: GPT-2 small, DistilBERT

3. **Chỉ cần test/demo**
   - Không cần tốc độ cao
   - Chỉ test tính năng
   - Development và debugging

4. **Tiết kiệm chi phí**
   - GPU tốn điện nhiều (300-450W)
   - CPU tiết kiệm hơn (65-150W)
   - Không cần đầu tư GPU đắt tiền

5. **Batch size nhỏ hoặc single request**
   - Chỉ xử lý 1 request tại một thời điểm
   - Không cần throughput cao
   - Personal use, không phải production

**Không nên dùng CPU khi:**

1. **Model lớn (> 7B parameters)**
   - Quá chậm (1-2 tokens/giây)
   - Trải nghiệm người dùng kém

2. **Production environment**
   - Cần tốc độ cao
   - Nhiều requests đồng thời
   - Latency thấp

3. **Training model**
   - Training trên CPU cực kỳ chậm
   - Có thể mất vài tuần thay vì vài giờ

4. **Batch inference**
   - Xử lý nhiều requests cùng lúc
   - Cần throughput cao

## 3. Các loại CPU phù hợp cho LLM

### 3.1. CPU Consumer (Desktop/Laptop) - Giá rẻ

**Intel Core i7/i9 (12th gen trở lên):**
- Cores: 8-16 cores
- Clock: 3.0-5.0 GHz
- RAM: Hỗ trợ DDR4/DDR5
- Phù hợp: Model < 3B, inference nhẹ
- Giá: $300-600
- Ưu điểm: Giá tốt, đủ cho model nhỏ
- Nhược điểm: Chậm, không phù hợp model lớn

**AMD Ryzen 7/9 (5000 series trở lên):**
- Cores: 8-16 cores
- Clock: 3.5-5.0 GHz
- RAM: Hỗ trợ DDR4/DDR5
- Phù hợp: Model < 3B, inference nhẹ
- Giá: $300-600
- Ưu điểm: Nhiều cores, giá tốt
- Nhược điểm: Chậm hơn GPU nhiều

### 3.2. CPU Workstation - Chuyên nghiệp

**Intel Xeon W-series:**
- Cores: 16-56 cores
- Clock: 2.5-4.5 GHz
- RAM: Hỗ trợ DDR4/DDR5, ECC memory
- Phù hợp: Model 3B-7B, inference vừa phải
- Giá: $1,000-3,000
- Ưu điểm: Nhiều cores, ECC memory, ổn định
- Nhược điểm: Đắt, vẫn chậm hơn GPU

**AMD Threadripper PRO:**
- Cores: 16-64 cores
- Clock: 3.0-4.5 GHz
- RAM: Hỗ trợ DDR4, ECC memory
- Phù hợp: Model 3B-7B, inference vừa phải
- Giá: $1,500-4,000
- Ưu điểm: Rất nhiều cores, mạnh
- Nhược điểm: Đắt, vẫn chậm hơn GPU

### 3.3. CPU Server - Data Center

**Intel Xeon Scalable:**
- Cores: 16-64 cores
- Clock: 2.0-4.0 GHz
- RAM: Hỗ trợ DDR4/DDR5, ECC memory
- Phù hợp: Model lớn, multi-CPU setup
- Giá: $2,000-10,000
- Ưu điểm: Rất nhiều cores, ECC, ổn định
- Nhược điểm: Rất đắt, vẫn chậm hơn GPU

**AMD EPYC:**
- Cores: 16-128 cores
- Clock: 2.0-4.0 GHz
- RAM: Hỗ trợ DDR4, ECC memory
- Phù hợp: Model lớn, multi-CPU setup
- Giá: $2,000-8,000
- Ưu điểm: Cực nhiều cores, mạnh nhất
- Nhược điểm: Rất đắt, vẫn chậm hơn GPU

### 3.4. Bảng so sánh CPU

| CPU | Cores | Clock | Phù hợp Model | Tốc độ Inference | Giá |
|-----|-------|-------|--------------|------------------|-----|
| Core i7-12700 | 12 | 2.1-4.9 GHz | < 3B | 1-2 tok/s | $300 |
| Ryzen 7 5800X | 8 | 3.8-4.7 GHz | < 3B | 1-2 tok/s | $350 |
| Core i9-13900K | 24 | 3.0-5.8 GHz | < 7B | 2-4 tok/s | $600 |
| Xeon W-2295 | 18 | 3.0-4.6 GHz | < 7B | 3-5 tok/s | $1,500 |
| Threadripper PRO 5995WX | 64 | 2.7-4.5 GHz | < 7B | 5-10 tok/s | $6,500 |

**Lưu ý:** Tất cả CPU đều chậm hơn GPU rất nhiều!

## 4. Tối ưu hóa CPU cho LLM

### 4.1. Sử dụng SIMD Instructions

**Giống như:** Dùng công cụ đặc biệt để làm nhanh hơn

- **AVX/AVX2/AVX-512**: Tính nhiều số cùng lúc
- **Tăng tốc**: 2-8x cho các phép tính ma trận
- **Cách kích hoạt**: 
  - PyTorch tự động dùng nếu CPU hỗ trợ
  - Hoặc compile với flags: `-mavx2` hoặc `-mavx512f`

**Ví dụ:**
- Không AVX: Inference 1 token/giây
- Có AVX-512: Inference 3-4 tokens/giây (nhanh 3-4x!)

### 4.2. Tăng số threads

**Giống như:** Phân công nhiều công nhân hơn

- **Mặc định**: PyTorch dùng tất cả cores
- **Có thể điều chỉnh**: 
  ```python
  import torch
  torch.set_num_threads(16)  # Dùng 16 threads
  ```
- **Tối ưu**: Số threads = số cores CPU

**Ví dụ:**
- 4 threads: Inference 1 token/giây
- 8 threads: Inference 1.8 tokens/giây
- 16 threads: Inference 3 tokens/giây

### 4.3. Quantization (Giảm độ chính xác)

**Giống như:** Dùng đơn vị nhỏ hơn để tính nhanh hơn

- **FP32 → FP16**: Giảm memory 50%, tăng tốc 1.5-2x
- **FP32 → INT8**: Giảm memory 75%, tăng tốc 2-3x
- **Trade-off**: Có thể giảm accuracy nhẹ

**Ví dụ:**
- FP32: Inference 1 token/giây
- FP16: Inference 1.8 tokens/giây
- INT8: Inference 2.5 tokens/giây

### 4.4. Batch Processing

**Giống như:** Xử lý nhiều việc cùng lúc

- **Batch size > 1**: Tận dụng CPU tốt hơn
- **Tối ưu**: Batch size = 4-8 (tùy CPU)
- **Lưu ý**: Cần nhiều RAM hơn

**Ví dụ:**
- Batch 1: Inference 1 token/giây
- Batch 4: Inference 3 tokens/giây (tăng 3x throughput)
- Batch 8: Inference 5 tokens/giây (tăng 5x throughput)

### 4.5. Model Optimization

**Giống như:** Tối ưu code để chạy nhanh hơn

- **ONNX Runtime**: Tối ưu cho CPU, nhanh hơn PyTorch 1.5-2x
- **OpenVINO**: Tối ưu Intel CPU, nhanh hơn 2-3x
- **TensorRT**: Tối ưu NVIDIA (nhưng cần GPU)

**Ví dụ:**
- PyTorch: Inference 1 token/giây
- ONNX Runtime: Inference 1.8 tokens/giây
- OpenVINO: Inference 2.5 tokens/giây

### 4.6. RAM Optimization

**Giống như:** Tổ chức kho chứa tốt hơn

- **DDR5 thay vì DDR4**: Nhanh hơn 2x
- **Dual channel/Quad channel**: Tăng bandwidth
- **Đủ RAM**: Tránh swap (rất chậm)

**Ví dụ:**
- DDR4 single channel: Inference 1 token/giây
- DDR4 dual channel: Inference 1.3 tokens/giây
- DDR5 dual channel: Inference 1.8 tokens/giây

## 5. Hạn chế của CPU cho LLM

### 5.1. Tốc độ chậm

**Vấn đề:**
- CPU chậm hơn GPU 20-300 lần
- Inference mất vài chục giây thay vì vài giây
- Trải nghiệm người dùng kém

**Ví dụ:**
- CPU: Trả lời câu hỏi 100 tokens → 20-50 giây
- GPU: Trả lời câu hỏi 100 tokens → 0.7-1.3 giây

### 5.2. Không phù hợp model lớn

**Vấn đề:**
- Model > 7B: Quá chậm (1-2 tokens/giây)
- Model > 13B: Gần như không dùng được
- Model > 70B: Không thể chạy

**Ví dụ:**
- Llama 2 7B trên CPU: 2-5 tokens/giây (chấp nhận được)
- Llama 2 13B trên CPU: 1-2 tokens/giây (quá chậm)
- Llama 2 70B trên CPU: Không thể chạy (cần quá nhiều RAM)

### 5.3. Training cực kỳ chậm

**Vấn đề:**
- Training trên CPU chậm hơn GPU 100-1000 lần
- Có thể mất vài tuần thay vì vài giờ
- Không thực tế cho production

**Ví dụ:**
- Training Llama 2 7B trên CPU: ~2-4 tuần
- Training Llama 2 7B trên GPU: ~4-8 giờ

### 5.4. Không tận dụng được Tensor Cores

**Vấn đề:**
- CPU không có Tensor Cores
- Không thể tận dụng mixed precision tốt
- Không có hardware acceleration cho ma trận

### 5.5. Memory Bandwidth thấp

**Vấn đề:**
- CPU RAM bandwidth: 50-100 GB/s
- GPU VRAM bandwidth: 500-2000 GB/s
- CPU phải chờ đọc dữ liệu từ RAM → chậm

## 6. Khi nào CPU là lựa chọn tốt?

### 6.1. Development và Testing

**Phù hợp khi:**
- Đang phát triển code
- Test tính năng mới
- Debug model
- Không cần tốc độ cao

**Lý do:**
- CPU có sẵn, không cần GPU
- Đủ để test
- Tiết kiệm chi phí

### 6.2. Model nhỏ (< 3B)

**Phù hợp khi:**
- Model nhỏ như GPT-2 small, DistilBERT
- Tốc độ chấp nhận được (5-10 tokens/giây)
- Personal use

**Lý do:**
- Model nhỏ chạy được trên CPU
- Không cần đầu tư GPU
- Đủ cho nhu cầu cá nhân

### 6.3. Edge Devices

**Phù hợp khi:**
- Chạy trên mobile, IoT devices
- Không có GPU
- Model đã được optimize (quantized)

**Lý do:**
- Edge devices thường không có GPU
- Model nhỏ + quantization → chạy được
- Ví dụ: Smartphone, Raspberry Pi

### 6.4. Cost-sensitive Applications

**Phù hợp khi:**
- Ngân sách hạn chế
- Không cần tốc độ cao
- Low traffic

**Lý do:**
- CPU rẻ hơn GPU nhiều
- Không tốn điện nhiều
- Đủ cho use case đơn giản

## 7. Best Practices khi dùng CPU

### 7.1. Chọn CPU phù hợp

- **Nhiều cores**: Ưu tiên cores hơn clock speed
- **Hỗ trợ AVX-512**: Tăng tốc đáng kể
- **DDR5 RAM**: Nhanh hơn DDR4
- **Đủ RAM**: Tránh swap (rất chậm)

### 7.2. Tối ưu code

- **Dùng ONNX Runtime hoặc OpenVINO**: Nhanh hơn PyTorch
- **Quantization**: Giảm memory, tăng tốc
- **Batch processing**: Tận dụng CPU tốt hơn
- **Multi-threading**: Dùng tất cả cores

### 7.3. Quản lý tài nguyên

- **Đóng các ứng dụng khác**: Giải phóng CPU và RAM
- **Monitor CPU usage**: Đảm bảo sử dụng tối đa
- **Tránh swap**: Đảm bảo đủ RAM
- **Cooling**: CPU nóng → giảm tốc độ

### 7.4. Kỳ vọng thực tế

- **Không mong đợi tốc độ cao**: CPU chậm hơn GPU nhiều
- **Chỉ dùng cho model nhỏ**: Model lớn không thực tế
- **Không dùng cho training**: Quá chậm
- **Chấp nhận latency cao**: 20-50 giây cho câu trả lời là bình thường

## 8. Kết luận

### 8.1. Tóm tắt

- **CPU chậm hơn GPU 20-300 lần** cho LLM vì tính toán tuần tự
- **Chỉ phù hợp model nhỏ** (< 3B) hoặc development/testing
- **Không thực tế cho training** hoặc model lớn
- **Tối ưu hóa giúp** nhưng vẫn chậm hơn GPU nhiều

### 8.2. Khi nào dùng CPU?

✅ **Nên dùng khi:**
- Development và testing
- Model nhỏ (< 3B)
- Không có GPU
- Tiết kiệm chi phí
- Edge devices

❌ **Không nên dùng khi:**
- Production environment
- Model lớn (> 7B)
- Training model
- Cần tốc độ cao
- Batch inference lớn

### 8.3. Lựa chọn thay thế

**Nếu có ngân sách:**
- Mua GPU (RTX 3060/4060) - $300-600
- Nhanh hơn CPU 20-50 lần
- Đáng đầu tư nếu nghiêm túc với LLM

**Nếu không có ngân sách:**
- Dùng CPU cho model nhỏ
- Hoặc dùng cloud GPU (Colab, RunPod, etc.)
- Rẻ hơn mua GPU riêng

### 8.4. Tương lai

- **CPU đang cải thiện**: AVX-512, nhiều cores hơn
- **Nhưng vẫn chậm hơn GPU**: Kiến trúc khác nhau
- **Khuyến nghị**: Nếu nghiêm túc với LLM → đầu tư GPU

