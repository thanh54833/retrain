# SFT và PEFT: So Sánh và Ứng Dụng

## Giới Thiệu

Tùy chỉnh model cho phép bạn điều chỉnh một LLM pretrained tổng quát cho một use case hoặc domain cụ thể. Quá trình này tạo ra một model được fine-tune, vừa hưởng lợi từ dữ liệu pretraining rộng lớn, vừa cho kết quả chính xác hơn cho tác vụ downstream cụ thể. Tùy chỉnh model được thực hiện thông qua supervised fine-tuning và được chia thành hai loại phổ biến:

* **Full-Parameter Fine-Tuning** - được gọi là **Supervised Fine-Tuning (SFT)** trong NeMo
* **Parameter-Efficient Fine-Tuning (PEFT)**

## Supervised Fine-Tuning (SFT)

### Định Nghĩa

Trong SFT, **tất cả các tham số của model đều được cập nhật** để tạo ra các output được điều chỉnh cho tác vụ.

### Đặc Điểm

- ✅ **Hiệu suất tốt nhất**: Thường cho kết quả tốt nhất có thể
- ❌ **Tốn kém**: Yêu cầu cập nhật tất cả tham số
- ❌ **Yêu cầu phần cứng cao**: Cần GPU/TPU mạnh và nhiều bộ nhớ
- ❌ **Thời gian huấn luyện dài**: Do phải cập nhật toàn bộ model

### Khi Nào Sử Dụng SFT

- Khi bạn có đủ tài nguyên tính toán
- Khi hiệu suất là ưu tiên hàng đầu
- Khi model không quá lớn (< 7B parameters)
- Khi bạn cần tối ưu hóa toàn bộ model cho một tác vụ cụ thể

## Parameter-Efficient Fine-Tuning (PEFT)

### Định Nghĩa

PEFT điều chỉnh một số lượng tham số nhỏ hơn nhiều, được chèn vào model gốc ở các vị trí chiến lược. Khi fine-tuning với PEFT, **trọng số model gốc vẫn bị đóng băng (frozen)**, và chỉ các module adapter mới được huấn luyện. Kết quả là số lượng tham số có thể huấn luyện giảm đáng kể (**<< 1%**).

### Đặc Điểm

- ✅ **Tiết kiệm tài nguyên**: Giảm đáng kể chi phí tính toán
- ✅ **Yêu cầu phần cứng thấp**: Có thể chạy trên GPU consumer-grade
- ✅ **Huấn luyện nhanh**: Do chỉ cập nhật một phần nhỏ tham số
- ✅ **Hiệu suất gần tương đương**: Có thể đạt độ chính xác gần như SFT
- ✅ **Linh hoạt**: Dễ dàng chuyển đổi giữa các adapter cho các tác vụ khác nhau
- ⚠️ **Hiệu suất có thể thấp hơn một chút**: So với SFT full-parameter

### Khi Nào Sử Dụng PEFT

- Khi tài nguyên tính toán hạn chế
- Khi model quá lớn (> 7B parameters)
- Khi cần huấn luyện nhiều adapter cho nhiều tác vụ khác nhau
- Khi muốn giữ nguyên model gốc và chỉ thêm adapter
- Khi cần triển khai nhanh với chi phí thấp

## So Sánh SFT và PEFT

| Tiêu Chí | SFT | PEFT |
|----------|-----|------|
| **Số tham số trainable** | 100% | << 1% |
| **Hiệu suất** | Tốt nhất | Gần tương đương |
| **Yêu cầu bộ nhớ** | Rất cao | Thấp |
| **Yêu cầu GPU** | GPU mạnh (A100, H100) | GPU consumer (RTX 3090, 4090) |
| **Thời gian huấn luyện** | Dài | Ngắn |
| **Chi phí** | Cao | Thấp |
| **Linh hoạt** | Thấp (1 model = 1 tác vụ) | Cao (nhiều adapter) |
| **Bảo toàn model gốc** | Không | Có |

## Các Phương Pháp PEFT Được Hỗ Trợ trong NeMo

NeMo hỗ trợ **5 phương pháp PEFT** có thể sử dụng với các model transformer-based khác nhau:

### 1. LoRA (Low-Rank Adaptation)
- Phân rã ma trận trọng số thành hai ma trận hạng thấp
- Phổ biến và hiệu quả nhất
- Hỗ trợ rộng rãi cho nhiều model

### 2. QLoRA (Quantized LoRA)
- Kết hợp quantization và LoRA
- Giảm thêm yêu cầu bộ nhớ
- Lý tưởng cho model rất lớn

### 3. P-tuning
- Tối ưu hóa prompt embeddings
- Hiệu quả cho các tác vụ classification và generation

### 4. Adapters (Canonical)
- Thêm các lớp adapter nhỏ vào model
- Cấu trúc đơn giản và dễ hiểu

### 5. IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- Điều chỉnh inner activations
- Hiệu quả cho một số tác vụ cụ thể

## Hỗ Trợ Model trong NeMo

Bảng dưới đây cho thấy các phương pháp PEFT được hỗ trợ cho từng model:

| Model | SFT | LoRA | QLoRA | P-tuning | Adapters | IA3 |
|-------|-----|------|-------|----------|----------|-----|
| **GPT 3** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Nemotron** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Llama 1/2/3 & CodeLlama 2** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ChatGLM 2/3** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Falcon** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Starcoder 1/2** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Mistral** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Mixtral** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Gemma 1/2 & CodeGemma** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Space Gemma** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **T5** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

## Quyết Định: SFT hay PEFT?

### Chọn SFT Nếu:

1. **Tài nguyên dồi dào**: Bạn có quyền truy cập vào GPU mạnh (A100, H100)
2. **Hiệu suất là ưu tiên**: Cần kết quả tốt nhất có thể
3. **Model vừa phải**: Model < 7B parameters
4. **Tác vụ quan trọng**: Ứng dụng production-critical
5. **Một tác vụ duy nhất**: Chỉ cần fine-tune cho một tác vụ cụ thể

### Chọn PEFT Nếu:

1. **Tài nguyên hạn chế**: GPU consumer-grade hoặc tài nguyên cloud hạn chế
2. **Model lớn**: Model > 7B parameters
3. **Nhiều tác vụ**: Cần nhiều adapter cho nhiều tác vụ khác nhau
4. **Prototype nhanh**: Cần thử nghiệm và triển khai nhanh
5. **Bảo toàn model gốc**: Muốn giữ nguyên model base cho nhiều mục đích
6. **Chi phí thấp**: Cần giảm chi phí huấn luyện và lưu trữ

## Khuyến Nghị Thực Tế

### Workflow Đề Xuất

1. **Bắt đầu với PEFT (LoRA)**:
   - Nhanh, rẻ, dễ thử nghiệm
   - Đánh giá hiệu suất ban đầu
   - Nếu đạt yêu cầu → dừng lại

2. **Nâng cấp lên SFT nếu cần**:
   - Nếu PEFT không đạt hiệu suất mong muốn
   - Khi có đủ tài nguyên
   - Khi cần tối ưu hóa cuối cùng

3. **Kết hợp khi có thể**:
   - Sử dụng PEFT cho nhiều tác vụ khác nhau
   - Sử dụng SFT cho tác vụ quan trọng nhất

### Best Practices

#### Cho PEFT:
- Bắt đầu với `r=16` hoặc `r=32` cho LoRA
- Thử nghiệm với các target modules khác nhau
- Sử dụng QLoRA nếu model quá lớn
- Lưu trữ nhiều adapter cho nhiều tác vụ

#### Cho SFT:
- Sử dụng learning rate thấp hơn (1e-5 đến 5e-5)
- Sử dụng gradient checkpointing để tiết kiệm bộ nhớ
- Monitor cả training và validation loss
- Sử dụng early stopping để tránh overfitting

## Tài Liệu Tham Khảo

- [NVIDIA NeMo Framework - SFT and PEFT](https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/index.html)
- [NeMo Developer Quick Start](https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/developer_quick_start.html)
- [Supported PEFT Methods](https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/supported_peft_methods.html)
- [NeMo QLoRA Guide](https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/nemo_qlora_guide.html)

## Kết Luận

Cả SFT và PEFT đều có vai trò quan trọng trong việc tùy chỉnh LLM:

- **SFT** phù hợp khi bạn cần hiệu suất tối đa và có đủ tài nguyên
- **PEFT** phù hợp khi bạn cần giải pháp hiệu quả về chi phí và linh hoạt

Với sự phát triển của các model ngày càng lớn, PEFT đang trở nên phổ biến hơn do yêu cầu phần cứng nhẹ nhàng. Tuy nhiên, SFT vẫn là lựa chọn tốt nhất khi hiệu suất là ưu tiên hàng đầu và tài nguyên không phải là vấn đề.

Việc lựa chọn giữa SFT và PEFT phụ thuộc vào:
- Tài nguyên tính toán có sẵn
- Yêu cầu hiệu suất
- Kích thước model
- Số lượng tác vụ cần hỗ trợ
- Ngân sách và thời gian

