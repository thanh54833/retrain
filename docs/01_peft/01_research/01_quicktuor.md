# Hướng Dẫn Nhanh về LoRA

## Giới Thiệu

**LoRA (Low-Rank Adaptation)** là một phương pháp PEFT (Parameter-Efficient Fine-Tuning) phân rã ma trận trọng số lớn thành hai ma trận hạng thấp nhỏ hơn trong các lớp attention. Điều này giảm đáng kể số lượng tham số cần fine-tune, giúp tiết kiệm bộ nhớ và huấn luyện nhanh hơn.

### Lợi Ích Chính
- **Tiết Kiệm Bộ Nhớ**: Chỉ huấn luyện một phần nhỏ (thường <1%) tham số gốc
- **Huấn Luyện Nhanh**: Giảm đáng kể thời gian huấn luyện so với fine-tuning toàn bộ
- **Mô-đun Hóa**: Dễ dàng chuyển đổi giữa các adapter khác nhau cho các tác vụ khác nhau
- **Bảo Toàn Model Gốc**: Trọng số model gốc không thay đổi

## Khái Niệm Cốt Lõi

### LoRA Hoạt Động Như Thế Nào

LoRA phân rã ma trận trọng số `W` có kích thước `(d × k)` thành:
- Ma trận `A`: `(d × r)` - khởi tạo với giá trị ngẫu nhiên
- Ma trận `B`: `(r × k)` - khởi tạo bằng 0

Trọng số được điều chỉnh trở thành: `W' = W + BA`

Trong đó:
- `r` (rank) nhỏ hơn nhiều so với `d` và `k`
- Hệ số scaling là `lora_alpha / r`

## Các Tham Số LoraConfig

### Tham Số Cơ Bản

#### `r` (int)
- **Kích thước attention của LoRA** (rank)
- Giá trị thấp hơn = ít tham số hơn nhưng khả năng có thể kém hơn
- Giá trị phổ biến: 8, 16, 32, 64
- Mặc định: Phải được chỉ định

#### `target_modules` (Optional[Union[List[str], str]])
- Tên các module để áp dụng LoRA
- Có thể là:
  - Danh sách chuỗi: khớp chính xác hoặc khớp kết thúc
  - Chuỗi: khớp regex
  - `'all-linear'`: tất cả module linear/Conv1D (trừ lớp output)
  - `[]`: không có module nào (dùng với `target_parameters`)
- **Quan trọng**: Nếu không chỉ định, module sẽ được chọn tự động dựa trên kiến trúc

#### `lora_alpha` (int)
- Tham số alpha cho scaling của LoRA
- Điều khiển độ lớn của sự điều chỉnh
- Thực hành phổ biến: `lora_alpha = 2 * r` hoặc `lora_alpha = r`
- Giá trị cao hơn = điều chỉnh mạnh hơn

#### `lora_dropout` (float)
- Xác suất dropout cho các lớp LoRA
- Phạm vi: 0.0 đến 1.0
- Giá trị phổ biến: 0.05, 0.1, 0.2
- Giúp ngăn chặn overfitting

### Tham Số Nâng Cao

#### `bias` (str)
- Tùy chọn: `'none'`, `'all'`, `'lora_only'`
- `'none'`: không huấn luyện bias (mặc định, phổ biến nhất)
- `'all'`: huấn luyện tất cả bias
- `'lora_only'`: chỉ huấn luyện bias của LoRA
- **Lưu ý**: Nếu không phải 'none', output của model sẽ khác ngay cả khi adapter bị tắt

#### `use_rslora` (bool)
- **Rank-Stabilized LoRA**: Sử dụng `lora_alpha / sqrt(r)` thay vì `lora_alpha / r`
- Đã được chứng minh hoạt động tốt hơn trong một số trường hợp
- Mặc định: `False`

#### `fan_in_fan_out` (bool)
- Đặt `True` nếu lớp lưu trọng số dạng `(fan_in, fan_out)`
- Bắt buộc cho GPT-2 (sử dụng Conv1D)
- Mặc định: `False`

#### `init_lora_weights` (bool | str)
- Cách khởi tạo trọng số LoRA
- Tùy chọn:
  - `True` (mặc định): Mặc định của Microsoft (B=0, A=ngẫu nhiên) - LoRA không có tác dụng trước khi huấn luyện
  - `False`: Khởi tạo ngẫu nhiên - hữu ích cho debug
  - `'gaussian'`: Khởi tạo Gaussian được scale theo rank
  - `'eva'`: Explained Variance Adaptation (dựa trên dữ liệu, hiệu suất SOTA)
  - `'olora'`: Khởi tạo OLoRA
  - `'pissa'`: Khởi tạo PiSSA (hội tụ nhanh hơn)
  - `'corda'`: Khởi tạo CORDA
  - `'loftq'`: Khởi tạo LoftQ (cho model đã quantize)
  - `'orthogonal'`: Khởi tạo trực giao (yêu cầu `r` chẵn)

#### `exclude_modules` (Optional[Union[List[str], str]])
- Các module loại trừ khỏi điều chỉnh LoRA
- Hữu ích khi dùng `target_modules='all-linear'` nhưng muốn loại trừ các lớp cụ thể

#### `layers_to_transform` (Union[List[int], int])
- Chỉ áp dụng LoRA cho các chỉ số lớp cụ thể
- Hữu ích cho các thí nghiệm fine-tuning theo từng lớp

#### `rank_pattern` (dict)
- Rank khác nhau cho các lớp khác nhau
- Ví dụ: `{'^model.layers.0': 32, '^model.layers.1': 16}`

#### `alpha_pattern` (dict)
- Alpha khác nhau cho các lớp khác nhau
- Ví dụ: `{'^model.layers.0.attention': 64}`

## Ví Dụ Sử Dụng Cơ Bản

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load model gốc
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# Cấu hình LoRA
lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Hệ số scaling
    target_modules=["c_attn", "c_proj"],  # Module đích
    lora_dropout=0.1,              # Dropout
    bias="none",                   # Huấn luyện bias
    task_type="CAUSAL_LM"          # Loại tác vụ
)

# Áp dụng LoRA vào model
model = get_peft_model(model, lora_config)

# Kiểm tra tham số có thể huấn luyện
model.print_trainable_parameters()
# Output: trainable params: 0.1M || all params: 117M || trainable%: 0.09
```

## Target Modules Phổ Biến Theo Model

### GPT-2 / DialoGPT
```python
target_modules=["c_attn", "c_proj"]
fan_in_fan_out=True
```

### LLaMA / Mistral
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
# Hoặc cho các lớp MLP:
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### BERT
```python
target_modules=["query", "key", "value", "dense"]
```

### T5
```python
target_modules=["q", "k", "v", "o"]
```

## Phương Pháp Khởi Tạo Nâng Cao

### EVA (Explained Variance Adaptation)
Khởi tạo dựa trên dữ liệu đạt hiệu suất SOTA:

```python
from peft import LoraConfig, get_peft_model, initialize_lora_eva_weights
from peft.tuners.lora import EvaConfig

# Cấu hình với EVA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="eva",
    eva_config=EvaConfig(
        rho=2.0,              # Hệ số nhân rank tối đa
        tau=0.99,              # Ngưỡng cosine similarity
        use_label_mask=True,   # Sử dụng label mask
        label_mask_value=-100 # Giá trị mask
    )
)

model = get_peft_model(model, lora_config)

# Khởi tạo với dữ liệu
initialize_lora_eva_weights(
    model=model,
    dataloader=train_dataloader,
    adapter_name="default"
)
```

### PiSSA (Principal Singular values and Singular vectors Adaptation)
Hội tụ nhanh hơn LoRA:

```python
lora_config = LoraConfig(
    r=16,
    init_lora_weights="pissa"  # Hoặc "pissa_niter_16" cho SVD nhanh hơn
)
```

### LoftQ (cho Model Đã Quantize)
Giảm lỗi quantization:

```python
from peft import LoraConfig, get_peft_model, LoftQConfig

lora_config = LoraConfig(
    r=16,
    init_lora_weights="loftq",
    loftq_config=LoftQConfig(
        loftq_bits=4,  # Số bit quantization
        loftq_iter=1   # Số lần lặp
    )
)
```

## Lưu và Tải

### Lưu LoRA Adapter
```python
# Chỉ lưu adapter (file nhỏ)
model.save_pretrained("./lora_adapter")
```

### Tải LoRA Adapter
```python
from peft import PeftModel

# Tải model gốc
base_model = AutoModelForCausalLM.from_pretrained("base_model_name")

# Tải adapter
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```

### Merge và Unload
```python
# Merge trọng số adapter vào model gốc
merged_model = model.merge_and_unload()

# Lưu model đã merge
merged_model.save_pretrained("./merged_model")
```

## Thực Hành Tốt Nhất

### 1. Chọn Rank
- Bắt đầu với `r=16` hoặc `r=32` cho hầu hết các tác vụ
- Với tác vụ đơn giản: `r=8` có thể đủ
- Với tác vụ phức tạp: thử `r=64` hoặc cao hơn
- Sử dụng `rank_pattern` để tuning theo từng lớp

### 2. Chọn Alpha
- Tỷ lệ phổ biến: `lora_alpha = 2 * r`
- Alpha cao hơn = điều chỉnh mạnh hơn
- Có thể dùng `alpha_pattern` để kiểm soát chi tiết

### 3. Target Modules
- Với causal LM: nhắm vào các lớp attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- Để hiệu suất tốt hơn: cũng bao gồm các lớp MLP
- Sử dụng `exclude_modules` để tránh các lớp không mong muốn

### 4. Dropout
- Bắt đầu với `lora_dropout=0.1`
- Tăng nếu overfitting (0.2-0.3)
- Giảm hoặc đặt 0 nếu underfitting

### 5. Khởi Tạo
- Dùng mặc định (`True`) cho hầu hết trường hợp
- Dùng `'eva'` để có hiệu suất tốt nhất (cần dữ liệu)
- Dùng `'pissa'` để hội tụ nhanh hơn
- Dùng `'loftq'` cho model đã quantize

### 6. Mẹo Huấn Luyện
- Learning rate thấp hơn so với fine-tuning toàn bộ (1e-4 đến 5e-4)
- Có thể dùng batch size lớn hơn (ít tham số cần cập nhật)
- Theo dõi cả training và validation loss
- Sử dụng gradient checkpointing để tiết kiệm bộ nhớ

## Xử Lý Sự Cố

### Kiến Trúc Model Không Được Nhận Diện
```python
# Chỉ định target modules thủ công
lora_config = LoraConfig(
    r=16,
    target_modules=["your", "module", "names"]  # Chỉ định rõ ràng
)
```

### Hết Bộ Nhớ
- Giảm `r` (ví dụ: từ 32 xuống 16)
- Giảm batch size
- Sử dụng gradient accumulation
- Bật gradient checkpointing

### Hiệu Suất Kém
- Tăng `r` hoặc `lora_alpha`
- Thêm nhiều target modules hơn (bao gồm các lớp MLP)
- Thử khởi tạo EVA hoặc PiSSA
- Kiểm tra xem `target_modules` có đúng với model của bạn không

### Output Khác Khi Adapter Bị Tắt
- Điều này xảy ra nếu `bias != 'none'`
- Đặt `bias='none'` nếu bạn muốn output giống hệt khi adapter tắt

## Tài Liệu Tham Khảo

- [Hugging Face PEFT LoRA Documentation](https://huggingface.co/docs/peft/en/package_reference/lora)
- Paper LoRA gốc: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Paper EVA: [Explained Variance Adaptation](https://huggingface.co/papers/2402.03344)
- Paper PiSSA: [Principal Singular values and Singular vectors Adaptation](https://huggingface.co/papers/2404.02948)
