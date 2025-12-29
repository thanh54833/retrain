# Cấu Hình và Model PEFT

## Giới Thiệu

Kích thước khổng lồ của các model pretrained hiện đại - thường có hàng tỷ tham số - tạo ra thách thức đáng kể trong việc huấn luyện vì chúng yêu cầu nhiều không gian lưu trữ và sức mạnh tính toán hơn để xử lý tất cả các phép tính. Bạn sẽ cần truy cập vào GPU hoặc TPU mạnh để huấn luyện các model pretrained lớn này, điều này rất tốn kém, không phổ biến với mọi người, không thân thiện với môi trường và không thực tế lắm. Các phương pháp PEFT giải quyết nhiều thách thức này.

Có nhiều loại phương pháp PEFT (soft prompting, matrix decomposition, adapters), nhưng tất cả đều tập trung vào cùng một mục tiêu: giảm số lượng tham số có thể huấn luyện. Điều này làm cho việc huấn luyện và lưu trữ các model lớn trên phần cứng tiêu dùng trở nên dễ tiếp cận hơn.

Thư viện PEFT được thiết kế để giúp bạn nhanh chóng huấn luyện các model lớn trên GPU miễn phí hoặc chi phí thấp. Trong tutorial này, bạn sẽ học cách thiết lập cấu hình để áp dụng phương pháp PEFT vào một model gốc pretrained để huấn luyện. Sau khi cấu hình PEFT được thiết lập, bạn có thể sử dụng bất kỳ framework huấn luyện nào bạn thích (lớp [Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.Trainer) của Transformers, [Accelerate](https://hf.co/docs/accelerate), hoặc một vòng lặp huấn luyện PyTorch tùy chỉnh).

## Cấu Hình PEFT

> [!TIP]
> Tìm hiểu thêm về các tham số bạn có thể cấu hình cho từng phương pháp PEFT trong trang API reference tương ứng của chúng.

Một cấu hình lưu trữ các tham số quan trọng chỉ định cách một phương pháp PEFT cụ thể nên được áp dụng.

Ví dụ, hãy xem cấu hình [`LoraConfig`](https://huggingface.co/ybelkada/opt-350m-lora/blob/main/adapter_config.json) sau đây để áp dụng LoRA và [`PromptEncoderConfig`](https://huggingface.co/smangrul/roberta-large-peft-p-tuning/blob/main/adapter_config.json) để áp dụng p-tuning (các file cấu hình này đã được JSON-serialized). Bất cứ khi nào bạn tải một PEFT adapter, nên kiểm tra xem nó có file `adapter_config.json` liên quan hay không, file này là bắt buộc.

### Ví Dụ LoraConfig (JSON)

```json
{
  "base_model_name_or_path": "facebook/opt-350m",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "revision": null,
  "target_modules": [
    "q_proj",
    "v_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

### Tạo LoraConfig trong Python

Bạn có thể tạo cấu hình của riêng mình để huấn luyện bằng cách khởi tạo [`LoraConfig`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/lora#peft.LoraConfig):

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
```

### Ví Dụ PromptEncoderConfig (JSON)

```json
{
  "base_model_name_or_path": "roberta-large",
  "encoder_dropout": 0.0,
  "encoder_hidden_size": 128,
  "encoder_num_layers": 2,
  "encoder_reparameterization_type": "MLP",
  "inference_mode": true,
  "num_attention_heads": 16,
  "num_layers": 24,
  "num_transformer_submodules": 1,
  "num_virtual_tokens": 20,
  "peft_type": "P_TUNING",
  "task_type": "SEQ_CLS",
  "token_dim": 1024
}
```

### Tạo PromptEncoderConfig trong Python

Bạn có thể tạo cấu hình của riêng mình để huấn luyện bằng cách khởi tạo [`PromptEncoderConfig`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/p_tuning#peft.PromptEncoderConfig):

```python
from peft import PromptEncoderConfig, TaskType

p_tuning_config = PromptEncoderConfig(
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=128,
    num_attention_heads=16,
    num_layers=24,
    num_transformer_submodules=1,
    num_virtual_tokens=20,
    token_dim=1024,
    task_type=TaskType.SEQ_CLS
)
```

## Model PEFT

Với cấu hình PEFT trong tay, bạn có thể áp dụng nó vào bất kỳ model pretrained nào để tạo một [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel). Chọn từ bất kỳ model state-of-the-art nào từ thư viện [Transformers](https://hf.co/docs/transformers), một model tùy chỉnh, và thậm chí các kiến trúc transformer mới và chưa được hỗ trợ.

### Tải Model Gốc

Cho tutorial này, hãy tải model gốc [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) để fine-tune:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```

### Tạo PeftModel

Sử dụng hàm [`get_peft_model()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.get_peft_model) để tạo một [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) từ model gốc facebook/opt-350m và `lora_config` bạn đã tạo trước đó:

```python
from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# Output: "trainable params: 1,572,864 || all params: 332,769,280 || trainable%: 0.472659014678278"
```

> [!WARNING]
> Khi gọi [`get_peft_model()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.get_peft_model), model gốc sẽ bị **sửa đổi tại chỗ** (in-place). Điều này có nghĩa là, khi gọi [`get_peft_model()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.get_peft_model) trên một model đã được sửa đổi theo cách tương tự trước đó, model này sẽ bị đột biến thêm. Do đó, nếu bạn muốn sửa đổi cấu hình PEFT sau khi đã gọi `get_peft_model()` trước đó, bạn sẽ phải unload model bằng [`unload()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.unload) trước, sau đó gọi `get_peft_model()` với cấu hình mới của bạn. Ngoài ra, bạn có thể khởi tạo lại model để đảm bảo trạng thái mới, chưa được sửa đổi trước khi áp dụng cấu hình PEFT mới.

### Lưu Model

Bây giờ bạn có thể huấn luyện [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) với framework huấn luyện ưa thích của bạn! Sau khi huấn luyện, bạn có thể lưu model của mình cục bộ bằng [`save_pretrained()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel.save_pretrained) hoặc tải lên Hub bằng phương thức [`push_to_hub`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub):

```python
# Lưu cục bộ
lora_model.save_pretrained("your-name/opt-350m-lora")

# Tải lên Hub
lora_model.push_to_hub("your-name/opt-350m-lora")
```

### Tải Model Để Inference

Để tải một [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) cho inference, bạn cần cung cấp [`PeftConfig`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/config#peft.PeftConfig) được sử dụng để tạo nó và model gốc mà nó được huấn luyện từ đó:

```python
from peft import PeftModel, PeftConfig

# Tải cấu hình
config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")

# Tải model gốc
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

# Tải adapter
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```

> [!TIP]
> Theo mặc định, [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) được đặt cho inference, nhưng nếu bạn muốn huấn luyện adapter thêm một chút, bạn có thể đặt `is_trainable=True`.
>
> ```python
> lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora", is_trainable=True)
> ```

### Sử Dụng AutoPeftModel

Phương thức [`PeftModel.from_pretrained()`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained) là cách linh hoạt nhất để tải một [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) vì nó không quan trọng framework model nào được sử dụng (Transformers, timm, một model PyTorch generic). Các lớp khác, như [`AutoPeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/auto_class#peft.AutoPeftModel), chỉ là một wrapper tiện lợi xung quanh [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) gốc, và làm cho việc tải PEFT model trực tiếp từ Hub hoặc cục bộ nơi trọng số PEFT được lưu trữ trở nên dễ dàng hơn.

```python
from peft import AutoPeftModelForCausalLM

lora_model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
```

Xem [AutoPeftModel](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/auto_class) API reference để tìm hiểu thêm về các lớp [`AutoPeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/auto_class#peft.AutoPeftModel).

## Ví Dụ Hoàn Chỉnh

Dưới đây là một ví dụ hoàn chỉnh từ đầu đến cuối:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Tải model và tokenizer gốc
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Tạo cấu hình LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 3. Áp dụng LoRA vào model
lora_model = get_peft_model(model, lora_config)

# 4. Kiểm tra tham số có thể huấn luyện
lora_model.print_trainable_parameters()
# Output: trainable params: 1,572,864 || all params: 332,769,280 || trainable%: 0.47

# 5. Huấn luyện model (ví dụ với Trainer)
# ... training code ...

# 6. Lưu model sau khi huấn luyện
lora_model.save_pretrained("./opt-350m-lora")

# 7. Tải lại model để inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)
loaded_model = PeftModel.from_pretrained(base_model, "./opt-350m-lora")
```

## Các Bước Tiếp Theo

Với [`PeftConfig`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/config#peft.PeftConfig) phù hợp, bạn có thể áp dụng nó vào bất kỳ model pretrained nào để tạo một [`PeftModel`](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) và huấn luyện các model lớn mạnh mẽ nhanh hơn trên GPU miễn phí có sẵn! Để tìm hiểu thêm về cấu hình và model PEFT, các hướng dẫn sau có thể hữu ích:

- Tìm hiểu cách cấu hình phương pháp PEFT cho các model không phải từ Transformers trong hướng dẫn [Working with custom models](https://huggingface.co/docs/peft/en/developer_guides/custom_models).

## Lưu Ý Quan Trọng

### File adapter_config.json

Khi bạn lưu một PEFT adapter, một file `adapter_config.json` sẽ được tạo tự động. File này chứa tất cả các tham số cấu hình cần thiết để tải lại adapter sau này. Luôn đảm bảo file này tồn tại khi chia sẻ hoặc sử dụng adapter.

### In-Place Modification

Nhớ rằng `get_peft_model()` sửa đổi model gốc tại chỗ. Nếu bạn cần giữ nguyên model gốc, hãy tạo một bản sao trước:

```python
import copy

# Tạo bản sao nếu cần giữ nguyên model gốc
model_copy = copy.deepcopy(model)
lora_model = get_peft_model(model_copy, lora_config)
```

### Nhiều Adapter

Bạn có thể thêm nhiều adapter vào cùng một model gốc:

```python
# Thêm adapter đầu tiên
lora_model.add_adapter("adapter1", lora_config_1)

# Thêm adapter thứ hai
lora_model.add_adapter("adapter2", lora_config_2)

# Chuyển đổi giữa các adapter
lora_model.set_adapter("adapter1")
```

## Tài Liệu Tham Khảo

- [Hugging Face PEFT Tutorial: PEFT configurations and models](https://huggingface.co/docs/peft/en/tutorial/peft_model_config)
- [PEFT Model API Reference](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel)
- [PEFT Config API Reference](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/config#peft.PeftConfig)

