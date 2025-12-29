# Các Phương Pháp Prompt-Based trong PEFT

## Giới Thiệu

Một prompt có thể mô tả một tác vụ hoặc cung cấp một ví dụ về tác vụ mà bạn muốn model học. Thay vì tạo các prompt này thủ công, các phương pháp **soft prompting** thêm các tham số có thể học được vào input embeddings, có thể được tối ưu hóa cho một tác vụ cụ thể trong khi giữ nguyên các tham số của model pretrained (frozen). Điều này làm cho việc fine-tune các Large Language Models (LLM) cho các tác vụ downstream mới trở nên nhanh hơn và dễ dàng hơn.

Thư viện PEFT hỗ trợ nhiều loại phương pháp prompting:
- **P-tuning**
- **Prefix tuning**
- **Prompt tuning**

> [!TIP]
> Bạn có thể tìm hiểu thêm về cách các phương pháp này hoạt động về mặt khái niệm trong [Soft prompts guide](https://huggingface.co/docs/peft/en/conceptual_guides/prompting). Nếu bạn quan tâm đến việc áp dụng các phương pháp này cho các tác vụ và use case khác, hãy xem [notebook collection](https://huggingface.co/spaces/PEFT/soft-prompting)!

## Cài Đặt

Trước khi bắt đầu, đảm bảo bạn đã cài đặt tất cả các thư viện cần thiết:

```bash
pip install -q peft transformers datasets
```

## Dataset

Trong hướng dẫn này, chúng ta sẽ sử dụng tập con `twitter_complaints` của dataset [RAFT](https://huggingface.co/datasets/ought/raft). Tập con `twitter_complaints` chứa các tweet được gắn nhãn là `complaint` và `no complaint`.

### Tải Dataset

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={
        "train": "hf://datasets/ought/raft@refs/convert/parquet/twitter_complaints/train/0000.parquet",
        "test": "hf://datasets/ought/raft@refs/convert/parquet/twitter_complaints/test/0000.parquet"
    }
)

# Tạo cột text_label để dễ hiểu hơn
classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
ds = ds.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)

# Xem ví dụ
ds["train"][0]
# {"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2, "text_label": "no complaint"}
```

### Chuẩn Bị Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Xác định độ dài tối đa của label đã tokenize
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)
```

### Hàm Tiền Xử Lý

Tạo hàm tiền xử lý để tokenize tweet text và labels, padding inputs và labels trong mỗi batch, tạo attention mask, và truncate sequences đến `max_length`:

```python
import torch

max_length = 64

def preprocess_function(examples, text_column="Tweet text", label_column="text_label"):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
    
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        
        # Padding
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(label_input_ids)) + label_input_ids
        
        # Truncate và convert sang tensor
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

### Áp Dụng Tiền Xử Lý

```python
processed_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```

### Tạo DataLoader

```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_ds = processed_ds["train"]
eval_ds = processed_ds["test"]

batch_size = 16

train_dataloader = DataLoader(
    train_ds, 
    shuffle=True, 
    collate_fn=default_data_collator, 
    batch_size=batch_size, 
    pin_memory=True
)
eval_dataloader = DataLoader(
    eval_ds, 
    collate_fn=default_data_collator, 
    batch_size=batch_size, 
    pin_memory=True
)
```

## Model

Tải một model pretrained để sử dụng làm base model cho phương pháp soft prompt:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
```

## Các Phương Pháp Prompt-Based

### 1. P-tuning

**P-tuning** thêm một tensor embedding có thể học được nơi các prompt tokens có thể được thêm vào bất kỳ đâu trong input sequence.

#### Cấu Hình P-tuning

```python
from peft import PromptEncoderConfig, get_peft_model

peft_config = PromptEncoderConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,          # Số lượng virtual tokens để thêm và học
    encoder_hidden_size=128        # Hidden size của encoder để học prompt parameters
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: "trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"
```

#### Đặc Điểm P-tuning

- ✅ Thêm trainable embedding tensor
- ✅ Prompt tokens có thể được đặt ở bất kỳ đâu trong sequence
- ✅ Sử dụng encoder để học prompt parameters
- ✅ Linh hoạt trong việc đặt vị trí prompt

### 2. Prefix Tuning

**Prefix tuning** thêm các tham số cụ thể cho tác vụ vào tất cả các lớp của model, được tối ưu hóa bởi một mạng feed-forward riêng biệt.

#### Cấu Hình Prefix Tuning

```python
from peft import PrefixTuningConfig, get_peft_model

peft_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20          # Số lượng virtual tokens để thêm và học
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: "trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"
```

#### Đặc Điểm Prefix Tuning

- ✅ Thêm parameters vào tất cả các lớp của model
- ✅ Sử dụng feed-forward network riêng để tối ưu hóa
- ✅ Hiệu quả cho các tác vụ phức tạp
- ⚠️ Nhiều tham số trainable hơn so với P-tuning

### 3. Prompt Tuning

**Prompt tuning** xây dựng tất cả các tác vụ như một tác vụ *generation* và thêm một prompt cụ thể cho tác vụ vào input, được cập nhật độc lập.

#### Cấu Hình Prompt Tuning

```python
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# Định nghĩa prompt khởi tạo
prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"

peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,      # Khởi tạo từ text
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),  # Số tokens của prompt
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: "trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358"
```

#### Đặc Điểm Prompt Tuning

- ✅ Ít tham số trainable nhất (chỉ prompt embeddings)
- ✅ Đơn giản và hiệu quả
- ✅ Có thể khởi tạo từ text
- ✅ Tốt cho các tác vụ generation

> [!TIP]
> Để có kết quả tốt nhất, `prompt_tuning_init_text` nên có cùng số lượng tokens với số tokens cần dự đoán. Để làm điều này, bạn có thể đặt `num_virtual_tokens` bằng số tokens của `prompt_tuning_init_text`.

## So Sánh Các Phương Pháp

| Phương Pháp | Trainable Params | Độ Phức Tạp | Hiệu Suất | Use Case |
|-------------|------------------|-------------|-----------|----------|
| **P-tuning** | Trung bình (~0.05%) | Trung bình | Tốt | Linh hoạt, prompt ở bất kỳ đâu |
| **Prefix Tuning** | Cao (~0.18%) | Cao | Rất tốt | Tác vụ phức tạp, cần hiệu suất cao |
| **Prompt Tuning** | Rất thấp (~0.001%) | Thấp | Tốt | Đơn giản, ít tài nguyên |

## Huấn Luyện

### Thiết Lập Optimizer và Scheduler

```python
from transformers import get_linear_schedule_with_warmup

lr = 3e-2
num_epochs = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```

### Vòng Lặp Huấn Luyện

```python
from tqdm import tqdm

device = "cuda"
model = model.to(device)

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Evaluation
    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(
                torch.argmax(outputs.logits, -1).detach().cpu().numpy(), 
                skip_special_tokens=True
            )
        )

    # Tính toán metrics
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

## Lưu và Chia Sẻ Model

Sau khi huấn luyện xong, bạn có thể tải model lên Hub:

```python
from huggingface_hub import notebook_login

# Đăng nhập vào Hugging Face
notebook_login()

# Tải model lên Hub
account = "your-username"
peft_model_id = f"{account}/bloomz-560m-peft-method"
model.push_to_hub(peft_model_id)
```

> [!TIP]
> Nếu bạn kiểm tra kích thước file model trong repository, bạn sẽ thấy nó nhỏ hơn rất nhiều so với model đầy đủ! Ví dụ, adapter weights cho model opt-350m được lưu trên Hub chỉ ~6MB so với kích thước model đầy đủ có thể ~700MB.

## Inference

Tải model để inference và test trên một tweet:

```python
from peft import AutoPeftModelForCausalLM

# Tải model đã huấn luyện
model = AutoPeftModelForCausalLM.from_pretrained("peft_model_id").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

# Chuẩn bị input
i = 15
text_column = "Tweet text"
inputs = tokenizer(
    f'{text_column} : {ds["test"][i]["Tweet text"]} Label : ', 
    return_tensors="pt"
)
print(ds["test"][i]["Tweet text"])
# "@NYTsupport i have complained a dozen times &amp; yet my papers are still thrown FAR from my door. Why is this so hard to resolve?"
```

Generate prediction:

```python
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], 
        max_new_tokens=10
    )
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
# "['Tweet text : @NYTsupport i have complained a dozen times &amp; yet my papers are still thrown FAR from my door. Why is this so hard to resolve? Label : complaint']"
```

## Best Practices

### 1. Chọn Phương Pháp Phù Hợp

- **Prompt Tuning**: Khi bạn có ít tài nguyên và cần giải pháp đơn giản
- **P-tuning**: Khi bạn cần linh hoạt trong việc đặt prompt
- **Prefix Tuning**: Khi bạn cần hiệu suất cao nhất và có đủ tài nguyên

### 2. Số Lượng Virtual Tokens

- Bắt đầu với 10-20 virtual tokens
- Tăng lên nếu tác vụ phức tạp
- Giảm nếu overfitting

### 3. Learning Rate

- Prompt-based methods thường cần learning rate cao hơn (1e-2 đến 3e-2)
- Thử nghiệm với các learning rate khác nhau
- Sử dụng learning rate scheduler

### 4. Khởi Tạo Prompt

- Với Prompt Tuning, sử dụng text mô tả rõ ràng tác vụ
- Đảm bảo số tokens của prompt khởi tạo khớp với `num_virtual_tokens`
- Thử nghiệm với các prompt khởi tạo khác nhau

### 5. Epochs

- Prompt-based methods thường cần nhiều epochs hơn (20-50)
- Monitor validation loss để tránh overfitting
- Sử dụng early stopping nếu cần

## Ưu và Nhược Điểm

### Ưu Điểm

- ✅ **Tiết kiệm tài nguyên**: Chỉ train một phần nhỏ tham số
- ✅ **Nhanh**: Huấn luyện nhanh hơn so với full fine-tuning
- ✅ **Linh hoạt**: Dễ dàng chuyển đổi giữa các tác vụ
- ✅ **Bảo toàn model gốc**: Model base không bị thay đổi
- ✅ **Kích thước nhỏ**: Adapter weights rất nhỏ

### Nhược Điểm

- ❌ **Hiệu suất có thể thấp hơn**: So với full fine-tuning trên một số tác vụ
- ❌ **Cần nhiều epochs**: Thường cần nhiều epochs hơn để hội tụ
- ❌ **Phụ thuộc vào prompt**: Chất lượng prompt ảnh hưởng đến kết quả
- ❌ **Learning rate cao**: Cần learning rate cao hơn, có thể không ổn định

## Kết Luận

Các phương pháp prompt-based trong PEFT cung cấp một cách hiệu quả để fine-tune LLM cho các tác vụ mới mà không cần cập nhật toàn bộ model. Mỗi phương pháp có ưu và nhược điểm riêng:

- **Prompt Tuning**: Đơn giản nhất, ít tham số nhất
- **P-tuning**: Linh hoạt, có thể đặt prompt ở bất kỳ đâu
- **Prefix Tuning**: Hiệu suất cao nhất, nhưng nhiều tham số hơn

Việc lựa chọn phương pháp phụ thuộc vào tài nguyên có sẵn, yêu cầu hiệu suất, và độ phức tạp của tác vụ.

## Tài Liệu Tham Khảo

- [Hugging Face PEFT - Prompt-based methods](https://huggingface.co/docs/peft/en/task_guides/prompt_based_methods)
- [Soft prompts conceptual guide](https://huggingface.co/docs/peft/en/conceptual_guides/prompting)
- [PEFT Notebook Collection](https://huggingface.co/spaces/PEFT/soft-prompting)
- [RAFT Dataset](https://huggingface.co/datasets/ought/raft)

