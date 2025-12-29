# PEFT Method Selection Guide

## Overview

Choosing the right PEFT method is crucial to achieve the best performance with available resources. This document analyzes the three main groups of PEFT methods and provides guidance on when to use which method based on specific conditions and requirements.

## Overall Comparison of Methods

| Method | Trainable Parameters | Training Speed | Memory Requirements | Complexity | Performance |
|--------|---------------------|----------------|---------------------|------------|--------------|
| **Prompt Tuning** | ~0.001% | Slow (many epochs) | Very low | Low | Good |
| **P-tuning** | ~0.05% | Slow (many epochs) | Low | Medium | Good |
| **Prefix Tuning** | ~0.18% | Slow (many epochs) | Low | High | Very good |
| **LoRA** | ~0.13-0.77% | Fast | Low | Low | Very good |
| **LoHa** | ~1.44% | Fast | Medium | Medium | Very good |
| **LoKr** | ~0.13% | Fast | Low | Medium | Good |
| **AdaLoRA** | ~0.59% | Fast | Medium | High | Very good |
| **IA3** | ~0.02% | Very fast | Very low | Low | Good |

## Decision Matrix

### 1. Based on Model Type

#### Sequence-to-Sequence Models (T5, mT0, BART)
- ✅ **IA3**: Best choice - fewest parameters, fastest
- ✅ **LoRA**: Good choice - popular, stable
- ⚠️ **Prompt-based**: Can be used but not optimal

**Recommendation**: Start with **IA3**, switch to **LoRA** if desired performance is not achieved.

#### Causal Language Models (GPT, LLaMA, Mistral)
- ✅ **LoRA**: Best choice - widely supported, high performance
- ✅ **Prompt-based**: Good for generation tasks
- ⚠️ **IA3**: Can be used but less documented

**Recommendation**: Start with **LoRA**, try **Prompt Tuning** if fewer parameters are needed.

#### Vision Models (ViT, CLIP)
- ✅ **LoRA**: Best choice - popular for vision tasks
- ✅ **LoHa**: Good when higher rank is needed
- ❌ **Prompt-based**: Not suitable
- ❌ **IA3**: Not well supported

**Recommendation**: Use **LoRA** with `target_modules=["query", "value"]`.

#### Diffusion Models (Stable Diffusion)
- ✅ **LoRA**: Most popular
- ✅ **LoHa**: Good for controllable generation
- ✅ **LoKr**: Good for vectorization
- ❌ **Prompt-based**: Not suitable
- ❌ **IA3**: Not supported

**Recommendation**: Start with **LoRA**, try **LoHa** if higher rank is needed.

### 2. Based on Available Resources

#### Very Limited Resources (< 8GB GPU)
- ✅ **Prompt Tuning**: Fewest parameters (~0.001%)
- ✅ **IA3**: Very few parameters (~0.02%)
- ✅ **LoRA with r=8**: Few parameters (~0.13%)

**Recommendation**: Start with **Prompt Tuning** or **IA3**, upgrade to **LoRA** if needed.

#### Limited Resources (8-16GB GPU)
- ✅ **LoRA with r=16**: Good balance
- ✅ **P-tuning**: If suitable for task
- ✅ **IA3**: For Seq2Seq models

**Recommendation**: Use **LoRA with r=16**, try **IA3** for Seq2Seq.

#### Sufficient Resources (16-24GB GPU)
- ✅ **LoRA with r=32-64**: Higher performance
- ✅ **Prefix Tuning**: For complex tasks
- ✅ **LoHa**: When higher rank is needed

**Recommendation**: Use **LoRA with r=32**, try **Prefix Tuning** or **LoHa** if needed.

#### Abundant Resources (> 24GB GPU)
- ✅ **LoRA with r=64+**: Maximum performance
- ✅ **AdaLoRA**: Smart parameter allocation
- ✅ **Prefix Tuning**: For most complex tasks

**Recommendation**: Use **LoRA with r=64** or **AdaLoRA** for optimization.

### 3. Based on Task Type

#### Text Classification
- ✅ **LoRA**: Best choice
- ✅ **Prompt Tuning**: If fewer parameters needed
- ⚠️ **IA3**: Can be used

**Recommendation**: **LoRA with r=16-32**.

#### Text Generation
- ✅ **Prompt Tuning**: Good for generation
- ✅ **LoRA**: Popular choice
- ✅ **Prefix Tuning**: For complex tasks

**Recommendation**: Start with **Prompt Tuning**, upgrade to **LoRA** or **Prefix Tuning** if needed.

#### Translation
- ✅ **IA3**: Best choice for Seq2Seq
- ✅ **LoRA**: Good choice
- ⚠️ **Prompt-based**: Can be used

**Recommendation**: **IA3** for Seq2Seq models, **LoRA** for others.

#### Summarization
- ✅ **IA3**: Good for Seq2Seq
- ✅ **LoRA**: Popular choice
- ✅ **Prefix Tuning**: For complex tasks

**Recommendation**: **IA3** for Seq2Seq, **LoRA** for others.

#### Image Classification
- ✅ **LoRA**: Only suitable choice
- ✅ **LoHa**: If higher rank needed

**Recommendation**: **LoRA with r=16-32**.

#### Controllable Generation (Text-to-Image)
- ✅ **LoHa**: Good for high rank
- ✅ **LoRA**: Popular choice
- ✅ **OFT/BOFT**: Good for subject preservation

**Recommendation**: **LoHa** or **LoRA with r=32+**.

### 4. Based on Performance Requirements

#### Need Maximum Performance
- ✅ **LoRA with r=64+**: Highest performance
- ✅ **Prefix Tuning**: For complex tasks
- ✅ **AdaLoRA**: Smart parameter allocation

**Recommendation**: **LoRA with r=64** or **AdaLoRA**.

#### Need Balance Between Performance and Efficiency
- ✅ **LoRA with r=16-32**: Good balance
- ✅ **P-tuning**: For prompt-based
- ✅ **IA3**: For Seq2Seq

**Recommendation**: **LoRA with r=16-32**.

#### Need Maximum Efficiency (Few Parameters)
- ✅ **Prompt Tuning**: Fewest parameters
- ✅ **IA3**: Very few parameters
- ✅ **LoRA with r=8**: Few parameters

**Recommendation**: **Prompt Tuning** or **IA3**.

### 5. Based on Dataset Size

#### Small Dataset (< 1K samples)
- ✅ **Prompt Tuning**: Few parameters, less overfitting
- ✅ **LoRA with r=8**: Few parameters
- ⚠️ **LoRA with high r**: May overfit

**Recommendation**: **Prompt Tuning** or **LoRA with r=8**.

#### Medium Dataset (1K-10K samples)
- ✅ **LoRA with r=16**: Good balance
- ✅ **P-tuning**: For prompt-based
- ✅ **IA3**: For Seq2Seq

**Recommendation**: **LoRA with r=16** or **IA3**.

#### Large Dataset (> 10K samples)
- ✅ **LoRA with r=32-64**: High performance
- ✅ **Prefix Tuning**: For complex tasks
- ✅ **AdaLoRA**: Smart parameter allocation

**Recommendation**: **LoRA with r=32-64** or **AdaLoRA**.

### 6. Based on Training Time

#### Need Fast Training
- ✅ **IA3**: Fastest
- ✅ **LoRA**: Fast
- ❌ **Prompt-based**: Slow (requires many epochs)

**Recommendation**: **IA3** or **LoRA**.

#### Have Sufficient Training Time
- ✅ **LoRA**: Good balance
- ✅ **Prompt-based**: Can be used
- ✅ **AdaLoRA**: Needs time for allocation

**Recommendation**: **LoRA** or **Prompt-based**.

#### Have Long Training Time
- ✅ **LoRA with high r**: Maximum performance
- ✅ **Prefix Tuning**: For complex tasks
- ✅ **AdaLoRA**: Optimize allocation

**Recommendation**: **LoRA with r=64+** or **AdaLoRA**.

### 7. Based on Experience and Support

#### New to PEFT
- ✅ **LoRA**: Simplest, most documentation
- ✅ **Prompt Tuning**: Simple
- ❌ **AdaLoRA**: Complex, requires custom training loop

**Recommendation**: Start with **LoRA**.

#### Experienced
- ✅ **All methods**: Can experiment
- ✅ **AdaLoRA**: For optimization
- ✅ **LoHa/LoKr**: For special use cases

**Recommendation**: Experiment with different methods.

#### Need Community Support
- ✅ **LoRA**: Best support
- ✅ **Prompt Tuning**: Good support
- ⚠️ **IA3**: Less support
- ⚠️ **LoHa/LoKr**: Less support

**Recommendation**: **LoRA** or **Prompt Tuning**.

## Decision Workflow

### Step 1: Identify Model Type
```
Seq2Seq Model?
├─ Yes → Consider IA3 or LoRA
└─ No → Consider LoRA or Prompt-based
```

### Step 2: Evaluate Resources
```
GPU Memory < 8GB?
├─ Yes → Prompt Tuning or IA3
└─ No → LoRA with appropriate r
```

### Step 3: Determine Performance Requirements
```
Need maximum performance?
├─ Yes → LoRA r=64+ or AdaLoRA
└─ No → LoRA r=16-32 or Prompt Tuning
```

### Step 4: Consider Dataset
```
Dataset < 1K?
├─ Yes → Prompt Tuning or LoRA r=8
└─ No → LoRA r=16-32
```

### Step 5: Final Decision
Based on all factors above, choose the most suitable method.

## Concrete Examples

### Example 1: Fine-tune LLaMA 7B for Chatbot
- **Model**: Causal LM (LLaMA)
- **Resources**: 16GB GPU
- **Task**: Text generation
- **Dataset**: 5K samples
- **Requirement**: Good performance

**Recommendation**: **LoRA with r=32, target_modules=["q_proj", "v_proj"]**

### Example 2: Fine-tune T5 for Translation
- **Model**: Seq2Seq (T5)
- **Resources**: 8GB GPU
- **Task**: Translation
- **Dataset**: 10K samples
- **Requirement**: Maximum efficiency

**Recommendation**: **IA3** (fewest parameters, fastest for Seq2Seq)

### Example 3: Fine-tune ViT for Image Classification
- **Model**: Vision Transformer
- **Resources**: 12GB GPU
- **Task**: Image classification
- **Dataset**: 20K samples
- **Requirement**: Good performance

**Recommendation**: **LoRA with r=16, target_modules=["query", "value"]**

### Example 4: Fine-tune GPT-2 for Text Classification
- **Model**: Causal LM (GPT-2)
- **Resources**: 6GB GPU
- **Task**: Text classification
- **Dataset**: 2K samples
- **Requirement**: Few parameters

**Recommendation**: **Prompt Tuning** (fewest parameters, suitable for classification)

### Example 5: Fine-tune Stable Diffusion
- **Model**: Diffusion Model
- **Resources**: 24GB GPU
- **Task**: Text-to-image
- **Dataset**: 50K samples
- **Requirement**: High performance

**Recommendation**: **LoRA with r=64** or **LoHa** (for higher rank)

## Quick Summary Table

| Condition | Recommended Method | Rank/Config |
|-----------|-------------------|-------------|
| **Seq2Seq + limited resources** | IA3 | - |
| **Seq2Seq + sufficient resources** | LoRA | r=16-32 |
| **Causal LM + limited resources** | Prompt Tuning | - |
| **Causal LM + sufficient resources** | LoRA | r=16-32 |
| **Vision Model** | LoRA | r=16-32 |
| **Diffusion Model** | LoRA/LoHa | r=32-64 |
| **Small dataset** | Prompt Tuning | - |
| **Large dataset** | LoRA | r=32-64 |
| **Need maximum performance** | LoRA/AdaLoRA | r=64+ |
| **Need maximum efficiency** | Prompt Tuning/IA3 | - |
| **New to PEFT** | LoRA | r=16 |

## Best Practices

### 1. Start Simple
- Always start with **LoRA r=16** if uncertain
- This is a good starting point for most cases

### 2. Experiment Gradually
- Start with low rank (r=8-16)
- Gradually increase if higher performance is needed
- Decrease if overfitting

### 3. Monitor Metrics
- Track both training and validation loss
- Adjust rank based on performance
- Use early stopping if needed

### 4. Combine When Needed
- Can combine multiple methods
- Example: LoRA + Prompt Tuning for some tasks

### 5. Optimize Gradually
- Start with simple method
- Upgrade to more complex methods if needed
- Evaluate trade-off between performance and cost

## Conclusion

Choosing the right PEFT method depends on many factors:

1. **Model type**: Seq2Seq → IA3, Causal LM → LoRA, Vision → LoRA
2. **Resources**: Limited → Prompt Tuning/IA3, Sufficient → LoRA
3. **Performance requirements**: High → LoRA high r, Low → Prompt Tuning
4. **Dataset size**: Small → Prompt Tuning, Large → LoRA
5. **Experience**: New → LoRA, Experienced → Experiment

**General recommendation**: 
- Start with **LoRA r=16** for most cases
- Use **IA3** for Seq2Seq models when fewer parameters are needed
- Use **Prompt Tuning** when resources are very limited
- Upgrade to **LoRA high r** or **AdaLoRA** when maximum performance is needed

Remember that there is no "best" method for all cases. The choice depends on your specific situation.

## References

- [Prompt-based Methods Guide](../01_research/03_peft_method_prompt_base.md)
- [LoRA Methods Guide](../01_research/04_peft_method_lora_method.md)
- [IA3 Guide](../01_research/05_peft_method_ia3.md)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)

