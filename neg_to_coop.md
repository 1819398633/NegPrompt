# NegPrompt to CoOp Structure

- **Load DataSet:** 略
- **Load cfg:** 略

依赖文件不显式写出

Make a new trainer class to integrate into CoOp structure

2 parts to transfer:

1. model definition: to the same file of trainer, call by the new trainer class
2. training pipeline: to the new trainer class

引用代码：加粗表示NegPrompt的代码，斜体表示CoOp框架

## 1. model definition: from models/model.py

### Positive Prompt

- **pos prompt:**
  - Basically the same as *`trainers/coop.py`*
  - **`prompt learner`**
  - **`text encoder`**
  - **`original CLIP`** → *`Customize CLIP`*

### Negative Prompt

- **neg prompt:**
- Make a new trainer with:
  - **`nega prompt learner`**
    - use **`nega prompt learner.forward negative`**
  - **`nega text encoder`**
  - **`Nega Prompt CLIP`** (use **`forward negative`**)
- *`new trainer.forward_backward`* (get loss and update):
  - **`core/train_clip.py`** → function **`train-nega-clip`**

## 2. Training Pipeline: for the new trainer

### Stage 1: Get Positive Prompt

- Check if other parameters needed to store (e.g., consistent dimension)
- Store pos prompt

### Stage 2: Get Negative Prompt

- *`new trainer.before_train`*:
  - **`NegaPromptLearner.update_ctx_positive`**
  - (Load pos prompt and initialize neg prompt)

- *`new trainer.run_epoch`*:
  - **`core/train_clip.py`** → function **`train-nega-clip`**

- *`new trainer.after_train`*:
  - Visualize (t-SNE)
  - **`Nega Prompt CLIP.draw-tsne-plot`**

- *`new trainer.test`*:
  - **`core/test_clip.py`** → function **`test-nega-clip`**

## 琐碎的东西

in **`osr_nega_prompt.py`**:
1. get cfg
2. get class names
3. load pretrained clip -> *`trainer.load_clip_to_cpu`*
4. load pretrained pos prompt -> *`new trainer.before_train`*
5. 它loss没用到所以不管
6. new trainer.build model