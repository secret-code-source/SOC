Implementation of manuscript 'Second-order correlation learning for self-supervised speech emotion recognition'

- For data pre-processing and partitioning, please refer to the [EmoBox repository](https://github.com/emo-box/EmoBox).

- For the pre-trained up-stream models, please refer to the official Hugging Face repositories:
  - **Wav2Vec 2.0**: [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)
  - **HuBERT**: [`facebook/hubert-base-ls960`](https://huggingface.co/facebook/hubert-base-ls960)
  - **WavLM**: [`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus)

## 📝 To-Do List

- [x] Initial code release (Second-order correlation pooling module).
- [x] Instructions for data pre-processing and partitioning (Adapted from EmoBox).
- [x] Add instructions for **preparing pre-trained upstream models** (Wav2Vec 2.0, HuBERT, WavLM).
- [ ] Release the requirements.txt and environment setup guide (Tested on Ubuntu 20.04 & CUDA 11.8, Python 3.9, PyTorch 2.0+).
- [ ] Upload training and evaluation script (Coming soon).
- [ ] Add a Jupyter Notebook demo for t-SNE visualization.
