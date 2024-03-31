# Large Language Models (LLMs) 


-------
-------


## 1) What are Large Language Models (LLMs) and how do they work?

Large Language Models (LLMs) represent a breakthrough in natural language processing, leveraging advanced machine learning techniques to understand and generate human-like text. These models have the capability to analyze vast amounts of textual data and generate coherent responses.

### Core Components and Operation

- **Encoder-Decoder Framework:** LLMs employ an encoder-decoder framework, with models like GPT-3 focusing on single-directional processing and BERT incorporating bidirectional workflows.
  
- **Transformer Architecture:** LLMs utilize transformer architectures, comprising multiple transformer blocks with multi-headed self-attention mechanisms. This architecture enables the model to comprehend the context of words within a given text efficiently.
  
- **Vocabulary and Tokenization:** Text is segmented into manageable pieces known as "tokens," which are managed through a predefined vocabulary set.
  
- **Embeddings:** LLMs utilize embeddings, which are high-dimensional numerical representations of tokens, providing contextual understanding to the model.
  
- **Self-Attention Mechanisms:** These mechanisms enable the model to capture relationships between different tokens within the same sentence or across sentence pairs.

### Training Mechanism

- **Unsupervised Pretraining:** LLMs undergo unsupervised pretraining, where they familiarize themselves with the structure of text using massive datasets, often sourced from internet content.
  
- **Fine-Tuning:** Algorithms fine-tune specific parameters based on the objectives of the task at hand.
  
- **Prompt-Based Learning:** This method involves providing the model with specific questions or directives, guiding it towards creating tailored and targeted content.
  
- **Continual Training:** LLMs undergo continual training to ensure they stay updated with the latest data trends, linguistic nuances, and language shifts.

------

## 2) Describe the architecture of a transformer model that is commonly used in LLMs.

## Core Components

### Encoder-Decoder Model
The Transformer initially featured separate encoders for processing the input sequence and decoders for generating outputs. However, variants like GPT (Generative Pre-trained Transformer) have focused on using only the encoder for tasks such as language modeling.

### Self-Attention Mechanism
This enables the model to weigh different parts of the input sequence when processing each element. This mechanism forms the heart of both the encoder and decoder.

## Model Overview

<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2017.04.49.png" alt="Transformer Model" />
</div>

<br>
Let's begin by looking at the model as a single black box. In a machine translation application, it would take a sentence in one language and output its translation in another.

### Internal Components

- **Encoding Component:** This comprises a stack of N encoders.
- **Decoding Component:** This comprises a stack of N decoders, with connections between them.

### Encoder Architecture

Each encoder consists of two sub-layers:
1. **Self-Attention Layer:** Processes inputs to weigh different parts of the input sequence.
2. **Feed-Forward Neural Network Layer:** Receives outputs from the self-attention layer and applies further transformations.

The input sequence flows through the self-attention layer first, followed by the feed-forward neural network layer. This process repeats until it reaches the last encoder.

### Decoder Architecture

The decoder receives the output of the encoder component and includes both the self-attention layer and feed-forward layer. However, between them lies an attention layer that helps the decoder focus on relevant parts of the input sentence.

-------

<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2017.05.01.png" />
</div>
<br>



Now, each encoder is broken down into two sub-layers: the self-attention layer and the feed-forward neural network layer.

- The inputs first flow through a self-attention layer, and the outputs of the self-attention layer are fed to a feed-forward neural network. And this sequence is repeated till reaches the last encoder.

- Finally, the decoder receives the output of the encoder component and also has both the self-attention layer and feed-forward layer, and the flow is similar to before, but between them there is an attention layer that helps the decoder focus on relevant parts of the input sentence.


<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2017.05.14.png" />
</div>
<br>
