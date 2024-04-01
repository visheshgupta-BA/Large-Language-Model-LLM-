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
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2017.04.49.png" alt="Transformer Model" width="50%" height="50%" />
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
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2017.05.01.png" width="50%" height="50%" />
</div>
<br>



Now, each encoder is broken down into two sub-layers: the self-attention layer and the feed-forward neural network layer.

- The inputs first flow through a self-attention layer, and the outputs of the self-attention layer are fed to a feed-forward neural network. And this sequence is repeated till reaches the last encoder.

- Finally, the decoder receives the output of the encoder component and also has both the self-attention layer and feed-forward layer, and the flow is similar to before, but between them there is an attention layer that helps the decoder focus on relevant parts of the input sentence.


<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2017.05.14.png" width="50%" height="50%" />
</div>
<br>

-------

## 3) How do the Generative Language models works?


#### The very basic idea is the following: they take n tokens as input, and produce one token as output.

<b> A token is a chunk of text</b>. In the context of OpenAI GPT models, common and short words typically correspond to a single token and long and less commonly used words are generally broken up into several tokens.


This basic idea is applied in an expanding-window pattern. You give it n tokens in, it produces one token out, then it incorporates that output token as part of the input of the next iteration, produces a new token out, and so on. This pattern keeps repeating until a stopping condition is reached, indicating that it finished generating all the text you need.

<br>
<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2017.20.47.png" width="50%" height="50%" />
</div>
<br>


Now, behind the output is a probability distribution over all the possible tokens. What the model does is return a vector in which each entry expresses the probability of a particular token being chosen.

-------

## 4) What is token in the large language models context

ChatGPT and other LLMs rely on input text being broken into pieces. Each piece is about a word-sized sequence of characters or smaller. We call those sub-word tokens. That process is called tokenization and is done using a tokenizer.

Tokens can be words or just chunks of characters. For example, the word “hamburger” gets broken up into the tokens “ham”, “bur” and “ger”, while a short and common word like “pear” is a single token. Many tokens start with whitespace, for example, “ hello” and “ bye”.

The models understand the statistical relationships between these tokens and excel at producing the next token in a sequence of tokens.

<b><I>The number of tokens processed in a given API request depends on the length of both your inputs and outputs. As a rough rule of thumb, the 1 token is approximately 4 characters or 0.75 words for English text.</I><b/>


<br>
<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2019.10.38.png" width="50%" height="50%" />
</div>
<br>
-------

## 5) How can you evaluate the performance of Language Models?

There are two ways to evaluate language models in NLP: Extrinsic evaluation and Intrinsic evaluation.

- **Intrinsic evaluation** captures how well the model captures what it is supposed to capture, like probabilities.
- **Extrinsic evaluation** (or task-based evaluation) captures how useful the model is in a particular task.

<br>
<b>A common intrinsic evaluation of LM is the perplexity</b>. It's a geometric average of the inverse probability of words predicted by the model. Intuitively, perplexity means to be surprised. We measure how much the model is surprised by seeing new data. The lower the perplexity, the better the training is. Another common measure is the cross-entropy, which is the Logarithm (base 2) of perplexity. As a thumb rule, a reduction of 10-20% in perplexity is noteworthy.

<br>
- The extrinsic evaluation will depend on the task. Example: For speech recognition, we can compare the performance of two language models by running the speech recognizer twice, once with each language model, and seeing which gives the more accurate transcription.

-------

## 6) How LLMs are Pre-Trained?

Pre-training an LLM is basically training an LLM on a large amount of data (a few billion words at least) with the primary task of predicting words in a sentence. Now there are two ways in which we can do this, depending on the type of model:

<br>
<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2019.59.21.png" width="50%" height="50%" />
</div>
<br>


- One way is called **Masked Language Model (MLM)** used by bi-directional models like BERT, in which a certain percentage of words in the training set are masked, and the task of the model is to predict these missing words. Note that in this task, the model can see the words preceding as well as succeeding the missing word, and that's why it's called bi-directional.

- There are other kinds of models called **auto-regressive (eg. GPT)**, which are uni-directional and they are trained to predict the next word without seeing the succeeding ones. This is because these auto-regressive models are specifically designed for better language generation, which makes it necessary for the model to be pre-trained in a uni-directional manner.

-------

## 7) What kind of Tokenization techniques are there in LLMs?
<br>
**Tokenization** is the process of converting raw text into a sequence of smaller units, called tokens, which can be words, subwords, or characters. Some tokenization methods and techniques used in LLMs are:
<br>

- **Word-based tokenization**: This method splits text into individual words, treating each word as a separate token. While simple and intuitive, word-based tokenization can struggle with out-of-vocabulary words and may not efficiently handle languages with complex morphology.

- **Subword-based tokenization**: Subword-based methods, such as Byte Pair Encoding (BPE) and WordPiece, split text into smaller units that can be combined to form whole words. This approach enables LLMs to handle out-of-vocabulary words and better capture the structure of different languages. BPE, for instance, merges the most frequently occurring character pairs to create subword units, while WordPiece employs a data-driven approach to segment words into subword tokens.

- **Character-based tokenization**: This method treats individual characters as tokens. Although it can handle any input text, character-based tokenization often requires larger models and more computational resources, as it needs to process longer sequences of tokens.

---------


## 8) How NSP (Next Sentence Prediction) is used in Language Modelling?

**Next sentence prediction (NSP)** is used in language modeling as one-half of the training process behind the **BERT** model (the other half is masked-language modeling (MLM)). The objective of next-sentence prediction training is to predict **whether one sentence logically follows the other sentence presented to the model**.

During training, the model is presented with pairs of sentences, some of which are consecutive in the original text, and some of which are not. The model is then trained to predict whether a given pair of sentences are adjacent or not. This allows the model to understand longer-term dependencies across sentences.

Researchers have found that without **NSP, BERT** performs worse on every single metric — so its use it’s relevant to language modeling.

-----

## 9) What type of prompts can you use in Large Language Models?

<br>
<div align="center">
  <img src="https://github.com/visheshgupta-BA/Large-Language-Model-LLM-/blob/main/Image/Screenshot%202024-03-31%20at%2020.34.58.png"/>
</div>
<br>


-------

## 10) Discuss the significance of pre-training and fine-tuning in the context of LLMs ?

### Significance of Pre-Training in LLMs
- **Capturing General Language Patterns**: LLMs are pre-trained on vast amounts of text data, enabling them to understand general language structures, contexts, and nuances.
- **Learning Contextual Representations**: They capture contextual word representations based on surrounding words in sentences and paragraphs.
- **Domain Agnostic Learning**: LLMs trained on diverse datasets can be used as a starting point for various tasks and domains.
- **Universal Embeddings**: They produce word and sentence embeddings that are contextually rich and universally applicable to a wide range of tasks.

### Signficance of Fine-Tuning in LLMs
- **Task-Specific Adaptation**: By fine-tuning LLMs on task-specific data, you can leverage the general knowledge captured during pre-training to address specific requirements of the given task or domain.
- **Accommodating Data Imbalance**: Fine-tuning allows you to rectify skewed class distributions and dataset imbalances that are common in real-world applications.
- **Context Refinement**: When fine-tuned on domain-specific data, LLMs can improve their contextual understanding and textual generation accuracy within that particular domain or task.

### Code Example: Fine-Tuning BERT for Text Classification

#### Here is the Python code:

```python
# Load pre-trained BERT model
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Prepare training data and optimizer

# Fine-tune BERT on your specific text classification task
bert_model.train()
for input_ids, attention_mask, labels in training_data:
    optimizer.zero_grad()
    outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# After fine-tuning, you can utilize the tuned BERT model for text classification
```

### Distilling LLMs
Another advanced strategy involves knowledge distillation, where a large pre-trained LLM is used to train a smaller, task-specific LLM. This approach benefits from both the broad linguistic knowledge of the large model and the precision and efficiency of the smaller model, making it useful in scenarios with limited computational resources.

