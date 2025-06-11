📚 Small Language Model to Generate Short Stories

✨ Project Overview

This project demonstrates the development of a Small Language Model (SLM) built entirely from scratch using Python, designed to generate coherent and creative short stories from user prompts. It showcases a full pipeline including data preparation, tokenization, training logic, and text generation, all inside a Jupyter notebook.

Unlike Large Language Models (LLMs) which require vast computational resources, Small Language Models are lightweight, interpretable, and cost-effective—making them suitable for edge deployment, personalization, and constrained environments.

🧬 What is a Small Language Model?

A Small Language Model (SLM) is a scaled-down version of an LLM that:

Contains fewer parameters (e.g., millions vs. billions)

Is optimized for specific or domain-limited tasks (e.g., story generation, summarization)

Requires less data, compute, and memory

Can be trained or fine-tuned locally or on limited cloud infrastructure

⚡ Why Use a Small Language Model?

Faster Inference: Ideal for real-time applications

Lower Cost: Less hardware and electricity consumption

Privacy: Can run offline or on-device

Specialization: Easier to fine-tune for niche tasks

📚 Architecture Overview

□ Conceptual Pipeline

+-------------+     +--------------------+     +---------------+     +---------------+
|  Dataset    +---> | Tokenization       +---> | Model Training +---> | Text Generation|
+-------------+     +--------------------+     +---------------+     +---------------+

🌐 Module Breakdown

1. Dataset Loader

Uses HuggingFace datasets to import TinyStories, a curated dataset of short fictional narratives.

2. Tokenizer

Utilizes tiktoken to convert raw text into numerical token IDs compatible with transformer-style models.

3. Training Logic (simplified for educational purpose)

A basic transformer block or a recurrent-style model using PyTorch

Demonstrates the logic behind input sequences and next-token prediction

4. Generation Loop

Starts with a prompt and generates n tokens step-by-step

Uses greedy decoding or sampling for varied story outcomes

🌐 Comparative Analysis

Feature

Small Language Model

Large Language Model

Params

1M - 50M

1B - 500B+

Inference Speed

Fast

Slower

Deployment

On-device, Edge, WebApps

Cloud/Data Center

Training Cost

Low

High

Use Cases

Niche/Domain Tasks

General Knowledge

Privacy

High (can run offline)

Lower (cloud dependency)

“SLMs don't aim to replace LLMs but to democratize intelligent language generation.”

📄 Project Components

Small_Language_Model.ipynb — Full notebook with:

Dataset processing

Tokenization

Training loop

Story generation function

⚙️ Setup Instructions

1. Install Dependencies

pip install datasets tiktoken torch numpy

2. Run the Notebook

jupyter notebook Small_Language_Model.ipynb

🏆 Key Takeaways

You don't need billion-parameter models to generate coherent text

Small Language Models can be practical, interpretable, and privacy-friendly

They are ideal for educational, personal assistant, and edge-AI use cases

🌐 Future Improvements

Integrate with FastAPI and Streamlit for web app interface

Replace mock model with mini-transformer architecture

Add fine-tuning capability on user stories or domain-specific corpora

Quantize model for edge devices

📄 License

This project is licensed under the MIT License.

👤 Author

Ayush Dubey

Open for collaboration on AI research, model optimization, and low-resource LLM deployment.

