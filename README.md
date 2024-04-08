## Restaurant Menu Analysis with NLP
### Overview
This project aims to analyze restaurant menus to derive insights into how menu diversity correlates with restaurant ratings. Utilizing advanced natural language processing (NLP) techniques, it leverages both OpenAI's API and BERT models to understand and quantify menu item diversity. The project is structured around two main scripts:

1. OpenAI API Usage Script: Leverages OpenAI's powerful GPT models to analyze text data and generate embeddings for menu items.
2. BERT Analysis Script: Utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for feature extraction from menu descriptions, followed by clustering analysis.

### Project Structure
1. openai_analysis.py: Script for processing data with OpenAI's API.
2. bert_analysis.py: Script for embedding generation, clustering, and diversity analysis using BERT.

### Getting Started
Prerequisites
Python 3.6 or later
OpenAI API key (for the OpenAI-based analysis)
Access to Hugging Face's transformers library and PyTorch for BERT analysis

### Features
Data Integration and Cleanup: Merges and cleans datasets from multiple sources.
NLP Embeddings: Generates rich textual embeddings using OpenAI's API and BERT, facilitating nuanced text analysis.
Clustering and Diversity Analysis: Categorizes menu items into clusters to assess diversity and performs statistical analysis to explore its relationship with restaurant ratings.

### Contributing
Contributions, questions, and feedback are welcomed and encouraged. Please open an issue or pull request if you wish to contribute to the project.

### License
This project is released under the MIT License - see the LICENSE file for details.

