# 📊 Datasets & Benchmarks

Publicly available datasets for training, evaluation, and benchmarking machine learning models across various domains.

## 📖 Overview

Quality datasets are the foundation of successful machine learning projects. This collection provides access to curated, freely available datasets spanning image recognition, natural language processing, audio analysis, and more. Benchmarks help you evaluate and compare model performance against state-of-the-art results.

**Keywords:** datasets, machine-learning-datasets, benchmarks, open-data, training-data, image-datasets, nlp-datasets, kaggle, huggingface-datasets, computer-vision, benchmark-tasks

**Skill Levels:** 🟢 Beginner | 🟡 Intermediate | 🔴 Advanced

---

## 🌐 Major Dataset Platforms

### 🟢 Beginner-Friendly Platforms

- [Hugging Face Datasets](https://huggingface.co/datasets) **(All Levels)** - The largest hub of ready-to-use datasets for AI models with over 175,000+ datasets covering NLP, computer vision, audio, and multimodal tasks. One-line dataloaders for instant access with PyTorch, TensorFlow, JAX integration.
  - 📖 Access: Fully open, free downloads
  - 🐍 Python Library: `pip install datasets`
  - 🌍 Authority: Hugging Face (official platform)
  - [Tags: nlp-datasets computer-vision audio-datasets multimodal huggingface]

- [Kaggle Datasets](https://www.kaggle.com/datasets) **(All Levels)** - World's largest community-driven dataset platform with 100,000+ datasets spanning all machine learning domains. Features dataset usability scores, kernels/notebooks, and competitions.
  - 📖 Access: Free (requires free Kaggle account)
  - 💻 API Available: Download via Kaggle CLI
  - [Tags: kaggle datasets competitions community-datasets csv-data]

- [Papers with Code Datasets](https://paperswithcode.com/datasets) **(Intermediate/Advanced)** - Curated collection of datasets tied to research papers with benchmarks, leaderboards, and state-of-the-art results. Perfect for understanding dataset usage in cutting-edge research.
  - 📖 Access: Fully open
  - 🎓 Research-focused with code implementations
  - [Tags: research-datasets benchmarks leaderboards sota papers-with-code]

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) **(All Levels)** - Historic and well-established repository from UC Irvine with 600+ datasets for classification, regression, clustering, and more. Widely used for academic research and teaching.
  - 📖 Access: Fully open, direct downloads
  - 🎓 Authority: University of California Irvine
  - [Tags: uci classic-datasets academic tabular-data]

- [Google Dataset Search](https://datasetsearch.research.google.com/) **(All Levels)** - Google's search engine specifically for finding datasets across the web. Indexes millions of datasets from repositories, government databases, and research institutions.
  - 📖 Access: Free search tool
  - 🌍 Global dataset discovery
  - [Tags: google dataset-search data-discovery]

- **[OpenDataLab: AI Dataset Platform](https://arxiv.org/html/2407.13773v1)** 🟢 Beginner - Platform designed to bridge the gap between diverse data sources and unified data processing (June 2025). Integrates wide range of open-source AI datasets with intelligent querying and high-speed downloading. Uses next-generation AI Data Set Description Language (DSDL) for standardized multimodal data representation. Enhanced data acquisition efficiency for researchers.
  - 📖 Access: Fully open, arXiv paper + platform
  - 🏛️ Authority: arXiv research (academic)
  - 📊 Features: DSDL, intelligent search, fast downloads
  - [Tags: beginner dataset-platform multimodal open-source unified-processing 2025]

---

## 🖼️ Image & Computer Vision Datasets

### 🟢 Beginner Datasets

- [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/) - Classic dataset of 70,000 handwritten digit images (28x28 pixels), perfect for learning image classification.
  - 📖 Access: Fully open
  - [Tags: beginner mnist computer-vision image-classification]

- [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) - Popular datasets with 60,000 32x32 color images across 10 or 100 classes for object recognition.
  - 📖 Access: Fully open
  - [Tags: beginner cifar computer-vision object-recognition]

### 🟡 Intermediate/Advanced Datasets

- [ImageNet](https://image-net.org/) - Large-scale dataset with 14 million images across 20,000+ categories. Foundation for modern computer vision research.
  - 📖 Access: Registration required (free)
  - [Tags: advanced imagenet computer-vision large-scale]

- [COCO (Common Objects in Context)](https://cocodataset.org/) - 330K images with object detection, segmentation, and captioning annotations. Industry standard for vision tasks.
  - 📖 Access: Fully open
  - [Tags: intermediate coco object-detection segmentation]

---

## 📝 NLP & Text Datasets

- [The Pile](https://pile.eleuther.ai/) - 825GB diverse text corpus for language model training, created by EleutherAI.
  - 📖 Access: Fully open
  - [Tags: advanced nlp text-corpus language-models]

- [Common Crawl](https://commoncrawl.org/) - Petabytes of web crawl data available for free, used by major language models.
  - 📖 Access: Fully open
  - [Tags: advanced nlp web-data large-scale]

- [SQuAD (Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/) - 100K+ question-answer pairs on Wikipedia articles for reading comprehension tasks.
  - 📖 Access: Fully open
  - 🎓 Authority: Stanford University
  - [Tags: intermediate nlp question-answering stanford]

- **[FinePDFs: 3-Trillion-Token PDF Dataset (Hugging Face)](https://www.infoq.com/news/2025/09/finepdfs/)** 🔴 Advanced - Largest publicly available corpus built entirely from PDFs (Sept 2025). Spans 475 million documents in 1,733 languages, totaling ~3 trillion tokens. 3.65 terabytes in size. Perfect for long-context training, includes documented processing pipeline from OCR to deduplication. Open Data Commons Attribution license.
  - 📖 Access: Fully free, Hugging Face Hub
  - 🏛️ Authority: Hugging Face (official release)
  - 📊 Size: 3.65 TB, 475M documents, 3T tokens
  - 🐍 Tools: datasets, huggingface_hub, Datatrove
  - [Tags: advanced nlp pdf-corpus long-context multilingual huggingface 2025]

---

## 🎵 Audio & Speech Datasets

- [LibriSpeech](https://www.openslr.org/12/) - 1000 hours of English speech derived from audiobooks, standard for speech recognition.
  - 📖 Access: Fully open
  - [Tags: intermediate audio speech-recognition librispeech]

- [Common Voice by Mozilla](https://commonvoice.mozilla.org/en/datasets) - Multilingual speech dataset covering 100+ languages with crowd-sourced recordings.
  - 📖 Access: Fully open (CC0 license)
  - [Tags: intermediate audio multilingual speech-data]

---

## 🏆 Benchmark Leaderboards

- [Papers with Code State-of-the-Art](https://paperswithcode.com/sota) - Comprehensive leaderboards tracking best model performance across 5,000+ benchmarks and tasks.
  - 📖 Access: Fully open
  - [Tags: benchmarks leaderboards sota model-comparison]

- [SuperGLUE Benchmark](https://super.gluebenchmark.com/) - Natural language understanding benchmark suite for evaluating language models.
  - 📖 Access: Fully open
  - [Tags: nlp-benchmark language-understanding glue]

---

## 🛠️ Dataset Tools & Libraries

- [Hugging Face Datasets Library Documentation](https://huggingface.co/docs/datasets/) - Official docs for the `datasets` Python library with tutorials on loading, processing, and streaming datasets.
  - 📖 Access: Fully open
  - [Tags: documentation python-library data-loading]

- [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) - Collection of ready-to-use datasets for TensorFlow with standardized API.
  - 📖 Access: Fully open
  - [Tags: tensorflow datasets data-pipeline]

---

## 📊 Specialized & Emerging Datasets

### 🟡 Tabular Data

- **[TabPFN: Tabular Foundation Model (Nature 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)** 🟡 Intermediate - Tabular foundation model (Jan 2025, Nature publication) that outperforms all previous methods on datasets with up to 10,000 samples by wide margin using substantially less training time. Revolutionizes small-data ML in biomedical risk models, drug discovery, materials science. Game-changer for tabular data predictions.
  - 📖 Access: Fully open, Nature publication
  - 🏛️ Authority: Nature (peer-reviewed)
  - 📊 Focus: Tabular data, small datasets, foundation models
  - 💼 Applications: Healthcare, drug discovery, materials science
  - [Tags: intermediate tabular-data foundation-model small-data biomedical nature 2025]

### 📚 Curated Lists & Guides

- **[Best Public Datasets for Machine Learning in 2025 (365 Data Science)](https://365datascience.com/trending/public-datasets-machine-learning/)** 🟢 Beginner - Comprehensive curated guide (Jan 2025) to best ML datasets for beginner-to-advanced projects. Covers MNIST, Dog Breed Identification, WorldStrat (geospatial AI), MIMIC-IV (healthcare), MultiWOZ (conversational AI), and more. Includes real-world complexity, labeling quality, and relevance to 2025 challenges.
  - 📖 Access: Fully free, curated guide
  - 🏛️ Authority: 365 Data Science
  - 📊 Categories: Vision, NLP, healthcare, geospatial, conversational AI
  - [Tags: beginner curated-list dataset-guide mnist healthcare conversational-ai 2025]

---

## 🔗 Related Resources

**See also:**
- [Machine Learning Fundamentals](./machine-learning-fundamentals.md) - Understanding how to use datasets effectively
- [Deep Learning & Neural Networks](./deep-learning-neural-networks.md) - Training models with these datasets
- [Computer Vision](./computer-vision.md) - Image dataset applications
- [Natural Language Processing](./natural-language-processing.md) - Text dataset applications

**Cross-reference:**
- [AI Tools & Frameworks](./ai-tools-frameworks.md) - Libraries for working with datasets

---

## 🤝 Contributing

Found a great free dataset or benchmark? We'd love to add it!

**To contribute, use this format:**
```
- [Dataset/Benchmark Name](URL) - Clear description highlighting what data is included, size, and use cases. (Difficulty Level)
  - 📖 Access: [access details]
  - 📊 Size/Samples: [if notable]
  - [Tags: keyword1 keyword2 keyword3]
```

**Ensure all resources are:**
- ✅ Completely free to access and download (no payment required)
- ✅ Openly available or require only free registration
- ✅ High-quality with clear documentation
- ✅ Legally licensed for research/educational use
- ✅ From reputable sources or institutions

---

**Last Updated:** December 23, 2025 | **Total Resources:** 22 (+4 new)

**Keywords:** machine-learning-datasets, training-data, benchmarks, huggingface-datasets, kaggle-datasets, computer-vision, nlp-datasets, audio-datasets, imagenet, coco-dataset, mnist, papers-with-code, dataset-hub, open-data, ai-datasets, finepdfs, tabular-data, tabpfn, opendatalab, 2025