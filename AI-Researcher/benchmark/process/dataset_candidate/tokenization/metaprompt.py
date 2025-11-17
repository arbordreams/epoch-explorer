TASK = "Develop and evaluate tokenization methods for low-resource languages using meta-learning approaches. The focus is on rapid adaptation to new languages with minimal training data."

DATASET = r"""
For tokenization tasks, you should select from low-resource language datasets. Common options include:
- UD (Universal Dependencies) datasets for low-resource languages
- FLORES (Few-shot Learning with Multilingual Representations) datasets
- Language-specific corpora from the OPUS project
- Custom low-resource language datasets

You can download datasets from:
- Universal Dependencies: https://universaldependencies.org/
- FLORES: https://github.com/facebookresearch/flores
- OPUS: https://opus.nlpl.eu/

The dataset should contain text in the target low-resource language(s) for tokenization evaluation.
"""

BASELINE = r"""
• BPE (Byte Pair Encoding): Neural Machine Translation of Rare Words with Subword Units [1]
• SentencePiece: A simple and language independent subword tokenizer [2]
• Unigram Language Model: Subword Regularization: Improving Neural Network Translation Models [3]
• WordPiece: Google's Neural Machine Translation System [4]
• Meta-learning approaches: Model-Agnostic Meta-Learning (MAML) [5]
• Cross-lingual tokenization: Cross-lingual Language Model Pretraining [6]
• Few-shot tokenization: Few-Shot Learning with Meta-Learning [7]

References:
[1] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. ACL.
[2] Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. EMNLP.
[3] Kudo, K. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. ACL.
[4] Wu, Y., et al. (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.
[5] Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. ICML.
[6] Conneau, A., & Lample, G. (2019). Cross-lingual language model pretraining. NeurIPS.
[7] Finn, C., & Levine, S. (2018). Meta-learning and universality: Deep representations and gradient descent can approximate any learning algorithm. ICLR.
"""

COMPARISON = r"""
Evaluation metrics for tokenization include:
- Tokenization efficiency (vocabulary size, compression ratio)
- Downstream task performance (perplexity, BLEU, accuracy)
- Adaptation speed (number of examples needed)
- Cross-lingual transfer performance

Compare against baseline tokenization methods on:
1. Tokenization quality (vocabulary coverage, OOV rate)
2. Few-shot adaptation performance (performance with <1000 examples)
3. Cross-lingual transfer (performance on related languages)
4. Computational efficiency (training time, inference speed)
"""

EVALUATION = r"""
Key evaluation metrics:
- **Vocabulary Coverage**: Percentage of test tokens covered by the tokenizer vocabulary
- **OOV Rate**: Out-of-vocabulary rate on test data
- **Perplexity**: Language modeling perplexity using the tokenized vocabulary
- **Downstream Task Performance**: Performance on tasks like machine translation, named entity recognition, or part-of-speech tagging
- **Adaptation Efficiency**: Number of training examples required to reach baseline performance
- **Cross-lingual Transfer**: Performance on related languages without additional training

Evaluation should be performed on held-out test sets from the target low-resource language(s).
"""

REF = r"""
For tokenization evaluation, you can refer to:
- SentencePiece repository and documentation
- HuggingFace tokenizers library
- Meta-learning implementations (MAML, Reptile)
- Cross-lingual NLP benchmarks (XTREME, FLORES)

Standard tokenization libraries and tools are available in common NLP frameworks.
"""
