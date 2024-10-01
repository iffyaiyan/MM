# Continual Learning 

This project aims to create a customized Continuous Learning System. It includes data processing, model fine-tuning, and inference components.

The model training approach used a BERT-based architecture for Named Entity Recognition
(NER) tasks, implemented with PyTorch and the Transformers library. The training process
followed a continual learning paradigm, where the model was sequentially trained on three
datasets (G1, G2, G3) while trying to maintain performance on previous tasks.

## Key components of the training process:
1. Data Preparation:
  - Custom NERDataset class to handle data loading and preprocessing
  - Tag conversion to BIO format
  - Token alignment with BERT tokenizer
2. Model Architecture:
  - BERT-based token classification model (BertForTokenClassification)
3. Training Loop:
 - Sequential training on G1, G2, G3 datasets
 - Multiple epochs per task (7 in the final run)
 - AdamW optimizer

## Evaluation:
F1 score calculation for each entity type and weighted average



## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

## License

[MIT License](https://opensource.org/licenses/MIT)
