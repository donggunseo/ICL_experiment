from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer
import torch
from torch.utils.data import Dataset, DataLoader

#### Fine-tune model by myself for evaluate Confidence score of training set

