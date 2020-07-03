import sys
from pathlib import Path

from django.apps import AppConfig
sys.path.append("/content/drive/My Drive/TurkishGrammarCorrection/")
sys.path.append('C:/Users/furka/Desktop/TurkishGrammarCorrection')
sys.path.append('C:/Users/furka/Desktop/TurkishGrammarCorrection/seq2seq')
from train.train_funcs import load_for_prediction, create_predictor


class MyappConfig(AppConfig):
    name = 'grammarcorrection'
    # MODEL_PATH = Path("C:/Users/furka/Desktop/TurkishGrammarCorrection/train/Experiments/ep3")
    # BERT_PRETRAINED_PATH = Path("model/uncased_L-12_H-768_A-12/")
    zemberek,predictor=create_predictor("C:/Users/furka/Desktop/TurkishGrammarCorrection/train/Configuration/config.yaml")



