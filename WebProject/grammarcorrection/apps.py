import sys

from django.apps import AppConfig
sys.path.append("/content/drive/My Drive/TurkishGrammarCorrection/")
# sys.path.append('C:/Users/furka/Desktop/TurkishGrammarCorrection')
# sys.path.append('C:/Users/furka/Desktop/TurkishGrammarCorrection/seq2seq')
sys.path.append('C:/Users/bengi/PycharmProjects/TurkishGrammarCorrection')
sys.path.append('C:/Users/bengi/PycharmProjects/TurkishGrammarCorrection/seq2seq')
from train.train_funcs import  create_predictor


class MyappConfig(AppConfig):
    name = 'grammarcorrection'
    # zemberek,predictor=create_predictor("C:/Users/furka/Desktop/TurkishGrammarCorrection/train/Configuration/config.yaml")
    zemberek, predictor = create_predictor(
        "C:/Users/bengi/PycharmProjects/TurkishGrammarCorrection/train/Configuration/config.yaml")



