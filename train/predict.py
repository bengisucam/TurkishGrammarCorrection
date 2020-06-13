import logging
import sys
import pandas as pd
sys.path.append("/content/drive/My Drive/GrammarCorr/TurkishGrammarCorrection/")
from seq2seq.evaluator import Predictor


def predict(seq2seq,bilstm, input_vocab, output_vocab, testcsv, savePath,max_len=20, n=250, device='cuda'):
    frame = pd.read_csv(testcsv)
    frame = frame.sample(n, replace=True)
    predictor = Predictor(seq2seq,bilstm, input_vocab, output_vocab, device=device)
    predictions = ""
    for sample in frame.values:
        source = sample[1]
        if len(source.split(' '))>max_len:
            continue
        target = sample[2]
        prediction_source = " ".join(predictor.predict(source.strip().split(' ')))
        prediction_target = " ".join(predictor.predict(target.strip().split(' ')))
        predictions += "--------------------\n"
        predictions += "Source:     {}\nPrediction Source:  {}\nPrediction Target:  {}\nReal:         {}\n".format(source, prediction_source,prediction_target, target)
        predictions += "--------------------\n"

    with open(savePath + "/predictions.txt", mode='x', encoding='utf-8') as preds:
        preds.write(predictions)
        logging.info("- saved predictions to {}".format(savePath + "/predictions.txt"))

# checkpoint = Checkpoint.load(
#     './Experiments/lstm_512_NLLL_Adam_0.001_15_bidir - True_attention - True_update_embd - True_variable - True_nlayers - 2_schRate - 5',
#     'cpu')
# seq2seq = checkpoint.model
# input_vocab = checkpoint.input_vocab
# output_vocab = checkpoint.output_vocab
# predict(seq2seq, input_vocab, output_vocab, './Datasets/test.csv', './')
