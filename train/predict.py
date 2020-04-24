import argparse
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store', dest='load_dir',
                    help='Path to model folder')
path = parser.parse_args().load_dir
checkpoint = Checkpoint.load(path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

predictor = Predictor(seq2seq, input_vocab, output_vocab)
while True:
    seq_str = input("Type in a source sequence:")
    seq = seq_str.strip().split()
    prediction = predictor.predict(seq)
    print(" ".join((prediction[:-1])))