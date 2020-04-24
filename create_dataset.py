import argparse
import logging

from zemberek_python.base import ZemberekPython
from zemberek_python.rule import GrammarRule

parser = argparse.ArgumentParser()
parser.add_argument('--source', action='store', dest='source',
                    help='Source Path')
parser.add_argument('--target', action='store', dest='target',
                    help='Target Path')
parser.add_argument('--zemberek', action='store', dest='zemberek',
                    help='Zemberek Path')

opt=parser.parse_args()

logging.basicConfig(level=logging.INFO)

ZEMBEREK_PATH = opt.zemberek
zemberek = ZemberekPython(ZEMBEREK_PATH)
zemberek = zemberek.startJVM().CreateTokenizer().CreateTurkishMorphology().CreateSpellChecker()

dataset = opt.source

frequentlyMiswrittenWords = ['birçok', 'poğaça', 'yalnız', 'birkaç', 'yanlış']
miswrittenStates = ['bir çok', 'poğça', 'yanlız', 'bir kaç', 'yalnış']

Rules = \
    [
        GrammarRule().IfContentIs(['şey']).ChangeTo(['-şey']).AddDescription("Şey'leri ayır"),
        GrammarRule().IfContentIs(frequentlyMiswrittenWords).ChangeTo(miswrittenStates).AddDescription(
            "Sıklıkla hata yapılan kelimeler"),
        GrammarRule().IfContentIs(['?', '!']).IfPrimaryPosIs('Punc').ChangeTo(['-', '-']).WithChance(1).AddDescription(
            "Soru ve ünlem işaretlerini sil"),
        GrammarRule().IfContentIs([',', '.']).IfPrimaryPosIs('Punc').ChangeTo(['-', '-']).WithChance(
            1.0).AddDescription("Nokta ve virgül sil"),
        GrammarRule().IfUpperCase().Lower().AddDescription("Büyük harfleri küçült"),
        GrammarRule().IfSecondaryPosIs('ProperNoun').IfUpperCase().Remove('\'').AddDescription(
            "Kesme işaretlerini sil"),
        GrammarRule().IfContentIs(['de', 'da']).IfPrimaryPosIs('Conj').ChangeTo(['-de', '-da'])
            .AddDescription("Ayri yazılan de/da'yi birlestir"),
        GrammarRule().IfContentIs(['ki']).IfPrimaryPosIs('Conj').ChangeTo(['-ki'])
            .AddDescription("Ayri yazılan ki'yi birlestir"),
        GrammarRule().IfSuffixPosIs('Loc').IfContentEndsWith(['de', 'da', 'te', 'ta']).ChangeTo(
            [' de', ' da', ' de', ' da'])
            .AddDescription("Birlesik yazilan -de/-da'nın basina space ekle"),
        GrammarRule().IfSuffixPosIs('Conj').IfContentEndsWith(['ki']).ChangeTo([' ki'])
            .AddDescription("Birlesik yazılan ki'yi[Conj olan] space+ki'ye cevir"),
        GrammarRule().IfSuffixPosIs('Adj').IfContentEndsWith(['ki']).ChangeTo([' ki'])
            .AddDescription("Birlesik yazılan ki'yi[Adj olan] space+ki'ye cevir"),

    ]
zemberek.AddRules(Rules)

zemberek = zemberek.open(dataset).process().write(opt.target, mode='x')

zemberek.endJVM()