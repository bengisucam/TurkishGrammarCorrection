import logging

from zemberek_python.base import ZemberekPython
from zemberek_python.rule import GrammarRule

logging.basicConfig(level=logging.INFO)

ZEMBEREK_PATH = '../data/zemberek/zemberek-full.jar'
zemberek = ZemberekPython(ZEMBEREK_PATH)
zemberek = zemberek.startJVM().CreateTokenizer().CreateTurkishMorphology().CreateSpellChecker()

frequentlyMiswrittenWords = ['birçok', 'poğaça', 'yalnız', 'birkaç', 'yanlış']
miswrittenStates = ['bir çok', 'poğça', 'yanlız', 'bir kaç', 'yalnış']

soru_ekleri = ['mıyım', 'miyim', 'muyum', 'müyüm',
               'musun', 'müsün', 'mısın', 'misin',
               'mu', 'mü', 'mı', 'mi',
               'muyuz', 'müyüz', 'mıyız', 'miyiz',
               'musunuz', 'müsünüz', 'mısınız', 'misiniz',
               ]
soru_ekleri_Target = ['-' + a for a in soru_ekleri]
Rules = \
    [
        GrammarRule().IfContentIs(['şey']).ChangeTo(['-şey']).AddDescription("Şey'leri ayır"),
        GrammarRule().IfContentIs(frequentlyMiswrittenWords).ChangeTo(miswrittenStates).AddDescription(
            "Sıklıkla hata yapılan kelimeler"),
        # GrammarRule().IfContentIs(['?', '!', '\x92']).IfPrimaryPosIs('Punc').ChangeTo(['-', '-', '-']).WithChance(
        #     1).AddDescription(
        #     "Soru ve ünlem işaretlerini sil"),
        # GrammarRule().IfContentIs([',']).IfPrimaryPosIs('Punc').ChangeTo(['-']).WithChance(
        #     1.0).AddDescription("virgül sil"),

        GrammarRule().IfContentIs(soru_ekleri).
            ChangeTo(soru_ekleri_Target)
            .AddDescription("Ayri yazılan de/da'yi birlestir"),
        # GrammarRule().IfUpperCase().Lower().AddDescription("Büyük harfleri küçült"),
        # GrammarRule().IfSecondaryPosIs('ProperNoun').IfUpperCase().Remove('\'').AddDescription(
        #     "Kesme işaretlerini sil"),
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


def preprocess(document):
    return document


def postprocess(document):
    return document.lower()


for file in ['xaa_lower_stripped.txt','xab_lower_stripped.txt','xac_lower_stripped.txt','xad_lower_stripped.txt']:
    zemberek.open(path=f'../data/newscor/raw/lower/{file}').process(min_tok=1, max_tok=11,
                        ).write(
        f'../data/newscor/processed/source_{file}', mode='x',
        write_target=True, target_path=f'../data/newscor/processed/target_{file}')

zemberek.endJVM()
