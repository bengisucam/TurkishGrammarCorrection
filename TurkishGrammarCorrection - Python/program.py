from zemberek_python.base import ZemberekPython
from zemberek_python.rule import GrammarRule

ZEMBEREK_PATH = '../../../Google Drive/Furkan&Bengisu/zemberek-full.jar'
zemberek = ZemberekPython(ZEMBEREK_PATH)
zemberek = zemberek.startJVM().CreateTokenizer().CreateTurkishMorphology()

dataset = './xa.txt'

# print(ZemberekPython.GetTokenCount())
# zemberek.endJVM()
# exit(0)


Rules = \
    [
        GrammarRule().IfSecondMorphemeIs('ProperNoun').IfUpperCase(),
        GrammarRule().IfSecondMorphemeIs('Abbreviation').IfUpperCase(),
        GrammarRule().IfContentIs(['de', 'da']).IfFirstMorphemeIs('Conj').ChangeTo(['-de', '-da'])
            .AddDescription("Ayri yazılan de/da'yi birlestir"),
        GrammarRule().IfContentIs(['ki']).IfFirstMorphemeIs('Conj').ChangeTo(['-ki'])
            .AddDescription("Ayri yazılan ki'yi birlestir"),
        GrammarRule().IfLastMorphemeIs('Loc').IfContentEndsWith(['de', 'da', 'te', 'ta']).ChangeTo(
            [' de', ' da', ' de', ' da'])
            .AddDescription("Birlesik yazilan -de/-da'nın basina space ekle"),
        GrammarRule().IfLastMorphemeIs('Conj').IfContentEndsWith(['ki']).ChangeTo([' ki'])
            .AddDescription("Birlesik yazılan ki'yi[Conj olan] space+ki'ye cevir"),
        GrammarRule().IfLastMorphemeIs('Adj').IfContentEndsWith(['ki']).ChangeTo([' ki'])
            .AddDescription("Birlesik yazılan ki'yi[Adj olan] space+ki'ye cevir"),

    ]
zemberek.AddRules(Rules)

zemberek = zemberek.open(dataset).process().write('./xac_out.txt', mode='w')

zemberek.endJVM()
