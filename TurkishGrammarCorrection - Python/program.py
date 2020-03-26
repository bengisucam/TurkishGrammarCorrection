from zemberek_python.base import ZemberekPython
from zemberek_python.rule import GrammarRule

ZEMBEREK_PATH = 'C:/Users/bengi/PycharmProjects/zemberek-jarfile/zemberek-full.jar'
zemberek = ZemberekPython(ZEMBEREK_PATH)
zemberek = zemberek.startJVM().CreateTokenizer().CreateTurkishMorphology()

dataset = 'C:/Users/bengi/Downloads/xa.txt'

# print(ZemberekPython.GetTokenCount())
# zemberek.endJVM()
# exit(0)


Rules = \
    [
        # GrammarRule().IfContentIs(['de', 'da']).IfFirstMorphemeIs('Conj').ChangeTo(['-de', '-da'])
        #         #     .AddDescription("Ayri yazılan de/da'yi birlestir"),
        #         # GrammarRule().IfContentIs(['ki']).IfFirstMorphemeIs('Conj').ChangeTo(['-ki'])
        #         #     .AddDescription("Ayri yazılan ki'yi birlestir"),
        #         # GrammarRule().IfLastMorphemeIs('Loc').IfContentEndsWith(['de', 'da', 'te', 'ta']).ChangeTo([' de', ' da', ' de', ' da'])
        #         #     .AddDescription("Birlesik yazilan -de/-da'nın basina space ekle"),
        #         # GrammarRule().IfLastMorphemeIs('Conj').IfContentEndsWith(['ki']).ChangeTo([' ki'])
        #         #     .AddDescription("Birlesik yazılan ki'yi[Conj olan] space+ki'ye cevir"),
        #         # GrammarRule().IfLastMorphemeIs('Adj').IfContentEndsWith(['ki']).ChangeTo([' ki'])
        #         #     .AddDescription("Birlesik yazılan ki'yi[Adj olan] space+ki'ye cevir"),
        GrammarRule().IfSecondMorphemeIs('ProperNoun').AddDescription("ProperNounların ilk harflerini büyük/küçük yap")

    ]

zemberek.AddRules(Rules)

zemberek = zemberek.open(dataset).process().write('C:/Users/bengi/Downloads/xac_out.txt', mode='x')

zemberek.endJVM()
