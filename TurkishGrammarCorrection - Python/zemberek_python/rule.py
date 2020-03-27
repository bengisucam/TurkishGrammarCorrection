from typing import List

from jpype import JClass


class GrammarRule(object):

    # SecondaryPOS: JClass = JClass('zemberek.core.turkish.SecondaryPos')
    # secondaryPOS : SecondaryPOS = SecondaryPOS.ProperNoun

    def __init__(self):
        self.contents: List[str] = []
        self.firstMorpheme: str = ""
        self.secondMorpheme: str = ""
        self.lastMorpheme: str = ""
        self.replacements: List[str] = []
        self.endings: List[str] = []

        self.CheckFirstMorpheme = False
        self.CheckSecondMorpheme = False
        self.CheckUpperCase = False
        self.CheckLowerCase = False
        self.CheckContent = False
        self.CheckLastMorpheme = False
        self.CheckEnding = False

        self.RuleDescription = ""

    def IfContentIs(self, contents):
        self.contents = contents
        self.CheckContent = True
        return self

    def IfContentEndsWith(self, endings):
        self.endings = endings
        self.CheckEnding = True
        return self

    def IfUpperCase(self):
        self.CheckUpperCase = True
        return self

    def IfSecondMorphemeIs(self, morpheme="ProperNoun"):
        self.secondMorpheme = morpheme
        self.CheckSecondMorpheme = True
        return self

    def IfFirstMorphemeIs(self, morpheme="Conj"):
        self.firstMorpheme = morpheme
        self.CheckFirstMorpheme = True
        return self

    def IfLastMorphemeIs(self, morpheme="Loc"):
        self.lastMorpheme = morpheme
        self.CheckLastMorpheme = True
        return self

    def Satisfies(self, token, analysisList) -> bool:
        if not analysisList:
            return False
        if self.CheckContent:
            if str(token.content) not in self.contents:
                return False
        if self.CheckFirstMorpheme:
            if analysisList[0].getMorphemes().get(0).id != self.firstMorpheme:
                return False
        if self.CheckSecondMorpheme:
            if str(analysisList[0].getDictionaryItem().secondaryPos) != self.secondMorpheme:
                return False
        if self.CheckUpperCase:
            if not str(token.content)[0].isupper():
                return False
        if self.CheckLowerCase:
            if not str(token)[1].islower():
                return False
        if self.CheckLastMorpheme:
            lastAnalysis = analysisList[-1]
            morphemes = lastAnalysis.getMorphemes()
            lastMorph = morphemes[-1]
            pos = lastMorph.id
            if pos != self.lastMorpheme:
                return False
        if self.CheckEnding:
            truth = False
            for ending in self.endings:
                if str(token.content).endswith(ending):
                    truth = True
                    break
            if truth is False:
                return False
        return True

    def Apply(self, tokenStr):
        if self.CheckUpperCase:
            tokenStr = tokenStr[0].lower() + tokenStr[1:]
            print(tokenStr)
            return tokenStr
        if self.CheckFirstMorpheme:
            index = self.contents.index(tokenStr)
            return self.replacements[index]
        if self.CheckEnding:
            for i, ending in enumerate(self.endings):
                if tokenStr.endswith(ending):
                    l = len(ending)
                    length = len(tokenStr)
                    # TODO:Türkiye’de -> Türkiye de?
                    base=str(tokenStr[:length - l]).strip('\'’')
                    if str(tokenStr[:length - l]).endswith('\'') or str(tokenStr[:length - l]).endswith('’'):
                        return base + str(self.replacements[i]).lstrip()
                    else:
                        return base + self.replacements[i]

        # lowecase ise gerçekten ProperNoun değildir öyleyse bişey yapmamalıyız
        # if self.CheckLowerCase:
        #     tokenStr = tokenStr[0].upper() + tokenStr[1:]
        #     print(tokenStr)
        #     return tokenStr

    def ChangeTo(self, changeList):
        self.replacements = changeList
        return self

    def AddDescription(self, desc):
        self.RuleDescription = desc
        return self

    def __str__(self):
        return "Description :{}".format(self.RuleDescription)
