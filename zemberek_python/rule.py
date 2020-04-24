import random
import string
from typing import List

from jpype import JClass


class GrammarRule(object):

    # SecondaryPOS: JClass = JClass('zemberek.core.turkish.SecondaryPos')
    # secondaryPOS : SecondaryPOS = SecondaryPOS.ProperNoun

    def __init__(self):
        self.lower = False
        self.chance = 1.0
        self.contents: List[str] = []
        self.primaryPOS: str = ""
        self.secondaryPos: str = ""
        self.suffix: str = ""
        self.replacements: List[str] = []
        self.endings: List[str] = []

        self.RemoveChar = ""
        self.CheckCharacter = False
        self.RemoveCharacter = False

        self.CheckPrimaryPos = False
        self.checkSecondaryPos = False
        self.CheckUpperCase = False
        self.CheckContent = False
        self.checkSuffix = False
        self.CheckEnding = False

        self.ReplacementIndex = 0

        self.RuleDescription = ""
        self.Replace = False

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

    def IfSecondaryPosIs(self, morpheme="ProperNoun"):
        self.secondaryPos = morpheme
        self.checkSecondaryPos = True
        return self

    def IfPrimaryPosIs(self, morpheme="Conj"):
        self.primaryPOS = morpheme
        self.CheckPrimaryPos = True
        return self

    def Remove(self, char):
        self.RemoveChar = char
        self.CheckCharacter = True
        return self

    def IfSuffixPosIs(self, morpheme="Loc"):
        self.suffix = morpheme
        self.checkSuffix = True
        return self

    def Satisfies(self, token, analysisList) -> bool:
        if not analysisList:
            return False

        if self.CheckContent:
            if token not in self.contents:
                return False
            self.ReplacementIndex = self.contents.index(token)
        if self.CheckPrimaryPos:
            if analysisList[0].getMorphemes().get(0).id != self.primaryPOS:
                return False
        if self.checkSecondaryPos:
            if str(analysisList[0].getDictionaryItem().secondaryPos) != self.secondaryPos:
                return False
        if self.CheckUpperCase:
            if not token[0].isupper():
                return False
        if self.checkSuffix:

            lastAnalysis = analysisList[-1]
            morphemes = lastAnalysis.getMorphemes()

            lastMorph = morphemes[-1]
            pos = lastMorph.id
            if pos != self.suffix:
                return False
        if self.CheckCharacter:

            if self.RemoveChar not in token:
                return False
        if self.CheckEnding:

            truth = False
            for i, ending in enumerate(self.endings):
                if token.endswith(ending):
                    self.ReplacementIndex = i
                    truth = True
                    break

            if truth is False:
                return False

        return True

    def Apply(self, tokenStr):

        if self.lower:
            tokenStr = tokenStr.lower()
        if self.CheckCharacter:
            tokenStr = tokenStr.replace(self.RemoveChar, '')
        if self.CheckEnding:
            for i, ending in enumerate(self.endings):
                if tokenStr.endswith(ending):
                    l = len(ending)
                    length = len(tokenStr)
                    # TODO:Türkiye’de -> Türkiye de?
                    base = str(tokenStr[:length - l]).strip('\'’')
                    if str(tokenStr[:length - l]).endswith('\'') or str(tokenStr[:length - l]).endswith('’'):
                        return base + str(self.replacements[i]).lstrip()
                    else:
                        return base + self.replacements[i]
        elif self.Replace:
            if random.random() <= self.chance:
                tokenStr = self.replacements[self.ReplacementIndex]
        return tokenStr
        # lowecase ise gerçekten ProperNoun değildir öyleyse bişey yapmamalıyız
        # if self.CheckLowerCase:
        #     tokenStr = tokenStr[0].upper() + tokenStr[1:]
        #     print(tokenStr)
        #     return tokenStr

    def ChangeTo(self, changeList):
        self.replacements = changeList
        self.Replace = True
        return self

    def Lower(self):
        self.lower = True
        return self

    def WithChance(self, chance):
        self.chance = chance
        return self

    def AddDescription(self, desc):
        self.RuleDescription = desc
        return self

    def __str__(self):
        return "Description :{}".format(self.RuleDescription)
