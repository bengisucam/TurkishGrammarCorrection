import string
from datetime import datetime
from typing import List

from jpype import JClass, getDefaultJVMPath, java, shutdownJVM, startJVM, JString


class ZemberekPython(object):

    def __init__(self, pathToZemberek):
        self.jarPath: str = pathToZemberek
        self.running: bool = False

        self.morphology = None
        self.tokenizer = None
        self.dataset = None

        self.rules: List[GrammarRule] = []

        self.outSentences = []

    def open(self, path: str):
        self.dataset = open(path, encoding='utf-8', mode='r')
        return self

    @staticmethod
    def GetTokenCount(path: str) -> int:
        TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
        tokenizer: TurkishTokenizer = TurkishTokenizer.builder().acceptAll().build()
        with open(path, encoding='utf-8', mode='r') as file:
            print("Opening file..")
            data = file.read()
            print("Finished reading..")
            count = len(tokenizer.tokenize(data))
            file.close()
        return count

    def write(self, filePath: str, mode='w'):
        with open(filePath, mode=mode, encoding="utf-8") as outfile:
            outfile.write("\n".join(self.outSentences))
            outfile.close()
        return self

    def process(self):
        print('Start time: {}'.format(datetime.now()))
        self.outSentences.clear()
        lines = self.dataset.readlines()
        print("There are {} lines in file.".format(len(lines)))
        for index, line in enumerate(lines):
            builtSentence = ""
            line = line.strip()
            tokens = self.Tokenize(line)
            for token in tokens:
                newToken = self.__ApplyRules__(token)
                if newToken[0] == '-':
                    builtSentence = builtSentence.strip(" ")
                    newToken = newToken[1:]
                builtSentence += newToken + ' '
            if index == int(len(lines)/2):
                print('%50 is done..')
            self.outSentences.append(builtSentence.strip())
        print('End time: {}'.format(datetime.now()))

        return self

    def __ApplyRules__(self, token):
        if str(token.content) in string.punctuation:
            return str(token.content)
        for rule in self.rules:
            tokenAsStr = str(token.content)
            if rule.Satisfies(token, self.AnalyzeWord(tokenAsStr)):
                return rule.Apply(tokenAsStr)
        return str(token.content)

    def startJVM(self):
        try:
            startJVM(
                getDefaultJVMPath(),
                '-ea',
                f'-Djava.class.path={self.jarPath}',
                convertStrings=False
            )
            self.running = True
            return self
        except OSError as oserr:
            print("JVM is already running or it can not be started.\n{}".format(oserr))

    def AddRule(self, rule):
        self.rules.append(rule)

    def AddRules(self, rules):
        print('Adding Rules..')
        for i, rule in enumerate(rules):
            print('Rule {} - {} '.format(i, rule))
            self.AddRule(rule)

    def endJVM(self):
        if self.running:
            shutdownJVM()
        if self.dataset is not None:
            self.dataset.close()

    def CreateTurkishMorphology(self):
        if not self.running:
            return None
        TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
        self.morphology = TurkishMorphology.createWithDefaults()
        return self

    def CreateTokenizer(self):
        if not self.running:
            return None
        TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
        Token: JClass = JClass('zemberek.tokenization.Token')
        tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
            Token.Type.Punctuation,
            Token.Type.NewLine,
            Token.Type.SpaceTab
        ).build()
        self.tokenizer = tokenizer
        return self

    def Tokenize(self, sentence: str):
        return self.tokenizer.tokenize(JString(sentence))

    def AnalyzeSentence(self, sentence: str):
        analysis: java.util.ArrayList = (
            self.morphology.analyzeAndDisambiguate(sentence).bestAnalysis()
        )
        return analysis

    def AnalyzeWord(self, word: str):
        WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')
        results: WordAnalysis = self.morphology.analyze(word)
        return results.getAnalysisResults()


class GrammarRule(object):
    def __init__(self):
        self.contents: List[str] = []
        self.firstMorpheme: str = ""
        self.lastMorpheme: str = ""
        self.replacements: List[str] = []
        self.endings: List[str] = []

        self.CheckFirstMorpheme = False
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
        if self.CheckFirstMorpheme:
            index = self.contents.index(tokenStr)
            return self.replacements[index]
        if self.CheckEnding:
            for i, ending in enumerate(self.endings):
                if tokenStr.endswith(ending):
                    l = len(ending)
                    length = len(tokenStr)
                    # TODO:Türkiye’de -> Türkiye de?
                    base = tokenStr[:length - l].strip('’\'')
                    return base + self.replacements[i]

    def ChangeTo(self, changeList):
        self.replacements = changeList
        return self

    def AddDescription(self, desc):
        self.RuleDescription = desc
        return self

    def __str__(self):
        return "Description :{}".format(self.RuleDescription)
