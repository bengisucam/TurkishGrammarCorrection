import datetime
import logging
import random
from time import time
from typing import List

from jpype import JClass, getDefaultJVMPath, java, shutdownJVM, startJVM, JString


def mixwords(sentence: str):
    def trans(sent, word1, word2):
        return sent.replace(word1.strip(), "").replace(word2.strip(), word2.strip() + " " + word1).strip()

    words = sentence.split()
    sentences = []
    for i in range(len(words) - 1):
        sentences.append(trans(" ".join(words).strip(), words[i].strip(), words[i + 1]).strip())
    return random.choice(sentences)


class ZemberekPython(object):

    def __init__(self, pathToZemberek):
        self.jarPath: str = pathToZemberek
        self.running: bool = False

        self.morphology = None
        self.tokenizer = None
        self.spellChecker = None
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
        logging.info('- start time: {}'.format(datetime.datetime.now()))
        self.outSentences.clear()
        lines = self.dataset.read().strip().split('\n')

        self.dataset.close()
        logging.info("- there are {} lines in file.".format(len(lines)))
        last_timing=time()
        print_ = 1
        for index, line in enumerate(lines):
            builtSentence = ""
            if random.random() <= 0.25:
                line = mixwords(line.strip()).strip()
            else:
                line = line.strip()
            tokens = self.Tokenize(line)
            for token in tokens:

                newToken = self.__ApplyRules__(str(token.content))
                if newToken[0] == '-':
                    builtSentence = builtSentence.strip(" ")
                    newToken = newToken[1:]
                builtSentence += newToken + ' '
            if index > 0 and index % (int(len(lines) / 400)) == 0:
                if print_ > 0:
                    time_passed=int(time()-last_timing)
                    estimated_time_left=(399-print_)*time_passed
                    estimated_time_left_hours=str(datetime.timedelta(seconds=estimated_time_left))
                    logging.info('- %{} is done at {} -- estimated time left: {}'.format(float(print_ / 4), datetime.datetime.now(),estimated_time_left_hours)),
                else:
                    logging.info('- %{} is done at {}'.format(float(print_ / 4), datetime.datetime.now())),
                last_timing = time()
                print_ += 1
            self.outSentences.append(builtSentence.strip())
        logging.info('- end time: {}'.format(datetime.datetime.now()))

        return self

    def __ApplyRules__(self, token):
        # if str(token.content) in string.punctuation:
        #     return str(token.content)
        analysis=self.AnalyzeWord(token)
        for rule in self.rules:
            if rule.Satisfies(token, analysis):
                token = rule.Apply(token)
        return token

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
        logging.info('- adding Rules..')
        for i, rule in enumerate(rules):
            logging.info('- rule {} - {} '.format(i, rule))
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

    def Normalize(self, token):

        if not isinstance(token, str):
            token = str(token.content)
        if self.spellChecker.suggestForWord(JString(token)):
            if not self.spellChecker.check(JString(token)):
                return str(self.spellChecker.suggestForWord(JString(token))[0])

        return str(token)

    def CreateTokenizer(self):
        if not self.running:
            return None
        TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
        Token: JClass = JClass('zemberek.tokenization.Token')
        tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
            Token.Type.NewLine,
            Token.Type.SpaceTab
        ).build()
        self.tokenizer = tokenizer
        return self

    def CreateSpellChecker(self):
        TurkishSpellChecker: JClass = JClass(
            'zemberek.normalization.TurkishSpellChecker'
        )

        self.spellChecker: TurkishSpellChecker = TurkishSpellChecker(self.morphology)
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
