package com.tgc;

import zemberek.morphology.TurkishMorphology;
import zemberek.morphology.analysis.SentenceAnalysis;
import zemberek.tokenization.Token;
import zemberek.tokenization.TurkishSentenceExtractor;
import zemberek.tokenization.TurkishTokenizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class AppUtilities {

    ArrayList<String> IgnoreWords = new ArrayList<String>() {
        {
            add("hatta");
            add("hafta");
            add("moda");
            add("lambda");
            add("usta");
            add("nerede");
            add("lada");
            add("oda");
            add("işte");
        }
    };
    public static TurkishSentenceExtractor extractor= TurkishSentenceExtractor.DEFAULT;
    public final TurkishMorphology morphology;
    public final TurkishTokenizer tokenizer;
    public AppUtilities()
    {
        morphology=TurkishMorphology.createWithDefaults();
        tokenizer = TurkishTokenizer
                .builder()
                .ignoreTypes(Token.Type.Punctuation, Token.Type.NewLine, Token.Type.SpaceTab)
                .build();

    }
    public List<Token> ExtractTokens(String input)
    {
        return tokenizer.tokenize(input);
    }
    public List<String> CreateSentences(String doc)
    {
        return  extractor.fromDocument(doc);
    }

    public SentenceAnalysis GetAnalysis(String sentence)
    {
        return morphology.analyzeAndDisambiguate(sentence);
    }

    public BufferedReader ReadFile(String path) throws FileNotFoundException {
        File file = new File(path);
       return new BufferedReader(new FileReader(file));
    }
}
