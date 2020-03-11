package com.tgc;

import zemberek.morphology.analysis.SingleAnalysis;
import zemberek.morphology.analysis.WordAnalysis;
import zemberek.tokenization.Token;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {

        AppUtilities utilities =new AppUtilities();
        BufferedReader br=utilities.ReadFile("./test_dataset - COPY.txt");

        List<String> wordListDeDa = new ArrayList<>();
        List<String> wordListKi = new ArrayList<>();

        String st;
        while ((st = br.readLine()) != null)
        {
            List<String> sentences=utilities.CreateSentences(st);
            for(String sentence : sentences)
            {
                List<Token> tokens=utilities.ExtractTokens(sentence);
                int tokenIndex=-1;
                // t -> word and word index in the sentence
                for(Token t : tokens)
                {

                    tokenIndex++;
                    if(!t.type.equals(Token.Type.Word))
                        continue;
                    //words
                    WordAnalysis results = utilities.morphology.analyze(t);
                    List<SingleAnalysis> analysisList=results.getAnalysisResults();

                    for(SingleAnalysis s: analysisList)
                    {
                        List<String> morphemes= Arrays.asList(s.formatMorphemesLexical().split("\\+"));
                        if(t.content.equals("de") || t.content.equals("da") || t.content.equals("te") || t.content.equals("ta"))
                        {
                            wordListDeDa.add(tokens.get(tokenIndex-1).getText() + t.content);
//                            System.out.println(tokens.get(tokenIndex-1).content+" - "+t.content);
//                            System.out.println(t.content);

                           if(morphemes.contains("Conj")){
                               System.out.println(tokens.get(tokenIndex-1).content+" - "+t.content);

                           }

                        }
                        else if (t.content.equals("ki")){
                            wordListKi.add(tokens.get(tokenIndex-1).getText() + t.content);

                            if(morphemes.contains("Rel->Adj")){
                                System.out.println(tokens.get(tokenIndex-1).content+" - "+t.content);

                            }
                        }
                        else if(!morphemes.get(morphemes.size()-1).equals("Loc"))
                            continue;
                        else if(utilities.IgnoreWordsDeDa.contains(t.content.toLowerCase()))
                            continue;

                    }
                }

            }
        }

        System.out.println(wordListDeDa);
        System.out.println(wordListKi);
    }
}
