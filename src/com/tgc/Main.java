package com.tgc;

import zemberek.core.logging.Log;
import zemberek.morphology.TurkishMorphology;
import zemberek.morphology.analysis.AnalysisFormatters;
import zemberek.morphology.analysis.SingleAnalysis;
import zemberek.morphology.analysis.WordAnalysis;
import zemberek.tokenization.Token;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Main {

    private final String ki_tag="Loc|Rel→Adj";
    private final String de_tag="Loc";


    public static void main(String[] args) throws IOException {




        AppUtilities utilities =new AppUtilities();
        BufferedReader br=utilities.ReadFile("./test_dataset.txt");

        String st;
        int lineIndex=0;
        StringBuilder memoryOfDataset= new StringBuilder();
        while ((st = br.readLine()) != null)
        {
            if(lineIndex==0)
            {
                memoryOfDataset.append(st).append("\n");
                lineIndex++;
                continue;
            }
            List<String> sentences=utilities.CreateSentences(st);
            int sentenceIndex=-1;
            for(String sentence : sentences)
            {
                sentenceIndex++;
                StringBuilder memoryOfSentence= new StringBuilder(sentence);
                int extraIndex=0;

                List<Token> tokens=utilities.ExtractTokens(sentence);
                int tokenIndex=-1;
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
                        int tokenStart=t.start;
                        int tokenEnd=t.end;


                        //ayrı yazılan de/da
                        if(t.content.equals("de")||t.content.equals("da"))
                        {
                            if(morphemes.contains("Conj"))
                            {
                                if(tokenIndex!=0)
                                {
                                    memoryOfSentence.replace(tokenStart-1+extraIndex,tokenEnd+1+extraIndex,t.content);
                                    extraIndex-=1;
                                    break;

                                }
                            }

                        }
                        //ayrı yazılan ki
                        else if(t.content.equals("ki"))
                        {
                            if(morphemes.contains("Conj"))
                            {
                                memoryOfSentence.replace(tokenStart-1+extraIndex,tokenEnd+1+extraIndex,t.content);
                                extraIndex-=1;
                                break;
                            }
                        }
                        //birleşik yazılan ki
                        else if(morphemes.get(morphemes.size()-1).equals("Loc|Rel→Adj")&&sentence.substring(tokenEnd-1+extraIndex,tokenEnd+extraIndex+1).equals("ki"))
                        {
                            tokenStart=tokenEnd-1;
                            memoryOfSentence.replace(tokenStart+extraIndex,tokenEnd+extraIndex+1," ki");
                            extraIndex+=1;
                            break;
                        }
                        else if(morphemes.get(morphemes.size()-1).equals("Loc")&&!utilities.ignoreWordsDeDa .contains(t.content.toLowerCase()))
                        {
                            String morph=sentence.substring(tokenEnd-1+extraIndex,tokenEnd+extraIndex+1);
                            if(morph.equals("de")||morph.equals("da")||morph.equals("te")||morph.equals("ta"))
                            {
                                tokenStart=tokenEnd-1;
                                System.out.println(sentence);

                                memoryOfSentence.replace(tokenStart+extraIndex,tokenEnd+extraIndex+1," "+morph);
                                System.out.println(memoryOfSentence);

                                extraIndex+=1;
                                break;

                            }

                        }



                    }
                }

            }
            lineIndex++;
        }
    }
}
