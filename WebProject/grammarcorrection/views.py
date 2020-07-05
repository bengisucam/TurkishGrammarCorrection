from django.shortcuts import render
from .apps import MyappConfig


def home_view(request):
    return render(request, "grammarcorrection/base.html")
    # return HttpResponse("<h1>Hello<h1>")


def form_view(request):
    return render(request, "grammarcorrection/forms.html")


def output_text_view(request, context):
    inp = context['input_text']
    # out = run([sys.executable, './test.py', inp],
    #           shell=False, stdout=PIPE)
    if inp == '' or inp is None:
        output = 'Lütfen geçerli bir giriş yapın!'
    else:
        token_analysis = MyappConfig.zemberek.Tokenize(MyappConfig.zemberek.NormalizeSentence(inp.strip()))
        tokList = [str(tok.content).lower() for tok in token_analysis]
        print(tokList)
        output=MyappConfig.predictor.predict(tokList)
        token_analysis = MyappConfig.zemberek.Tokenize(' '.join(output[:-1]).strip())
        tokList.clear()
        last_was_punc=False
        for token in token_analysis:
            analysis = MyappConfig.zemberek.AnalyzeWord(token.content)
            try:
                if str(analysis[0].getDictionaryItem().secondaryPos) =='ProperNoun':
                    tokList.append(str(token.content)[0].upper()+str(token.content)[1:])
                elif str(analysis[0].getDictionaryItem().secondaryPos) =='Abbreviation  ':
                    tokList.append(str(token.content).upper())
                else:
                    tokList.append(str(token.content))
            except:
                tokList.append(str(token.content))

            if not last_was_punc:
                if tokList[-1] in ['.',':','?','!']:
                    last_was_punc=True
            else:
                tokList[-1] = str(tokList[-1])[0].upper()+str(tokList[-1])[1:]
                last_was_punc=False

        output = ' '.join(tokList)
        output = str(output[0].upper()) + str(output)[1:]
    # out = run([sys.executable, '../train/train_funcs.py', inp],
    #           shell=False, stdout=PIPE)
    # output = unicode(out.stdout, "utf-8")
    # print(output)
    context = {'output_text': output,'input_text': inp}

    return render(request, 'grammarcorrection/forms.html', context)

def submit_text_view(request):
    input_text = request.POST.get("in_text")
    context = {'input_text': input_text}
    return output_text_view(request, context)
