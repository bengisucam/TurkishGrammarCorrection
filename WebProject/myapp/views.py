from django.shortcuts import render
from django.http import HttpResponse

import sys
from subprocess import run, PIPE


def home_view(request):
    print("in home_view")
    return render(request, "myapp/base.html")
    # return HttpResponse("<h1>Hello<h1>")


def form_view(request):
    print("in form_view")
    return render(request, "myapp/forms.html")


def output_text_view(request, context):
    print("in output_view")
    inp = context['input_text']
    print(context)
<<<<<<< HEAD
    # print(output_text)
    # request.GET.get(output_text)

    # inp = request.POST.get('out_text')
    out = run([sys.executable, './test.py', inp],
              shell=False, stdout=PIPE)
=======
    # out = run([sys.executable, './test.py', inp],
    #           shell=False, stdout=PIPE)

    out = run([sys.executable, '../train/train_char.py', inp],
              shell=False, stdout=PIPE)

>>>>>>> 0ed91d301b6398d7a696bdc7cc025fd4abd31058
    print(out.stdout)
    context = {'output_text': out.stdout}

    return render(request, 'myapp/forms.html', context)

<<<<<<< HEAD

=======
>>>>>>> 0ed91d301b6398d7a696bdc7cc025fd4abd31058
def submit_text_view(request):
    print("in submit_view")
    text = ''
    input_text = request.POST.get("in_text")
    # text += input_text
    print(input_text)
    context = {'input_text': input_text}
    return output_text_view(request, context)
