from django.shortcuts import render
from django.http import HttpResponse


def home_view(request):
    return render(request, "myapp/base.html")
    # return HttpResponse("<h1>Hello<h1>")


def form_view(request):
    return render(request, "myapp/forms.html")


def output_text_view(request, context):
    output_text = context['input_text']
    print(output_text)
    request.GET.get(output_text)
    return render(request, 'myapp/forms.html', context)


def submit_text_view(request):
    input_text = request.POST.get("input_text")
    print(input_text)
    context = {'input_text': input_text}
    return output_text_view(request, context)
