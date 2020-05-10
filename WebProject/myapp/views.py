from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def hi(request):
    return render(request, "myapp/display.html")
    # return HttpResponse("<h1>Hello<h1>")
