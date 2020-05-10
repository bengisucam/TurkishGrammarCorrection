from django.shortcuts import render
from django.http import HttpResponse
from .forms import SimpleForm

# Create your views here.


def create_form_view(request):
    form = SimpleForm(request.POST or None)
    if form.is_valid():
        form.save()

    return render(request, "myapp/forms.html", {'form': form})



def hi(request):
    return render(request, "myapp/base.html")
    # return HttpResponse("<h1>Hello<h1>")
