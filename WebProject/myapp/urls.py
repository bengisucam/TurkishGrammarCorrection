from django.urls import path
from . import views

urlpatterns = [
    path('', views.hi, name='home-page'),
    path('form', views.create_form_view, name='simpleForm')
]
