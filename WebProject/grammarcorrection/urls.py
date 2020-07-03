from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home-page'),
    path('form', views.form_view, name='form'),
    path('form/submit', views.submit_text_view, name='submit'),
    path('form', views.home_view, name='submit'),
    path('form/submit', views.home_view, name='submit')
]