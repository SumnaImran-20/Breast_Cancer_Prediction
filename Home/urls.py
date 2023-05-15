from django.contrib import admin
from django.urls import path
from Home import views

urlpatterns = [
     path("",views.index, name = 'home'),
     path("predict",views.predict, name = 'predict'),
     path("result",views.result, name = 'result')
]