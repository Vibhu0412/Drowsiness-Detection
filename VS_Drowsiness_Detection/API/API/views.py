from django.shortcuts import render
from django.http import HttpResponse

def LandingPage(request):
    context = {}
    return render(request, 'home.html', context)
