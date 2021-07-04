from django.shortcuts import render
from django.http import HttpResponse

def HomePage(request):
    context = {}
    return render(request, 'home_page.html', context)
