from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def home_view(request):
    context = {}
    #return HttpResponse("<h1> hello world <h2>")
    return render(request, "home.html", context)