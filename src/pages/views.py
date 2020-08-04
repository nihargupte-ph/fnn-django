from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.contrib import messages
from .forms import EmailForm
import requests
import json

# Create your views here.
class HomeView(TemplateView):
    template_name = "home.html"

    def get(self, request):
        form = EmailForm()
        return render(request, self.template_name, {'form':form})

    def post(self, request):
        # Getting token 
        captcha_token = request.POST.get('g-recaptcha-response')
        cap_url = "https://www.google.com/recaptcha/api/siteverify"
        cap_secret = "6LeOJ7oZAAAAAOBUuZo2wiskY0Ut-sxG83Wa4PUJ"
        cap_data = {'secret':cap_secret, 'response':captcha_token}
        # Sending request to google API to verify tokens
        cap_server_response = requests.post(url=cap_url, data=cap_data)
        cap_json = json.loads(cap_server_response.text)
        if cap_json['success'] == False:
            messages.error(request, "Invalid Captcha Try Again")
            return_success = False
            return HttpResponseRedirect("/home")
            


        form = EmailForm(request.POST)
        blank_form = EmailForm()
        if form.is_valid():
            form.save()
            user_email = form.cleaned_data['email']
            messages.success(request, f"You will recieve emails at {user_email} from fireneuralnetwork@gmail.com. Thanks for signing up! ")
            return render(request, self.template_name, {'form':blank_form, 'user_email':user_email})
        else:
            messages.success(request, f"You have already signed up! You will recieve emails at {request.POST['email']} from fireneuralnetwork@gmail.com. Thanks for signing up! ")
            
        return render(request, self.template_name, {'form':blank_form, 'invalid':True})


def howitworks_view(request):
    context = {}
    return render(request, "howitworks.html", context)