import requests
import json
import numpy as np
import json

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView
from django.contrib import messages

from .forms import EmailForm, UnsubEmailForm
from .models import FireModel, EmailModel
from .util.misc_functions import binfield_to_obj, dt64_to_datetime, unnan_arr

# Create your views here.
class HomeView(TemplateView):
    template_name = "home.html"

    def get(self, request):
        form = EmailForm()
        with open("/home/n/Keys/config_fnn.json") as config_file:
            SECRET_CONFIG = json.load(config_file)
            GOOGLE_MAPS_API = SECRET_CONFIG['GOOGLE_MAPS_API']

        # Querying firemodels to find ones that were in the last 5 hours
        

        return render(request, self.template_name, {'form':form, 'GOOGLE_MAPS_API':GOOGLE_MAPS_API})

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


class FirePageView(ListView):
    model = FireModel
    template_name = 'firepage.html'
    context_object_name = 'fire_list'
    ordering = ['-timestamp']
    paginate_by = 50

class EmailUnsubscribeView(TemplateView):
    template_name = "emailunsub.html"

    def get(self, request):
        form = UnsubEmailForm()
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
            return HttpResponseRedirect("/emailunsub")

        form = UnsubEmailForm(request.POST)
        blank_form = UnsubEmailForm()
        if form.is_valid():
            user_email = form.cleaned_data['email']
            try:
                messages.success(request, f"You will no longer recieve emails from fireneuralnetwork@gmail.com.")
                email_to_delete = EmailModel.objects.get(email=user_email)
                email_to_delete.delete()
                return render(request, self.template_name, {'form':blank_form, 'user_email':user_email})
            except:
                messages.success(request, "Invalid email")
                return render(request, self.template_name, {'form':blank_form, 'invalid':True})
        else:
            messages.success(request, "Invalid email")
            return render(request, self.template_name, {'form':blank_form, 'invalid':True})

def fire_detail_view(request, pk):
    fire = FireModel.objects.get(id=pk)
    time_graph_pts = binfield_to_obj(fire.time_graph_pts)
    pred_graph_pts = binfield_to_obj(fire.pred_graph_pts)
    diff_graph_pts = binfield_to_obj(fire.diff_graph_pts)
    cloud_graph_pts = binfield_to_obj(fire.cloud_graph_pts)
    actual_7_graph_pts = binfield_to_obj(fire.actual_7_graph_pts)
    actual_14_graph_pts = binfield_to_obj(fire.actual_14_graph_pts)

    # Changing nans to closest non-nan value (also does a simple interpolation)
    diff_graph_pts = unnan_arr(np.array(diff_graph_pts))
    pred_graph_pts = unnan_arr(np.array(pred_graph_pts))
    actual_7_graph_pts = unnan_arr(np.array(actual_7_graph_pts))

    # NOTE The negative one is is weird but it works, the actual timestamp is right but the indicator keeps being 1 off
    fire_start_idx = time_graph_pts.index(min(time_graph_pts, key=lambda x: np.abs(fire.timestamp - x))) - 1

    # TODO Lightning plotting not implemented just yet
    if fire.cause == 'lightning':
        lightning_idx = time_graph_pts.index(min(time_graph_pts, key=lambda x: np.abs(fire.lightning_timestamp - x))) - 1
    else:
        lightning_idx = None
    
    # Formatting time to be displayed in javascript new Date form
    time_graph_pts = [d.strftime('%Y-%m-%d %H:%M') for d in time_graph_pts]
    
    # Creating changing bound based on cloud and non-cloud
    cloud_bound = [.17 if pt else .55 for pt in cloud_graph_pts]

    # Zipping time graph points with others
    pred_graph_pts = list(zip(time_graph_pts, pred_graph_pts))
    diff_graph_pts = list(zip(time_graph_pts, diff_graph_pts))
    actual_7_graph_pts = list(zip(time_graph_pts, actual_7_graph_pts))
    cloud_bound = list(zip(time_graph_pts, cloud_bound))

    # Creating colors based on cloud or non cloud 
    cloud_colors = []
    for pt in cloud_graph_pts:
        if pt:
            cloud_colors.append(f"rgba(66, 135, 245, 1)")
        else:
            cloud_colors.append(f"rgba(220, 227, 223, 1)")
    content = {
        'fire':fire,
        'time_pts':time_graph_pts,
        'pred_pts':pred_graph_pts,
        'diff_pts':diff_graph_pts,
        'cloud_colors': cloud_colors,
        'cloud_bound': cloud_bound, 
        'fire_start_idx':fire_start_idx,
        'actual_7_pts': actual_7_graph_pts,
        'lightning_idx':lightning_idx,
        }
    return render(request, "firedetail.html", content)

def how_it_works_view(request):
    context = {}
    return render(request, "howitworks.html", context)