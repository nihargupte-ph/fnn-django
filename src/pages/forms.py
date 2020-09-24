from django import forms
from .models import UserModel

class UserForm(forms.Form):
    email = forms.EmailField(
        required=True,
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"input_email",
            'type':"email",
            'name':"email",
            'placeholder':"Enter email address...",
            'margin-top': '100px',
            'margin-bottom': '100px',
            }
        ),
    )
    first_name = forms.CharField(
        required=False,
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"input_first_name",
            'type':"text",
            'name':"first_name",
            'placeholder':"First Name (optional)"
            }
        ),
    )
    last_name = forms.CharField(
        required=False,
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"input_last_name",
            'type':"text",
            'name':"last_name",
            'placeholder':"Last Name (optional)"
            }
        ),
    )
    city = forms.CharField(
        required=True,
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"input_city",
            'type':"text",
            'name':"city",
            'placeholder':"City (if you want to recieve all updates enter \"everywhere\")"
            }
        ),
    )
    address = forms.CharField(
        required=False,
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"input_address",
            'type':"text",
            'name':"address",
            'placeholder':"Address (optional, you will recieve emails if fires are within 20mi (32 km) of your address)"
            }
        ),
    )
    # recieve_all = forms.BooleanField(
    #     required=False,
    #     label='Check this box to recieve updates everywhere. ',
    #     widget=forms.CheckboxInput(
    #         attrs={
    #             'name':'recieve all',
    #             'color':'#ffffff'
    #         }
    #     )
    # )
    # terms_agree = forms.BooleanField(
    #     required=True,
    #     label='''Except for loss or damages caused through gross negligence or intent, the Parties shall have no liability to each other hereunder. 
    #     It is understood by both parties that the Fire Neural Network (FNN) service is intended for informational purposes only, 
    #     it shall not be used as the only tool by emergency response services, and the Fire Neural Network Group assumes no liability arising from 
    #     the use of FNN. Check this box to indicate you agree''',
    #     widget=forms.CheckboxInput(
    #         attrs={
    #             'name':'terms and service',
    #             'color':'#ffffff',
    #         }
    #     )
    # )

class UnsubForm(forms.Form):
    email = forms.EmailField(
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"inputEmailUnsub",
            'type':"email",
            'name':"unsub-email",
            'placeholder':"Enter email address..."
            }
        ),
    )