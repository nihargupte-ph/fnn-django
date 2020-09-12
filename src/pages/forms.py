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
    address = forms.CharField(
        required=False,
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"input_address",
            'type':"text",
            'name':"address",
            'placeholder':"Address (optional, you will recieve emails if fires are within 20km of your address)"
            }
        ),
    )
    city = forms.CharField(
        required=False,
        label='',
        widget=forms.TextInput(
            attrs={
            'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
            'id':"input_city",
            'type':"text",
            'name':"city",
            'placeholder':"City (optional)"
            }
        ),
    )


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