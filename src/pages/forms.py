from django import forms
from .models import EmailModel

class EmailForm(forms.ModelForm):
    class Meta:
        model = EmailModel
        fields = ('email',)
        widgets = {
            'email':forms.TextInput(attrs={
                'class':"form-control flex-fill mr-0 mr-sm-2 mb-3 mb-sm-0",
                'id':"inputEmail",
                'type':"email",
                'name':"sub-email",
                'placeholder':"Enter email address..."}),
        }
        labels = {
            'email':'',
        }

class UnsubEmailForm(forms.Form):
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