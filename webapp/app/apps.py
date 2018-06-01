from django.apps import AppConfig


class AppConfig(AppConfig):
    name = 'app'


from django import forms   
class PhotoForm(forms.Form):

    image = forms.ImageField()