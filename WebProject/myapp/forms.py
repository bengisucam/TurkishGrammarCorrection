from django import forms

class SimpleForm(forms.Form):
    your_text = forms.CharField(label='Your text', max_length=500)
    message = forms.CharField(
        max_length=2000,
        widget=forms.Textarea(),
        help_text='Write here your message!'
    )