from django import forms


class InputForm(forms.Form):
     messages = forms.CharField(label='入力文',max_length=255,
     min_length=1,widget=forms.Textarea(attrs=
     {'id': 'messages','size':'100', 'placeholder':'ここに判定したい文章を入力してください\n'})
     )
     