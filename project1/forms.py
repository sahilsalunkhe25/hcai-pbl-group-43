from django import forms


class CSVUploadForm(forms.Form):
    file = forms.FileField(label="Select a CSV file")
    drop_first_column = forms.BooleanField(
        label="First column is an id (ignore it)",
        required=False,
    )
