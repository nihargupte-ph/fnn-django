# Generated by Django 3.0.3 on 2020-09-02 23:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0008_firemodel_video'),
    ]

    operations = [
        migrations.AlterField(
            model_name='firemodel',
            name='video',
            field=models.FileField(blank=True, null=True, upload_to='fires_gifs'),
        ),
    ]
