# Generated by Django 3.0.3 on 2020-08-04 22:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0002_firemodel'),
    ]

    operations = [
        migrations.AlterField(
            model_name='firemodel',
            name='image',
            field=models.ImageField(upload_to='fires'),
        ),
    ]
