# Generated by Django 3.0.3 on 2020-09-03 21:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0012_auto_20200903_1953'),
    ]

    operations = [
        migrations.AlterField(
            model_name='firemodel',
            name='video',
            field=models.ImageField(default='video_not_made', upload_to=''),
        ),
    ]
