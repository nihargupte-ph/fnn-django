# Generated by Django 3.0.3 on 2020-08-27 20:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='firemodel',
            name='actual_14_pts',
            field=models.BinaryField(default=b'gARdlC4='),
        ),
        migrations.AddField(
            model_name='firemodel',
            name='actual_7_pts',
            field=models.BinaryField(default=b'gARdlC4='),
        ),
        migrations.AddField(
            model_name='firemodel',
            name='cloud_graph_pts',
            field=models.BinaryField(default=b'gARdlC4='),
        ),
        migrations.AddField(
            model_name='firemodel',
            name='pred_graph_pts',
            field=models.BinaryField(default=b'gARdlC4='),
        ),
        migrations.AddField(
            model_name='firemodel',
            name='time_graph_pts',
            field=models.BinaryField(default=None),
        ),
    ]
