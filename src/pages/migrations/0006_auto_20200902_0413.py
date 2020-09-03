# Generated by Django 3.0.3 on 2020-09-02 04:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0005_auto_20200827_2041'),
    ]

    operations = [
        migrations.AddField(
            model_name='firemodel',
            name='lightning_lat',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='firemodel',
            name='lightning_lon',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='firemodel',
            name='lightning_timestamp',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]