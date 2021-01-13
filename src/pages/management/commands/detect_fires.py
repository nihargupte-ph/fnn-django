from django.core.management.base import BaseCommand, CommandError
from pages.models import FireModel
from django.core.mail import send_mail
import datetime

from pages.util import pipeline

class Command(BaseCommand):
    help = 'Django command for detecting fires in the background'

    def add_arguments(self , parser):
        parser.add_argument('--pipeline', type=str)

    def handle(self, *args, **options):
        switcher = {
            'brazil': pipeline.pipeline_brazil,
            'california': pipeline.pipeline_california
        }
        func = switcher.get(options['pipeline'], lambda: "Invalid value")
        func()
        # send_mail(
        # 'Pipeline ended',
        # f'Pipeline ended at {datetime.datetime.now()}',
        # 'fireneuralnetwork@gmail.com',
        # ['gupten8@gmail.com'],
        # fail_silently=False,
        # )
        print('pipeline ended, email sent to super user')