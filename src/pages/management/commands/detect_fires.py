from django.core.management.base import BaseCommand, CommandError
from pages.models import FireModel
from pages.util import pipeline

class Command(BaseCommand):
    help = 'Django command for detecting fires in the background'

    # def add_arguments(self, parser):
    #     parser.add_argument('message', type=str)

    def handle(self, *args, **options):
        pipeline.pipeline()
        print('pipeline ended')