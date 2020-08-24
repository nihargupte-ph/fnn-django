from django.core.management.base import BaseCommand, CommandError
from pages.models import FireModel
from pages.util import pipeline

class Command(BaseCommand):
    help = 'Called by external daemon listening to google pub/sub notifications. \
    Will activa te when a message in band 7 or 14 is recieved and will pass in the message as a str'

    # def add_arguments(self, parser):
    #     parser.add_argument('message', type=str)

    def handle(self, *args, **options):
        pipeline.pipeline()