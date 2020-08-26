from django.core.management.base import BaseCommand, CommandError
from pages.models import EmailModel
from django.core.mail import EmailMessage


class Command(BaseCommand):
    help = 'Django command for sending emails to interested parties'

    def add_arguments(self, parser):
        parser.add_argument('lon', type=float)
        parser.add_argument('lat', type=float)
        parser.add_argument('timestamp', type=str)
        parser.add_argument('link', type=str)

    def handle(self, *args, **options):
        email_lst = [email_model.email for email_model in EmailModel.objects.all()]
        msg = f"""  \
            Longitude: {options['lon']}\n\
            Latitude:  {options['lat']}\n\
            Timestamp: {options['timestamp']}\n\
            """
        message = EmailMessage(
            subject='New fire detectied',
            body=msg, 
            from_email='fireneuralnetwork@gmail.com',
            bcc=email_lst
        )
        message.send()