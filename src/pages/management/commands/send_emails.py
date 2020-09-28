import itertools

from django.core.management.base import BaseCommand, CommandError
from pages.models import UserModel
from pages.util.misc_functions import geodesic_point_buffer
from django.core.mail import EmailMessage


class Command(BaseCommand):
    help = 'Django command for sending emails to interested parties'

    def add_arguments(self, parser):
        parser.add_argument('lon', type=float)
        parser.add_argument('lat', type=float)
        parser.add_argument('timestamp', type=str)
        parser.add_argument('fire_id', type=str)

    def handle(self, *args, **options):
        lon = round(options['lon'], 9)
        lat = round(options['lat'], 9)
        time = f"{options['timestamp']} US/Pacific"
        link = f"www.fireneuralnetwork.com/firedetail/{options['fire_id']}"

        msg = f"""
            
            Longitude: {lon}
            Latitude:  {lat}
            Timestamp: {time}
            Fire Details Link: {link}

            
            If you would like to unsubscribe please go to : www.fireneuralnetwork.com/emailunsub/
            """

        shp = geodesic_point_buffer(lat=lat, lon=lon, km=32)
        shp = shp['coordinates'][0]

        min_lon = min([i[0] for i in shp])
        max_lon = max([i[0] for i in shp])
        min_lat = min([i[1] for i in shp])
        max_lat = max([i[1] for i in shp])

        queried_location_users = UserModel.objects.filter(longitude__lt=max_lon, longitude__gt=min_lon, latitude__lt=max_lat, latitude__gt=min_lat)
        no_location_users = UserModel.objects.filter(latitude__isnull=True, longitude__isnull=True)
        queried_users = list(itertools.chain(queried_location_users, no_location_users))

        email_lst = [email_model.email for email_model in queried_users]
        message = EmailMessage(
            subject='New fire detected',
            body=msg, 
            from_email='info@fireneuralnetwork.com',
            bcc=email_lst
        )
        message.send()