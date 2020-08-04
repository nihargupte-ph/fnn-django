from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
class EmailModel(models.Model):
    email = models.EmailField(max_length=100, unique=True)

class FireModel(models.Model):
    # Choices
    LIG = "LIGHTNING"
    MAN = "MANMADE"
    UNK = "UNKOWN"

    VH = "VERY HIGH"
    H = "HIGH"
    M = "MEDIUM"
    L = "LOW"

    fire_id              = models.IntegerField(unique=True)
    name                 = models.CharField(max_length=50, null=True, blank=True)
    latitude             = models.FloatField(validators=[MinValueValidator(-90.0), MaxValueValidator(90.0)])
    longitude            = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(180.0)])
    timestamp            = models.TimeField()
    image                = models.ImageField(upload_to='fires', null=True, blank=True)
    cause                = models.CharField(max_length=50, choices=((LIG, 'lightning'), (MAN, 'man-made'), (UNK, 'unknown'),), default='unknown')
    short_description    = models.CharField(max_length=200, null=True, blank=True)
    long_description     = models.CharField(max_length=1000, null=True, blank=True)
    probability          = models.CharField(max_length=10, choices=((VH, 'very high'), (H, 'high'), (M, 'medium'), (L, 'low'), (UNK, 'unknown'),), default='unknown')

