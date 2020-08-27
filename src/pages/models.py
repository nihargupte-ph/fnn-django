from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from pages.util.misc_functions import obj_to_binfield

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

    name                 = models.CharField(max_length=50, null=True, blank=True)
    latitude             = models.FloatField(validators=[MinValueValidator(-90.0), MaxValueValidator(90.0)])
    longitude            = models.FloatField(validators=[MinValueValidator(-180.0), MaxValueValidator(180.0)])
    timestamp            = models.DateTimeField()
    latest_timestamp     = models.DateTimeField()
    image                = models.ImageField(upload_to='fires_pics', null=True, blank=True)
    cause                = models.CharField(max_length=50, choices=((LIG, 'lightning'), (MAN, 'man-made'), (UNK, 'unknown'),), default='unknown')
    short_description    = models.CharField(max_length=200, null=True, blank=True)
    long_description     = models.CharField(max_length=1000, null=True, blank=True)
    probability          = models.CharField(max_length=10, choices=((VH, 'very high'), (H, 'high'), (M, 'medium'), (L, 'low'), (UNK, 'unknown'),), default='unknown')
    anomaly_arr          = models.BinaryField()
    cluster_lst          = models.BinaryField()
    px_idx_x             = models.IntegerField(default=0)
    px_idx_y             = models.IntegerField(default=0)
    time_graph_pts       = models.BinaryField(default=None)
    pred_graph_pts       = models.BinaryField(default=obj_to_binfield([]))
    diff_graph_pts       = models.BinaryField(default=obj_to_binfield([]))
    cloud_graph_pts      = models.BinaryField(default=obj_to_binfield([]))
    actual_7_graph_pts   = models.BinaryField(default=obj_to_binfield([]))
    actual_14_graph_pts  = models.BinaryField(default=obj_to_binfield([]))