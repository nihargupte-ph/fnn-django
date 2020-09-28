import pytz

from django import template

register = template.Library()

@register.filter(name='timezone_conversion_filter')
def timezone_conversion_filter(time, timezone):
    tz = pytz.timezone(timezone)
    local_dt = time.astimezone(tz)

    return local_dt.strftime("%b %e, %Y, %I:%M %p")