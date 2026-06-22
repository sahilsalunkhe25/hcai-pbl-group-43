from django import template

register = template.Library()


@register.filter
def get(mapping, key):
    """Dict lookup by a variable key: ``{{ mydict|get:somekey }}``."""
    try:
        return mapping.get(key)
    except AttributeError:
        return None
