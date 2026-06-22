from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="home:index")),   
    path("home/", include("home.urls")),
    path("admin/", admin.site.urls),
    path("demos/", include("demos.urls")),
    path("project1/", include("project1.urls")),
    path("project2/", include("project2.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)