# Generated by Django 4.1 on 2023-01-16 07:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detection_app", "0002_rename_image_info_image_1_info_image_2"),
    ]

    operations = [
        migrations.AddField(
            model_name="info",
            name="slug",
            field=models.CharField(default="", max_length=130),
        ),
    ]
