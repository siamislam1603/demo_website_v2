# Generated by Django 4.1 on 2023-01-16 07:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Quiz", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="question",
            name="image",
            field=models.ImageField(blank=True, null=True, upload_to="images/quiz"),
        ),
        migrations.AddField(
            model_name="question",
            name="slug",
            field=models.CharField(default="", max_length=130),
        ),
    ]
