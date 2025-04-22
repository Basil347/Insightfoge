from django.db import models
import json

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    independent_columns = models.TextField(null=True, blank=True)  # Changed from JSONField
    target_column = models.CharField(max_length=255, null=True, blank=True)
    column_selection_complete = models.BooleanField(default=False)
    saved_model = models.FileField(upload_to='saved_models/', null=True, blank=True)
    model_type = models.CharField(max_length=100, null=True, blank=True)
    encoding_mappings = models.TextField(null=True, blank=True)  # New field to store encoding mappings

    def __str__(self):
        return self.name

    def set_independent_columns(self, columns):
        if isinstance(columns, str):
            # Split the string if it contains commas
            columns = columns.split(',') if ',' in columns else [columns]
        self.independent_columns = json.dumps(columns)

    def get_independent_columns(self):
        if not self.independent_columns:
            return []
        try:
            columns = json.loads(self.independent_columns)
            # If we get a list with a single comma-separated string, split it
            if len(columns) == 1 and isinstance(columns[0], str) and ',' in columns[0]:
                return columns[0].split(',')
            return columns
        except json.JSONDecodeError:
            return []

    def set_encoding_mappings(self, mappings):
        """Save encoding mappings as JSON."""
        self.encoding_mappings = json.dumps(mappings)

    def get_encoding_mappings(self):
        """Retrieve encoding mappings as a dictionary."""
        if not self.encoding_mappings:
            return {}
        try:
            return json.loads(self.encoding_mappings)  # Deserialize JSON string to dictionary
        except json.JSONDecodeError:
            return {}  # Return an empty dictionary if decoding fails

    def remove_independent_column(self, column):
        """Remove a column from the independent_columns list."""
        columns = self.get_independent_columns()
        if column in columns:
            columns.remove(column)
            self.set_independent_columns(columns)
            self.save()

class ExampleDataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='example_datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    independent_columns = models.TextField(null=True, blank=True)
    target_column = models.CharField(max_length=255, null=True, blank=True)
    column_selection_complete = models.BooleanField(default=False)
    saved_model = models.FileField(upload_to='saved_models/', null=True, blank=True)
    model_type = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return self.name