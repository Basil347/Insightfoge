import json
import pandas as pd
from django.middleware.csrf import get_token
from django.template.loader import render_to_string
from django.utils.timezone import now
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .models import Dataset,ExampleDataset
import matplotlib
from prophet import Prophet
matplotlib.use('Agg')
import os
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score, recall_score, mean_absolute_error, \
    mean_squared_error


# Home Function
def Home(request):
    # Fetch all available datasets
    example_datasets = ExampleDataset.objects.all().order_by('-id')
    return render(request, 'eda/home.html', {'example_datasets': example_datasets})


def get_dataset_file(request, dataset_id):
    dataset = get_object_or_404(ExampleDataset, id=dataset_id)
    response = FileResponse(dataset.file, as_attachment=True)
    return response


# Upload Dataset Function
@csrf_exempt
def upload_dataset(request):
    if request.method == 'POST':
        dataset_name = request.POST.get('dataset_name')
        uploaded_file = request.FILES.get('file')

        if not dataset_name or not uploaded_file:
            return JsonResponse({'success': False, 'message': 'Dataset name and file are required.'})

        # Define the upload path
        upload_path = os.path.join(settings.MEDIA_ROOT, 'datasets')
        os.makedirs(upload_path, exist_ok=True)

        # Save the file using FileField
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        file_path = os.path.join(upload_path, file_name)

        # Create a Dataset instance and save the file
        dataset = Dataset(name=dataset_name)
        dataset.file.save(file_name, uploaded_file)

        # Return the dataset ID in the response
        return JsonResponse({'success': True, 'message': 'Dataset uploaded successfully!', 'dataset_id': dataset.id})

    return JsonResponse({'success': False, 'message': 'Invalid request method.'})


def analyze_dataset(request, dataset_id):
    dataset_obj = get_object_or_404(Dataset, id=dataset_id)
    file_path = dataset_obj.file.path

    try:
        data = pd.read_csv(file_path)
        rows_to_show = request.GET.get('rows', 5)
        view_type = request.GET.get('view', 'head')

        # Get head or tail based on selection
        preview_data = data.head(int(rows_to_show)) if view_type == 'head' else data.tail(int(rows_to_show))

        analysis = {
            "rows": data.shape[0],
            "columns": data.shape[1],
            "preview_data": preview_data.to_dict('records'),
            "column_names": list(data.columns),
            "null_analysis": [],
            "null_values": data.isnull().sum().to_dict(),
            "column_types": data.dtypes.apply(lambda x: str(x)).to_dict(),
            "string_columns": len(data.select_dtypes(include="object").columns),
            "numeric_columns": len(data.select_dtypes(include=["int64", "float64"]).columns),
            "unique_values": {col: data[col].nunique() for col in data.columns},
            "description": {},
            "null_recommendations": {}
        }

        # Enhanced null analysis with recommendations
        for column in data.columns:
            null_count = data[column].isnull().sum()
            if null_count > 0:
                null_rows = data[data[column].isnull()].index.tolist()
                dtype = str(data[column].dtype)

                recommendations = []
                if dtype in ['int64', 'float64']:
                    recommendations = [
                        {'method': 'Mean', 'value': data[column].mean()},
                        {'method': 'Median', 'value': data[column].median()},
                        {'method': 'Standard Deviation', 'value': data[column].std()}
                    ]
                elif dtype == 'object':
                    mode_value = data[column].mode()[0] if not data[column].mode().empty else None
                    recommendations = [
                        {'method': 'Mode', 'value': mode_value},
                        {'method': 'Most Common Value', 'value': mode_value},
                        {'method': 'Custom String', 'value': 'Unknown'}
                    ]

                analysis["null_analysis"].append({
                    "column": column,
                    "null_count": null_count,
                    "null_rows": null_rows,
                    "dtype": dtype,
                    "recommendations": recommendations
                })

        # Process description statistics
        description_data = data.describe(include="all").to_dict()
        for column, stats in description_data.items():
            analysis["description"][column] = {
                k.replace('%', ''): v for k, v in stats.items()
            }

        return render(request, "eda/analyze_results.html", {
            "analysis": analysis,
            "current_view": view_type,
            "rows_to_show": rows_to_show,
            "dataset_id": dataset_id,
        })

    except Exception as e:
        return render(request, "eda/error.html", {"error": str(e)})


# Feature Engineering

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
import pandas as pd
from .models import Dataset

import logging


def feature_engineering(request, dataset_id):
    # Retrieve the dataset object using the dataset_id
    dataset_obj = get_object_or_404(Dataset, id=dataset_id)
    file_path = dataset_obj.file.path  # Get the file path from the Dataset model

    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Check for missing values
        null_values_count = data.isnull().sum().sum()
        column_nulls = data.isnull().sum().to_dict()  # Per-column null count

        if request.method == "POST":
            action = request.POST.get("action")  # Get the user's selected action
            column = request.POST.get("column")  # Column for filling missing values

            if action == "dropna":
                # Drop rows with missing values in the specified column only
                data = data.dropna(subset=[column])  # Drop only missing values in the specified column
            elif action == "fill_mean":
                data[column] = data[column].fillna(data[column].mean())
            elif action == "fill_median":
                data[column] = data[column].fillna(data[column].median())
            elif action == "fill_std":
                data[column] = data[column].fillna(data[column].std())

            # Save the modified dataset back to the file
            data.to_csv(file_path, index=False)

            # Recalculate null values count after action
            null_values_count = data.isnull().sum().sum()

            # Send a success response
            return JsonResponse({"status": "success", "message": "Changes applied successfully!"})

        # Generate summary statistics for options
        stats = {
            col: {
                "mean": data[col].mean() if data[col].dtype != "object" else None,
                "median": data[col].median() if data[col].dtype != "object" else None,
                "std": data[col].std() if data[col].dtype != "object" else None,
            }
            for col in data.columns
        }

        # Pass necessary data to the template
        return render(request, "eda/feature_engineering.html", {
            "null_values_count": null_values_count,
            "column_nulls": column_nulls,
            "stats": stats,
            "columns": data.columns,
            "data_id":dataset_id
        })

    except Exception as e:
        return render(request, "eda/error.html", {"error": str(e)})



import matplotlib.pyplot as plt
from io import BytesIO
import base64


def vector_conversion(request, dataset_id):
    # Retrieve the dataset object based on the given dataset_id
    dataset_obj = get_object_or_404(Dataset, id=dataset_id)

    # Load the dataset from a CSV file or any other format you're using
    file_path = dataset_obj.file.path  # Assuming the file field in Dataset model holds the file path
    data = pd.read_csv(file_path)  # Assuming the dataset is in CSV format

    # Get the columns to display in the dropdown options
    columns = data.columns.tolist()

    # If the form is submitted via POST with the selected columns
    if request.method == 'POST' and request.content_type == 'application/json':
        body = json.loads(request.body)
        x_column = body.get('x_column')
        y_column = body.get('y_column')

        # Check if the columns exist in the dataset
        if x_column not in data.columns or y_column not in data.columns:
            return JsonResponse({'status': 'error', 'message': 'Invalid columns selected.'})

        # Plot the graph using the selected columns
        fig, ax = plt.subplots()
        ax.plot(data[x_column], data[y_column], label=f'{y_column} vs {x_column}')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f'Graph of {y_column} vs {x_column}')
        ax.legend()

        # Save the figure in a BytesIO buffer (to be used in the response)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Encode the plot as base64 for rendering in the frontend
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Return the labels, values, and graph as base64 to the frontend
        return JsonResponse({
            'status': 'success',
            'labels': data[x_column].tolist(),  # X-axis labels
            'values': data[y_column].tolist(),  # Y-axis values
            'image_base64': image_base64  # Graph image as base64
        })

    # If the request is a GET request, pass the columns to the template
    return render(request, 'eda/vector_conversion.html', {
        'columns': columns,'data_id':dataset_id  # Pass available columns to the template for selection
    })

import base64
import json
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import Dataset  # Assuming you have a Dataset model


@csrf_exempt
def generate_graph(request, dataset_id):
    """Generate a graph based on selected columns from a dataset."""
    if request.method == 'POST' and request.content_type == 'application/json':
        # Retrieve the dataset object using the dataset_id
        dataset_obj = get_object_or_404(Dataset, id=dataset_id)
        # Load the dataset (assuming it's a CSV file)
        file_path = dataset_obj.file.path  # Assuming the file field stores the file path
        data = pd.read_csv(file_path)

        # Parse the columns and graph type received from the AJAX call
        body = json.loads(request.body)
        x_column = body.get('x_column')
        y_column = body.get('y_column')
        graph_type = body.get('graph_type', 'line')  # Default to 'line' if no type is selected

        # Heatmap-specific logic
        if graph_type == 'heatmap':
            # Compute the correlation matrix
            corr_matrix = data.corr()

            # Generate the heatmap using seaborn
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
            plt.title("Feature Correlation Heatmap")

            # Save the figure in a BytesIO buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Encode the plot as base64 for rendering in the frontend
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            # Return the heatmap as base64 to the frontend
            return JsonResponse({
                'status': 'success',
                'image_base64': image_base64
            })

        # Validate if the selected columns exist in the dataset
        if x_column not in data.columns or y_column not in data.columns:
            return JsonResponse({'status': 'error', 'message': 'Invalid columns selected.'})

        # Plot the graph based on selected graph type
        fig, ax = plt.subplots()
        if graph_type == 'line':
            ax.plot(data[x_column], data[y_column], label=f'{y_column} vs {x_column}')
        elif graph_type == 'bar':
            ax.bar(data[x_column], data[y_column], label=f'{y_column} vs {x_column}')
        elif graph_type == 'scatter':
            ax.scatter(data[x_column], data[y_column], label=f'{y_column} vs {x_column}')
        elif graph_type == 'histogram':
            ax.hist(data[y_column], bins=20, label=f'{y_column} Distribution')
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid graph type.'})

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f'{graph_type.capitalize()} of {y_column} vs {x_column}')
        ax.legend()

        # Save the plot to a BytesIO buffer (so we can send it as base64)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Encode the plot to base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Return the base64 image, labels, and values to the frontend
        return JsonResponse({
            'status': 'success',
            'image_base64': image_base64,  # Graph image in base64
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})



def download_graph(request):
    """View for downloading the generated graph image."""
    if request.method == "POST" and 'graph_image' in request.FILES:
        graph_image = request.FILES['graph_image']

        # Save the image to the server or handle it accordingly
        file_path = f"media/graphs/{graph_image.name}"
        with open(file_path, 'wb') as f:
            for chunk in graph_image.chunks():
                f.write(chunk)

        return JsonResponse({'status': 'success', 'message': 'Graph image downloaded successfully.'})

    return JsonResponse({'status': 'error', 'message': 'No image file found.'})


from django.shortcuts import render
import pandas as pd
import numpy as np
def categorical_encoding_recommendations(request, dataset_id):
    """Render the data encoding page for categorical columns."""
    dataset_obj = get_object_or_404(Dataset, id=dataset_id)
    file_path = dataset_obj.file.path
    data = pd.read_csv(file_path)

    # Identify categorical columns
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']

    # Check if all columns are numeric (no categorical columns remaining)
    show_next = len(categorical_columns) == 0

    # Get sample values and recommendations for each column
    column_samples = {}
    column_recommendations = {}
    for column in categorical_columns:
        column_samples[column] = data[column].dropna().unique()[:5].tolist()
        column_recommendations[column] = get_column_recommendation(data, column)

    # Generate preview HTML without index
    preview_html = data.head().to_html(
        classes='table table-striped',
        index=False,
        float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x
    )

    context = {
        'columns': categorical_columns,
        'column_samples': column_samples,
        'column_recommendations': column_recommendations,
        'dataset_id': dataset_id,
        'data_preview': preview_html,
        'csrf_token': get_token(request),
        'show_next': show_next  # Add this to context
    }
    return render(request, 'eda/categorical_encoding.html', context)


@csrf_exempt
def drop_column(request, dataset_id):
    """Drop a specified column from the dataset."""
    if request.method == 'POST':
        dataset_obj = get_object_or_404(Dataset, id=dataset_id)
        file_path = dataset_obj.file.path
        data = pd.read_csv(file_path)

        # Get the column name from the request
        body = json.loads(request.body)
        column = body.get('column')

        # Strip whitespace and validate the column name
        if column not in data.columns:
            return JsonResponse({'status': 'error', 'message': f'Invalid column name: {column}. Available columns are: {list(data.columns)}'})

        # Drop the column
        data = data.drop(columns=[column])

        # Save the updated dataset back to the file
        data.to_csv(file_path, index=False)

        return JsonResponse({'status': 'success', 'message': f'Column {column} dropped successfully.'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})



import logging

logger = logging.getLogger(__name__)



def apply_encoding(request, dataset_id):
    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        data = pd.read_csv(dataset.file.path)
        body = json.loads(request.body)
        column = body.get('column')
        encoding_type = body.get('encoding_type')

        # Retrieve existing encoding mappings
        encoding_mappings = dataset.get_encoding_mappings()
        # Apply the encoding based on the selected type
        if encoding_type == "dummy":
            logger.info(f"Starting dummy encoding for column '{column}'")
            unique_values = data[column].unique()
            logger.debug(f"Unique values in column '{column}': {unique_values}")

            dummy_df = pd.get_dummies(data[column], prefix=column)
            data = pd.concat([data.drop(columns=[column]), dummy_df], axis=1)

            # Save dummy encoding mappings
            encoding_mappings[column] = {val: f"{column}_{val}" for val in unique_values}
            logger.debug(f"Dummy encoding mappings: {encoding_mappings[column]}")
        elif encoding_type == "one-hot":
            encoder = OneHotEncoder(sparse=False)
            encoded = encoder.fit_transform(data[[column]])
            column_names = [f"{column}_{val}" for val in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=column_names)
            data = pd.concat([data.drop(columns=[column]), encoded_df], axis=1)
            # Save one-hot encoding mappings
            encoding_mappings[column] = dict(zip(encoder.categories_[0], column_names))
        elif encoding_type == "label":
            encoder = LabelEncoder()
            data[f"{column}_encoded"] = encoder.fit_transform(data[column])
            data.drop(columns=[column], inplace=True)
            # Save label encoding mappings
            encoding_mappings[column] = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        elif encoding_type == "str_to_int":
            # Convert string numbers to integers, handling errors
            def safe_convert_to_int(x):
                try:
                    x_cleaned = str(x).strip().replace(',', '')  # Strip whitespace and remove commas
                    float_value = float(x_cleaned)  # Convert to float first
                    int_value = int(float_value)  # Convert to integer
                    return int_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert value '{x}' to integer: {e}")
                    return None
            data[f"{column}_int"] = data[column].apply(safe_convert_to_int)
            logger.debug(f"Converted values in column '{column}_int': {data[f'{column}_int'].tolist()}")
            data.drop(columns=[column], inplace=True)
            logger.info(f"Dropped original column '{column}'.")
            # No specific mapping needed for str_to_int

        # Save the transformed dataset
        data.to_csv(dataset.file.path, index=False)
        logger.info("Transformed dataset saved successfully")
        # Save updated encoding mappings
        dataset.set_encoding_mappings(encoding_mappings)
        dataset.save()
        logger.info("Encoding mappings saved successfully")
        # Check if any categorical columns remain
        remaining_cat_cols = [col for col in data.columns if data[col].dtype == 'object']
        show_next = len(remaining_cat_cols) == 0

        # Generate updated preview HTML
        preview_html = data.head().to_html(
            classes='table table-striped table-bordered',
            index=False,
            float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x,
            justify='center'
        )
        return JsonResponse({
            'status': 'success',
            'updated_table': preview_html,
            'show_next': show_next,
            'message': f'Successfully applied {encoding_type} encoding to column {column}'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

def get_column_recommendation(data, column):
    """Generate encoding recommendations based on column characteristics."""
    unique_values = data[column].nunique()
    value_counts = data[column].value_counts()
    sample_values = data[column].iloc[:5].tolist()

    recommendations = []

    # Add recommendations based on characteristics
    if unique_values <= 5:
        recommendations.append({
            'method': 'dummy',
            'confidence': 90,
            'reason': 'Few unique categories, perfect for dummy encoding'
        })
    elif unique_values <= 15:
        recommendations.append({
            'method': 'one-hot',
            'confidence': 80,
            'reason': 'Moderate number of categories, one-hot encoding suitable'
        })

    if data[column].str.match(r'^\d+$').all():
        recommendations.append({
            'method': 'label',
            'confidence': 90,
            'reason': 'Contains numeric strings, label encoding preserves ordering'
        })

    # Check for high cardinality
    if unique_values > 50:
        recommendations.append({
            'method': 'drop',
            'confidence': 75,
            'reason': 'High cardinality, consider dropping or using advanced encoding'
        })

    return {
        'recommendations': recommendations,
        'stats': {
            'unique_values': unique_values,
            'top_values': value_counts.head(3).to_dict()
        }
    }

from pandas import read_csv

def model_selection(request, dataset_id):
    dataset_obj = get_object_or_404(Dataset, id=dataset_id)
    file_path = dataset_obj.file.path
    dataset = read_csv(file_path)  # Replace with actual dataset
    context = {
        'dataset_id': dataset_id,
        'dataset': {
            'columns': dataset.columns.tolist(),
            'head': dataset.head(5).values.tolist(),
        },
    }
    return render(request, 'eda/model_selection.html', context)


from django.http import HttpResponse

def classification(request, dataset_id):
    return HttpResponse(f"Classification task initiated for dataset {dataset_id}.")

def prediction(request, dataset_id):
    return HttpResponse(f"Prediction task initiated for dataset {dataset_id}.")




from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from scipy import stats

from django.shortcuts import render, get_object_or_404
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def preview_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    df = pd.read_csv(dataset.file.path)

    # Extract column names and initial rows for display
    columns = df.columns.tolist()
    rows = df.head(10).values.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}
    graphs_data = {}

    # Handle POST request for outlier removal
    if request.method == 'POST':
        method = request.POST.get('outlier_method')
        selected_column = request.POST.get('column')

        if method and selected_column in numeric_cols:
            if method == 'remove':
                # Combine Z-score and IQR methods for outlier detection
                while True:
                    z_scores = np.abs(stats.zscore(df[selected_column]))
                    Q1 = df[selected_column].quantile(0.25)
                    Q3 = df[selected_column].quantile(0.75)
                    IQR = Q3 - Q1

                    z_outliers = z_scores > 3
                    iqr_outliers = (df[selected_column] < (Q1 - 1.5 * IQR)) | (df[selected_column] > (Q3 + 1.5 * IQR))

                    combined_outliers = z_outliers | iqr_outliers
                    if not combined_outliers.any():
                        break  # Exit loop if no more outliers
                    df = df[~combined_outliers].copy()

                # Reindex the dataframe
                df.reset_index(drop=True, inplace=True)
                df.to_csv(dataset.file.path, index=False)

            elif method == 'trim':
                z_scores = np.abs(stats.zscore(df[selected_column]))
                outlier_mask = z_scores > 3
                df.loc[outlier_mask, selected_column] *= 0.5
                df.to_csv(dataset.file.path, index=False)

            elif method == 'winsor':
                Q1 = df[selected_column].quantile(0.25)
                Q3 = df[selected_column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[selected_column] = df[selected_column].clip(lower=lower_bound, upper=upper_bound)
                df.to_csv(dataset.file.path, index=False)

            elif method == 'robust':
                median = df[selected_column].median()
                mad = stats.median_abs_deviation(df[selected_column])
                modified_z_scores = 0.6745 * (df[selected_column] - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
                df.loc[outlier_mask, selected_column] = median
                df.to_csv(dataset.file.path, index=False)

            rows = df.head(10).values.tolist()

    # Detect remaining outliers and generate graphs
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        z_outliers = z_scores > 3
        iqr_outliers = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        combined_outliers = z_outliers | iqr_outliers
        outliers_info[col] = int(combined_outliers.sum())

        # Generate boxplot
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Distribution of {col}')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        graphs_data[col] = base64.b64encode(buffer.getvalue()).decode('utf-8')

    outlier_note = """
    Outliers are data points that deviate significantly from the overall pattern of the dataset. 
    They can greatly impact model performance by skewing statistical measures and affecting the learning process. 
    Proper handling of outliers is crucial for building robust and accurate machine learning models. 
    Different methods can be used to handle outliers based on the specific needs of your analysis.
    """

    return render(request, 'eda/scailing.html', {
        'columns': columns,
        'rows': rows,
        'dataset_id': dataset_id,
        'outliers': outliers_info,
        'graphs': graphs_data,
        'outlier_note': outlier_note,
        'has_outliers': any(count > 0 for count in outliers_info.values())
    })


def overfitting_under(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    df = pd.read_csv(dataset.file.path)
    columns = df.columns.tolist()
    rows = df.head(10).values.tolist()
    print(dataset.get_independent_columns())

    if request.method == 'POST':
        # Get independent columns and target column from POST data
        independent_cols = request.POST.getlist('independent_columns[]')
        target_col = request.POST.get('target_column')

        # Save the independent columns and target column
        # Clean the independent columns (remove any empty strings)
        independent_cols = [col.strip() for col in independent_cols if col.strip()]
        dataset.set_independent_columns(independent_cols)  # Using the new helper method
        dataset.target_column = target_col
        dataset.column_selection_complete = True
        dataset.save()

        return JsonResponse({
            'status': 'success',
            'message': 'Columns saved successfully'
        })
    # Debug prints
    # Get independent columns and ensure they're properly split
    independent_cols = dataset.get_independent_columns()
    if len(independent_cols) == 1 and isinstance(independent_cols[0], str) and ',' in independent_cols[0]:
        independent_cols = independent_cols[0].split(',')

    # Debug prints
    print("Independent columns (processed):", independent_cols)
    print("Columns available:", columns)
    context = {
        'dataset_id':dataset_id,
        'dataset': dataset,
        'columns': columns,
        'rows': rows,
        'selected_independent': independent_cols,
        'selected_target': dataset.target_column,
    }
    return render(request, 'eda/over_under.html', context)





def drop_column_new(request, dataset_id):
    if request.method == "POST":
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            column_to_drop = json.loads(request.body).get("column")

            # Check if the dataset file exists
            dataset_path = dataset.file.path
            if not os.path.exists(dataset_path):
                return JsonResponse({"success": False, "message": "Dataset file not found."}, status=404)

            # Load the dataset file
            try:
                data = pd.read_csv(dataset_path)
            except pd.errors.EmptyDataError:
                return JsonResponse({"success": False, "message": "The dataset is empty or invalid."}, status=400)

            # Check if the column exists in the dataset
            if column_to_drop not in data.columns:
                return JsonResponse({"success": False, "message": f"Column '{column_to_drop}' not found in the dataset."}, status=400)

            # Drop the column
            data = data.drop(columns=[column_to_drop])

            # Save the updated dataset back to the same file
            data.to_csv(dataset_path, index=False)

            # Update independent columns in the database
            updated_columns = list(data.columns)
            dataset.set_independent_columns(updated_columns)
            dataset.save()

            # Return a success response
            return JsonResponse({"success": True, "message": f"Column '{column_to_drop}' dropped successfully!"})
        except Dataset.DoesNotExist:
            return JsonResponse({"success": False, "message": "Dataset not found."}, status=404)
        except Exception as e:
            return JsonResponse({"success": False, "message": f"An unexpected error occurred: {str(e)}"}, status=500)
    else:
        return JsonResponse({"success": False, "message": "Invalid request method."}, status=405)


from imblearn.over_sampling import SMOTE, ADASYN

from collections import Counter
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
def lasso_graph(request, dataset_id):
    try:
        # Fetch dataset
        dataset = get_object_or_404(Dataset, id=dataset_id)
        df = pd.read_csv(dataset.file.path)

        # Validate columns
        independent_columns = dataset.get_independent_columns()
        target_column = dataset.target_column
        if not independent_columns or not target_column:
            return render(request, 'eda/lasso_graph.html', {'error': 'Dataset columns not configured properly.'})

        # Separate numerical and categorical columns
        numerical_cols = df[independent_columns].select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df[independent_columns].select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle missing values systematically
        if numerical_cols:
            # Use median for skewed distributions, mean for normal distributions
            num_imputer = SimpleImputer(strategy='median')  # You can switch to 'mean' if needed
            df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

        if categorical_cols:
            # Use most frequent value for categorical columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

        # Update X and y with the cleaned dataset
        X = df[independent_columns]
        y = df[target_column]

        # Check if target is categorical or continuous
        is_categorical = y.dtype == 'object' or len(y.unique()) < 10  # Simple heuristic
        imbalance_detected = False
        df_resampled = None
        resampled_counts = None
        new_rows_count = 0
        new_rows_preview = []

        if is_categorical:
            target_counts = y.value_counts()
            imbalance_detected = len(target_counts) > 1 and target_counts.min() / target_counts.max() < 0.5

            if imbalance_detected and request.method == "POST":
                technique = request.POST.get('technique')

                # Ensure no NaN values remain in y
                if y.isnull().any():
                    y = y.fillna(y.mode()[0])  # Fill NaN in target column with the most frequent value

                if technique == 'SMOTE':
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X, y)
                elif technique == 'ADASYN':
                    adasyn = ADASYN(random_state=42)
                    X_res, y_res = adasyn.fit_resample(X, y)

                # Create resampled DataFrame
                df_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns),
                                          pd.Series(y_res, name=target_column)], axis=1)

                # Identify new rows
                original_indices = set(df.index)
                resampled_indices = set(df_resampled.index)
                new_rows_indices = list(resampled_indices - original_indices)
                new_rows_count = len(new_rows_indices)

                # Add a flag column to indicate new rows
                df_resampled['is_new'] = df_resampled.index.isin(new_rows_indices).astype(bool)

                # Get the first 5 new rows for preview
                new_rows_preview = df_resampled[df_resampled['is_new']].head().to_dict(orient='records')

                # Recompute class distribution after resampling
                resampled_target_counts = y_res.value_counts()
                imbalance_detected = len(resampled_target_counts) > 1 and resampled_target_counts.min() / resampled_target_counts.max() < 0.5

                resampled_counts = y_res.value_counts().to_dict()

                # Save the resampled dataset back to the file system
                df_resampled.to_csv(dataset.file.path, index=False)

        context = {
            'dataset_id': dataset_id,
            'columns': df.columns.tolist(),
            'rows': df.head(10).values.tolist(),
            'is_categorical': is_categorical,
            'imbalanced': imbalance_detected,
            'target_column': target_column,
            'independent_columns': independent_columns,
        }

        if is_categorical:
            context['original_counts'] = y.value_counts().to_dict()

        if df_resampled is not None:
            context.update({
                'df_resampled': df_resampled.head(10).to_dict(orient='records'),  # Convert to dictionary for easier handling
                'resampled_counts': resampled_counts,
                'new_rows_count': new_rows_count,
                'new_rows_preview': new_rows_preview,
            })

        return render(request, 'eda/lasso_graph.html', context)

    except Exception as e:
        return render(request, 'eda/lasso_graph.html', {'error': f'An error occurred: {str(e)}'})

def normalizing_scaling(request, dataset_id):
    from django.shortcuts import get_object_or_404
    import pandas as pd
    from django.http import JsonResponse

    dataset = get_object_or_404(Dataset, id=dataset_id)
    df = pd.read_csv(dataset.file.path)

    # Independent and target columns
    independent_columns = dataset.get_independent_columns() if hasattr(dataset, 'get_independent_columns') else df.columns
    target_column = dataset.target_column if hasattr(dataset, 'target_column') else None

    # Generate scaling recommendations, skipping binary columns (0/1 or True/False)
    recommendations = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if set(df[col].dropna().unique()).issubset({0, 1}):  # Skip binary columns
                recommendations.append({
                    'column': col,
                    'method': 'No Scaling Needed',
                    'options': ['No Scaling Needed']
                })
            else:
                scaling_method = 'Normalization' if df[col].max() <= 1 and df[col].min() >= 0 else 'Standardization'
                recommendations.append({
                    'column': col,
                    'method': scaling_method,
                    'options': ['Normalization', 'Standardization', 'Min-Max Scaling']
                })
        else:
            recommendations.append({
                'column': col,
                'method': 'No Scaling Needed',
                'options': ['No Scaling Needed']
            })

    # Handle form submission for scaling/normalization
    if request.method == 'POST':
        # Collect scaling choices
        scaling_choices = {}
        for col in df.columns:
            scaling_choices[col] = request.POST.get(f'scaling_{col}', None)

        # Apply scaling and normalization based on user-selected choices
        for col, method in scaling_choices.items():
            if col in df.columns and method:
                try:
                    if method == 'Normalization':
                        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())  # 0 to 1 range
                    elif method == 'Min-Max Scaling':
                        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    elif method == 'Standardization':
                        df[col] = (df[col] - df[col].mean()) / df[col].std()  # Mean = 0, Std Dev = 1
                except Exception as e:
                    print(f"Error scaling column {col}: {e}")

        # Save the updated dataset
        df.to_csv(dataset.file.path, index=False)

        # Return success response
        return redirect('normalizing',dataset_id=dataset_id)

    # Render the page with recommendations and a preview of the dataset
    return render(request, 'eda/normalizing.html', {
        'dataset': dataset,
        'df_preview': df.head(10).to_html(classes="table table-striped table-bordered"),
        'recommendations': recommendations,
        'dataset_id': dataset_id,
    })




def model_pre_configuration(request,dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    return render(request,'eda/model_pre_config.html',{'dataset_id':dataset_id})

def model_train(request, task, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    context = {
        'task': task,
        'dataset_id': dataset_id,
    }
    return render(request, 'eda/model_train.html', context)


import json
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle
import asyncio
from concurrent.futures import ProcessPoolExecutor
from asgiref.sync import sync_to_async




def start_training(request, task, dataset_id):
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        independent_columns = dataset.get_independent_columns()
        target_column = dataset.target_column

        if not independent_columns or not target_column:
            return JsonResponse({'error': 'Independent or target columns are not defined.'})

        # Load data
        file_path = dataset.file.path
        data = pd.read_csv(file_path)

        # Debug prints
        print("Data shape:", data.shape)
        print("Independent columns:", independent_columns)
        print("Target column:", target_column)

        X = data[independent_columns]
        y = data[target_column]

        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("y unique values:", y.unique())

        # Handle missing values in X
        if X.isnull().any().any():
            print("Warning: Missing values in features. Filling with mean/mode.")
            numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns

            for col in numeric_columns:
                X[col] = X[col].fillna(X[col].mean())
            for col in categorical_columns:
                X[col] = X[col].fillna(X[col].mode()[0])

        # Handle missing values in target column
        if y.isnull().any():
            return JsonResponse({'error': f'Target column "{target_column}" contains missing values. Please clean your data and try again.'})

        # Handle continuous target values for classification
        if task == "classification":
            if y.dtype in ['float64', 'float32']:
                try:
                    y = y.astype(int)
                except ValueError:
                    # Discretize continuous target into bins
                    num_bins = 3  # Adjust as needed
                    y = pd.cut(y, bins=num_bins, labels=range(num_bins))

            # Verify target is now categorical
            if y.dtype not in ['int64', 'int32', 'category']:
                return JsonResponse(
                    {'error': 'Target column is not suitable for classification. Please provide categorical data.'})

        # Train-Test Split
        test_size = float(request.GET.get('test_size', 0.2))
        random_state = int(request.GET.get('random_state', 42))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print("Training set shapes:", X_train.shape, y_train.shape)
        print("Test set shapes:", X_test.shape, y_test.shape)
        # Train models and store them in memory
        models_dict = {}
        # Train models
        results = []
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(),
            'Support Vector Machine': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
        }

        for name, model in models.items():
            try:
                # Check for infinite or NaN values
                if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                    print(f"Warning: NaN or infinite values found in training data")
                    X_train = np.nan_to_num(X_train)
                    X_test = np.nan_to_num(X_test)

                print(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Print predictions distribution
                print(f"{name} predictions distribution:", np.unique(y_pred, return_counts=True))

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                print(f"{name} metrics:", {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

                # Store the trained model in memory
                models_dict[name] = model

                results.append({
                    'model': name,
                    'accuracy': float(accuracy),  # Ensure values are JSON serializable
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                })
            except Exception as e:
                print(f"Error training {name}:", str(e))
                results.append({
                    'model': name,
                    'error': str(e),
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0
                })

        # Find best model
        # Store models in session for later retrieval
        request.session['trained_models'] = {name: pickle.dumps(model).hex() for name, model in models_dict.items()}
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_model = max(valid_results, key=lambda x: x['accuracy'])
        else:
            return JsonResponse({'error': 'All models failed to train. Please check your data.'})

        print("Best model:", best_model)
        return JsonResponse({
            'results': results,
            'best_model': best_model,
        })

    except Exception as e:
        print("Unexpected error:", str(e))
        return JsonResponse({'error': f'Unexpected error: {str(e)}'})


def training_progress(request):
    # Mimic live progress updates
    progress = request.session.get('progress', 0)
    if progress < 100:
        request.session['progress'] = progress + 20
        return JsonResponse({'progress': progress + 20})
    return JsonResponse({'progress': 100})

@csrf_exempt
def save_model(request, dataset_id):
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        model_name = request.POST.get('model_name')
        format_type = request.POST.get('format_type', 'joblib')  # 'joblib' or 'pickle'

        # Get the model from session
        models_dict = request.session.get('trained_models', {})
        if not model_name in models_dict:
            return JsonResponse({'error': 'Model not found in session'})

        # Deserialize the model
        model = pickle.loads(bytes.fromhex(models_dict[model_name]))

        # Create the saved_models directory if it doesn't exist
        save_dir = os.path.join(settings.MEDIA_ROOT, 'saved_models')
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename
        filename = f"model_{dataset_id}_{model_name.lower().replace(' ', '_')}"

        if format_type == 'joblib':
            file_path = os.path.join(save_dir, f"{filename}.joblib")
            joblib.dump(model, file_path)
        else:  # pickle
            file_path = os.path.join(save_dir, f"{filename}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)

        # Update dataset model
        relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT)
        dataset.saved_model.name = relative_path
        dataset.model_type = model_name
        dataset.save()

        return JsonResponse({
            'success': True,
            'message': f'Model successfully saved as {format_type} file',
            'next_url': f'/lasso_regression_fun/{dataset_id}/'  # Replace with your next URL
        })

    except Exception as e:
        return JsonResponse({'error': str(e)})



# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import math

def lasso_graph2(request, dataset_id):
    try:
        # Fetch dataset
        logger.info(f"Fetching dataset with ID: {dataset_id}")
        dataset = get_object_or_404(Dataset, id=dataset_id)

        # Read CSV file
        try:
            df = pd.read_csv(dataset.file.path)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
        except Exception as csv_error:
            logger.error(f"Error reading CSV: {csv_error}")
            return render(request, 'eda/lasso_graph.html', {
                'error': f'CSV loading error: {csv_error}'
            })

        # Validate columns
        independent_columns = dataset.get_independent_columns()
        target_column = dataset.target_column
        if not independent_columns or not target_column:
            logger.warning("Columns not configured properly")
            return render(request, 'eda/lasso_graph.html', {
                'error': 'Dataset columns not configured properly.'
            })

        # Prepare data
        try:
            X = df[independent_columns]
            y = df[target_column]
            logger.info(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}")
        except KeyError as column_error:
            logger.error(f"Column selection error: {column_error}")
            return render(request, 'eda/lasso_graph.html', {
                'error': f'Column selection error: {column_error}'
            })

        # Retrieve encoding mappings
        encoding_mappings = dataset.get_encoding_mappings()

        # Initialize context
        context = {
            'dataset_id': dataset_id,
            'independent_columns': independent_columns,
            'target_column': target_column,
            'model_file_url': dataset.saved_model.url if dataset.saved_model else None,
            'file_url': dataset.file.url if dataset.file else None,
            'encoding_mappings': encoding_mappings  # Pass encoding mappings to the template
        }

        # Model loading and evaluation
        if dataset.saved_model and os.path.exists(dataset.saved_model.path):
            try:
                # Load model with detailed logging
                logger.info(f"Attempting to load model from: {dataset.saved_model.path}")
                model = joblib.load(dataset.saved_model.path)
                logger.info(f"Model loaded successfully. Type: {type(model)}")

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                logger.info(f"Data split complete. Train shapes: {X_train.shape}, {y_train.shape}")

                # Predictions
                try:
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    logger.info("Predictions generated successfully")
                except Exception as pred_error:
                    logger.error(f"Prediction error: {pred_error}")
                    context['prediction_error'] = str(pred_error)
                    return render(request, 'eda/lasso_regression.html', context)

                # Metrics calculation
                try:
                    # Calculate RMSE manually for compatibility
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    train_rmse = math.sqrt(train_mse)
                    test_rmse = math.sqrt(test_mse)

                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    logger.info(f"Metrics - Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
                    logger.info(f"R2 Scores - Train: {train_r2}, Test: {test_r2}")
                except Exception as metrics_error:
                    logger.error(f"Metrics calculation error: {metrics_error}")
                    context['model_error'] = str(metrics_error)
                    model = None
                    model_status = "Metrics Error"
                    graph_data = None

                # Model status determination
                model_status = "Good"
                if abs(train_r2 - test_r2) > 0.1 or train_rmse < test_rmse * 0.5:
                    model_status = "Overfit"
                elif train_r2 < 0.7 or test_r2 < 0.7:
                    model_status = "Underfit"
                logger.info(f"Model Status: {model_status}")

                # Visualization
                try:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].scatter(y_train, y_train_pred, color='blue', alpha=0.6)
                    ax[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red',
                               linestyle='--')
                    ax[0].set_title("Train Data: Actual vs Predicted")
                    ax[1].scatter(y_test, y_test_pred, color='green', alpha=0.6)
                    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
                    ax[1].set_title("Test Data: Actual vs Predicted")
                    buf = BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    graph_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()
                    logger.info("Graph generated successfully")
                except Exception as plot_error:
                    logger.error(f"Plot generation error: {plot_error}")
                    graph_data = None
            except Exception as model_load_error:
                logger.error(f"Model loading error: {model_load_error}")
                context['model_error'] = str(model_load_error)
                model = None
                model_status = "Model not found"
                graph_data = None
        else:
            logger.warning("No saved model found")
            model = None
            model_status = "Model not found"
            graph_data = None

        # Update context
        context.update({
            'model_status': model_status,
            'graph_data': graph_data
        })

        # Handle prediction requests
        if request.method == "POST":
            try:
                if model is None:
                    raise Exception("Model not loaded - cannot make predictions")
                input_data = {}
                for col in independent_columns:
                    value = request.POST.get(col)
                    if value is None:
                        context['prediction_error'] = f'Missing value for column: {col}'
                        return render(request, 'eda/lasso_regression.html', context)
                    try:
                        input_data[col] = float(value)
                    except ValueError:
                        context['prediction_error'] = f'Invalid numeric value for column: {col}'
                        return render(request, 'eda/lasso_regression.html', context)
                input_df = pd.DataFrame([input_data])
                input_df = input_df[independent_columns]
                prediction = model.predict(input_df)[0]
                context['prediction'] = prediction
                logger.info(f"Single prediction generated: {prediction}")
            except Exception as pred_error:
                logger.error(f"Prediction generation error: {pred_error}")
                context['prediction_error'] = str(pred_error)

        return render(request, 'eda/lasso_regression.html', context)
    except Exception as unexpected_error:
        logger.critical(f"Unexpected error in view: {unexpected_error}")
        return render(request, 'eda/lasso_regression.html', {
            'error': f'Unexpected system error: {unexpected_error}'
        })


def prediction_function(request,task,dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    context = {
        'task': task,
        'dataset_id': dataset_id,
    }
    return render(request, 'eda/model_train_prediction.html', context)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import pickle


def start_training_prediction(request,dataset_id):
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        independent_columns = dataset.get_independent_columns()
        target_column = dataset.target_column

        if not independent_columns or not target_column:
            return JsonResponse({'error': 'Independent or target columns are not defined.'})

        # Load data
        file_path = dataset.file.path
        data = pd.read_csv(file_path)

        print("Data shape:", data.shape)
        print("Independent columns:", independent_columns)
        print("Target column:", target_column)

        X = data[independent_columns]
        y = data[target_column]

        print("X shape:", X.shape)
        print("y shape:", y.shape)

        # Handle missing values in X
        if X.isnull().any().any():
            print("Warning: Missing values in features. Filling with mean/mode.")
            numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns

            for col in numeric_columns:
                X[col] = X[col].fillna(X[col].mean())
            for col in categorical_columns:
                X[col] = X[col].fillna(X[col].mode()[0])

        # Handle missing values in target column
        if y.isnull().any():
            return JsonResponse({
                                    'error': f'Target column "{target_column}" contains missing values. Please clean your data and try again.'})

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_columns.empty:
            X = pd.get_dummies(X, columns=categorical_columns)

        # Train-Test Split
        test_size = float(request.GET.get('test_size', 0.2))
        random_state = int(request.GET.get('random_state', 42))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print("Training set shapes:", X_train.shape, y_train.shape)
        print("Test set shapes:", X_test.shape, y_test.shape)

        # Initialize models dictionary for storage
        models_dict = {}

        # Define regression models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'SVR': SVR(kernel='rbf'),
            'Ridge Regression': Ridge(random_state=random_state),
            'Lasso Regression': Lasso(random_state=random_state)
        }

        results = []
        for name, model in models.items():
            try:
                # Check for infinite or NaN values
                if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                    print(f"Warning: NaN or infinite values found in training data")
                    X_train = np.nan_to_num(X_train)
                    X_test = np.nan_to_num(X_test)

                print(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(f"{name} metrics:", {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae
                })

                # Store the trained model in memory
                models_dict[name] = model

                results.append({
                    'model': name,
                    'r2_score': float(r2),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mse': float(mse)
                })
            except Exception as e:
                print(f"Error training {name}:", str(e))
                results.append({
                    'model': name,
                    'error': str(e),
                    'r2_score': 0,
                    'rmse': 0,
                    'mae': 0,
                    'mse': 0
                })

        # Store models in session for later retrieval
        request.session['trained_models'] = {name: pickle.dumps(model).hex() for name, model in models_dict.items()}

        # Find best model based on R2 score
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_model = max(valid_results, key=lambda x: x['r2_score'])
        else:
            return JsonResponse({'error': 'All models failed to train. Please check your data.'})

        print("Best model:", best_model)
        return JsonResponse({
            'results': results,
            'best_model': best_model,
        })

    except Exception as e:
        print("Unexpected error:", str(e))
        return JsonResponse({'error': f'Unexpected error: {str(e)}'})







import pickle






def feature_creation(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    df = pd.read_csv(dataset.file.path)
    messages = []
    if request.method == "POST":
        try:
            selected_columns = request.POST.getlist('selected_columns')
            equation = request.POST.get('equation', '').strip()
            new_feature_name = request.POST.get('new_feature_name', '').strip()
            time_series_operation = request.POST.get('time_series_operation', None)
            window_size = int(request.POST.get('window_size', 3))

            if not new_feature_name:
                raise ValueError("Please provide a name for the new feature.")

            if time_series_operation:
                if not selected_columns:
                    raise ValueError("Please select at least one column for time series operation.")
                for col in selected_columns:
                    if time_series_operation == "rolling_mean":
                        df[new_feature_name] = df[col].rolling(window=window_size).mean()
                    elif time_series_operation == "lag":
                        df[new_feature_name] = df[col].shift(periods=window_size)
                    elif time_series_operation == "rolling_sum":
                        df[new_feature_name] = df[col].rolling(window=window_size).sum()
                    else:
                        raise ValueError("Invalid time series operation.")

                # Handle NaN values
                df[new_feature_name] = df[new_feature_name].fillna(0)  # Fill NaN with 0
                # OR: df[new_feature_name] = df[new_feature_name].dropna()  # Drop rows with NaN

            elif equation:
                # Create a safe namespace for equation evaluation
                safe_dict = {
                    'np': np,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum
                }
                # Add selected columns to namespace
                for col in selected_columns:
                    safe_dict[col] = df[col]
                # Evaluate equation safely
                result = eval(equation, {"__builtins__": {}}, safe_dict)
                # Add new feature to dataframe
                df[new_feature_name] = result
            else:
                raise ValueError("Please specify an equation or a time series operation.")

            # Save updated dataset
            df.to_csv(dataset.file.path, index=False)
            messages.append(('success', f'Created new feature: {new_feature_name}'))
        except Exception as e:
            messages.append(('error', f'Error: {str(e)}'))

    # Prepare examples for display
    examples = [
        {'name': 'Sum of columns', 'equation': 'col1 + col2', 'description': 'Adds values from two columns'},
        {'name': 'Average', 'equation': '(col1 + col2) / 2', 'description': 'Calculates average of two columns'},
        {'name': 'Percentage', 'equation': '(col1 / col2) * 100', 'description': 'Calculates percentage'},
        {'name': 'Log transformation', 'equation': 'np.log(col1)', 'description': 'Natural logarithm of values'}
    ]
    context = {
        'dataset': dataset,
        'df_preview': df.head(10).to_html(classes='table'),
        'columns': df.columns.tolist(),
        'messages': messages,
        'examples': examples
    }
    return render(request, 'eda/feature_creation.html', context)


import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO

@csrf_exempt
def heat_map_visualisation(request, dataset_id):
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        df = pd.read_csv(dataset.file.path)

        # Get independent columns and target column
        independent_columns = dataset.get_independent_columns()
        target_column = dataset.target_column

        if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            action = request.POST.get("action")
            if action == "remove_columns":
                columns_to_remove = request.POST.getlist("columns_to_remove[]")
                if not columns_to_remove:
                    return JsonResponse({"status": "error", "message": "No columns selected for removal."})

                # Remove columns from the DataFrame
                df = df.drop(columns=columns_to_remove)

                # Update independent_columns in the Dataset model
                for column in columns_to_remove:
                    dataset.remove_independent_column(column)

                # Save updated dataset
                df.to_csv(dataset.file.path, index=False)
                return JsonResponse({"status": "success", "message": f"Removed columns: {', '.join(columns_to_remove)}"})
            elif action == "generate_heatmap":
                # Generate correlation matrix
                corr_matrix = df.corr()

                # Plot heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
                plt.title("Feature Correlation Heatmap")

                # Save plot to buffer and encode as base64
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                # Calculate feature importance (absolute correlation with target)
                feature_importance = []
                if target_column in df.columns:
                    for col in independent_columns:
                        if col in df.columns:
                            importance = abs(df[col].corr(df[target_column]))
                            feature_importance.append({"column": col, "importance": importance})

                return JsonResponse({
                    "status": "success",
                    "image_base64": image_base64,
                    "feature_importance": feature_importance
                })

        # Perform PCA for dimensionality reduction recommendations
        X = df[independent_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()
        pca.fit(X_scaled)

        # Variance explained by each principal component
        variance_explained = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)

        # Recommend number of components to retain
        recommended_components = np.argmax(cumulative_variance >= 0.95) + 1  # Retain 95% variance
        recommended_columns = [col for i, col in enumerate(independent_columns) if i < recommended_components]

        context = {
            "dataset_id":dataset_id,
            "dataset": dataset,
            "independent_columns": independent_columns,
            "target_column": target_column,
            "recommended_columns": recommended_columns,
            "columns_to_remove": list(set(independent_columns) - set(recommended_columns)),
            "dataset_preview": df.head().to_html(classes='table table-striped', index=False),
        }
        return render(request, "eda/heat_map.html", context)

    except Exception as e:
        # Log the error for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

        # Return JSON error response for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({"status": "error", "message": str(e)})
        # For non-AJAX requests, re-raise the exception
        raise