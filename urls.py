from django.urls import path
from . import views



urlpatterns = [
    path('',views.Home,name='home'),
    path('upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('analyze/<int:dataset_id>/', views.analyze_dataset, name='analyze'),
    path('vector-conversion/<int:dataset_id>/', views.vector_conversion, name='vector_conversion'),
    path('apply-encoding/<int:dataset_id>/', views.apply_encoding, name='apply_encoding'),
    path('generate-graph/<int:dataset_id>/', views.generate_graph, name='generate_graph'),
    path('data-encoding/<int:dataset_id>/', views.categorical_encoding_recommendations, name='data_encoding_recommendations'),
    path('download-graph/', views.download_graph, name='download_graph'),
    path('drop-column/<int:dataset_id>/', views.drop_column, name='drop_column'),
    path('dataset/<int:dataset_id>/feature-engineering/', views.feature_engineering, name='feature_engineering'),
    path('data-encoding/<int:dataset_id>/',views.categorical_encoding_recommendations,name='data_encoding_recommendations'),
    path('model-selection/<int:dataset_id>/', views.model_selection, name='model_selection'),
    path('preview_dataset/<int:dataset_id>/',views.preview_dataset,name='preview_dataset'),

    path('classification/<int:dataset_id>/', views.classification, name='classification'),
    path('prediction/<int:dataset_id>/', views.prediction, name='prediction'),
    path('overfitting_under/<int:dataset_id>/',views.overfitting_under,name='over_under'),
    path('lasso_graph/<int:dataset_id>/',views.lasso_graph,name='lasso_graph'),
    path('model_pre_configuration/<int:dataset_id>/',views.model_pre_configuration,name='model_pre_configuration'),
    path('model_train/<str:task>/<int:dataset_id>/',views.model_train,name='model_train'),
    path('start_training/<str:task>/<int:dataset_id>/',views.start_training,name='start_training'),
    path('training_progress/',views.training_progress,name='training_progress'),
    path('save_model/<int:dataset_id>/', views.save_model, name='save_model'),
    path('lasso_regression_fun/<int:dataset_id>/',views.lasso_graph2,name='lasso_regression_fun'),
    path('drop_column_new/<int:dataset_id>/', views.drop_column_new, name='drop_column_new'),
    path('normalizing/<int:dataset_id>/',views.normalizing_scaling,name='normalizing'),

    path('prediction_function/<str:task>/<int:dataset_id>/',views.prediction_function,name='prediction'),
    path('start_training_prediction/<int:dataset_id>/',views.start_training_prediction,name='start_training_prediction'),

    path('feature_creation/<int:dataset_id>/',views.feature_creation,name='feature_creation'),
    path('heat_map_visualisation/<int:dataset_id>/', views.heat_map_visualisation, name='heat_map_visualisation'),
    path('get-dataset-file/<int:dataset_id>/', views.get_dataset_file, name='get_dataset_file'),



]
