def standardized_shap(shap_output: dict) -> dict:
    """
    Convert raw SHAP output into a standardized dictionary format for XAI consumption.

    Args:
        shap_output (dict): Raw output from a SHAP explainer, containing keys such as:
            - metadata: Explainer and model information.
            - instance_context: Prediction and raw feature context.
            - base_values: Model base values.
            - attribution_values: Detailed attribution values including interactions.
            - derived_insights: Top-k features and plot data.
            - raw_feature_attributions: Aggregated attributions per raw feature.

    Returns:
        dict: Standardized output with keys:
            - explanation_method_used (str)
            - model_info (dict)
            - prediction_summary (dict)
            - instance_features_raw (list of dicts)
            - feature_attributions (list of dicts)
            - aggregated_raw_feature_attributions (list of dicts)
            - interactions (list of dicts)
            - plot_data_available (dict)
            - warnings_or_limitations (list of str)
    """
    standardized_output = {}
    if shap_output:
        # ***** Extract relevant sections from SHAP output *****
        metadata = shap_output.get('metadata', {})
        instance = shap_output.get('instance_context', {})
        base_value = shap_output.get('base_values', {})
        attributions = shap_output.get('attribution_values', {})
        derived_insights = shap_output.get('derived_insights', {})
        raw_feature_attributions = shap_output.get('raw_feature_attributions', {})

        # ***** Populate explanation method and model information *****
        standardized_output['explanation_method_used'] = f"SHAP-{metadata.get('explainer_type')}"
        standardized_output['model_info'] = {
            'class': metadata.get('model_class')
        }

        # ***** Build prediction summary *****
        prediction = instance.get('prediction', {})
        standardized_output['prediction_summary'] = {
            'label': prediction.get('label'),
            'probability_scores': prediction.get('probabilities'),
            'model_base_value': base_value.get('default')
        }

        # ***** Collect raw instance features *****
        feature_names = instance.get('feature_names', [])
        feature_values = instance.get('feature_values', [])
        standardized_output['instance_features_raw'] = [
            {'name': name, 'value': value}
            for name, value in zip(feature_names, feature_values)
        ]

        # ***** Process top-k feature attributions *****
        top_k = derived_insights.get('top_k_features', [])
        feature_attributions: list[dict] = []
        for feature in top_k:
            shap_value = feature.get('shap_value', 0.0)
            direction = 'positive' if shap_value >= 0 else 'negative'
            feature_attributions.append({
                'raw_feature_name': feature['original_raw_feature_name'],
                'raw_feature_value': feature['original_raw_feature_value'],
                'model_input_feature_name': feature['model_input_name'],
                'model_input_feature_value': feature['model_input_value'],
                'attribution_score': shap_value,
                'rank_of_importance': feature['rank'],
                'attribution_method_detail': 'SHAP value for model input feature'
            })
        standardized_output['feature_attributions'] = feature_attributions

        # ***** Aggregate raw feature attribution values *****
        aggregated: list[dict] = []
        for name, value in raw_feature_attributions.items():
            direction = 'positive' if value >= 0 else 'negative'
            aggregated.append({
                'name': name,
                'total_attribution': value,
                'direction_of_impact': direction
            })
        standardized_output['aggregated_raw_feature_attributions'] = aggregated

        # ***** Include SHAP interaction matrix and feature order *****
        standardized_output['interactions'] = [{
            'type': 'shap_interaction_matrix_raw',
            'value_matrix': attributions.get('interactions'),
            'feature_order': feature_names
        }]

        # ***** Provide force plot data if available *****
        force_plot = derived_insights.get('plot_data', {}).get('force_plot', {})
        standardized_output['plot_data_available'] = {
            'force_plot_components': force_plot
        }

        # ***** Add warnings or limitations about SHAP *****
        standardized_output['warnings_or_limitations'] = [
            f"SHAP {metadata.get('explainer_type')} provides feature attributions. "
            "Values directly explain features as seen by this specific SHAP explainer."
        ]

    return standardized_output
        
        