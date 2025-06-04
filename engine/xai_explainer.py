# ***** IMPORT FRAMEWORK *****
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier

# ***** IMPORT GENERAL LIBRARIES *****
import pandas as pd
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

# ***** IMPORT WARNING *****
import warnings

# ***** IMPORT CUSTOM FUNCTION *****
from engine.llm_summarization import LLMSummarize
from helpers.standardized_xai_tool import standardized_shap

# ***** IMPORT ANNOTATION *****
from typing import Optional, Any

class XAIExplainer:
    """
    Base class for explainable AI (XAI) tools. Ensures the model is fitted, infers its type,
    and infers or accepts feature names for downstream XAI methods. Integrates LLM summarization
    for aggregating raw XAI outputs into human-readable summaries.

    Args:
        model (BaseEstimator): A fitted scikit-learn model or pipeline.
        model_type (str, optional): "classifier" or "regressor". Inferred if not provided.
        feature_names (list, optional): List of feature names. Inferred if not provided.
        X_train_for_xai (pd.DataFrame | np.ndarray, optional): Background data used to infer feature names.
    """
    def __init__(self, 
                model, 
                model_type: str = None,
                feature_names: list = None, 
                X_train_for_xai : pd.DataFrame | np.ndarray = None):
        
        if not isinstance(model, BaseEstimator):
            raise TypeError("Input must be a scikit-learn model (inheriting from sklearn.base.BaseEstimator)")
        
        self.model = model
    
        # ***** Ensure the model is already fitted *****
        try: 
            check_is_fitted(self.model)
        except Exception as error:
            raise ValueError("The provided model is not fitted.") from error
        
        # ***** Determine model type (classifier/regressor) *****
        self.model_type = model_type if model_type else self._infer_model_type(model)
        
        # ***** Determine feature names, or infer from pipeline/X_train *****
        self.feature_names = feature_names if feature_names else self._infer_feature_names(model, X_train_for_xai)
        
        # ***** Store background data for XAI methods *****
        self.X_background_for_xai = X_train_for_xai
        
        # ***** Create object for llm integration *****
        self.summarizer = LLMSummarize()
        
    def get_xai_tools(self, 
                      X_instance, 
                      xai_types: list, 
                      X_instance_raw: pd.DataFrame | np.ndarray = None,
                      feature_limit: int = None) -> dict:
        """
        Run specified XAI methods on a single instance and return cleaned outputs.

        Args:
            X_instance (pd.DataFrame | np.ndarray): Single-instance data for explanation.
            xai_types (list[str]): List of XAI method names to apply (e.g., ['shap']).
            X_instance_raw (pd.DataFrame | np.ndarray, optional): Raw feature values if preprocessing was applied.
            feature_limit (int, optional): Maximum number of features to return for methods like SHAP.

        Returns:
            dict[str, dict]: A mapping from XAI method name to its standardized output dictionary.
        """
        summary_xai_tools = {}
        for xai_type in xai_types:
            # ***** Currently only SHAP is supported; call corresponding helper *****
            if xai_type.lower() == 'shap':
                shap_output = self._run_shap(X_instance, X_instance_raw, feature_limit)
                shap_cleaned = standardized_shap(shap_output)
                summary_xai_tools['shap'] = shap_cleaned
            else: 
                raise ValueError(f"Unsupported XAI type: {xai_type}. Supported types: ['shap']") 
            
        return summary_xai_tools
    
    def summarize_insights(self, 
                          X_instance: pd.DataFrame | np.ndarray,
                          xai_types: list[str], 
                          feature_limit: int = None,
                          X_instance_raw: pd.DataFrame | np.ndarray = None,
                          ) -> str: 
        """
        Generate a human-readable summary of XAI outputs via the integrated LLM.

        Args:
            X_instance (pd.DataFrame | np.ndarray): Single-instance data for explanation.
            xai_types (list[str]): List of XAI method names to apply (e.g., ['shap']).
            feature_limit (int, optional): Maximum number of features to return for methods like SHAP.
            X_instance_raw (pd.DataFrame | np.ndarray, optional): Raw feature values if preprocessing was applied.

        Returns:
            str: The LLM-generated summarization text.
        """
        # ***** Obtain standardized XAI outputs for the instance *****
        xai_output = self.get_xai_tools(
            X_instance=X_instance, 
            xai_types=xai_types, 
            X_instance_raw=X_instance_raw, 
            feature_limit=feature_limit
        )
        
        # ***** Use LLM to summarize the aggregated XAI outputs *****
        result = self.summarizer.xai_summarization(xai_output)
        return result
    
    def create_conversational_explainer(self, 
                                        X_instance: pd.DataFrame | np.ndarray,
                                        xai_types: list[str], 
                                        feature_limit: int = None,
                                        X_instance_raw: pd.DataFrame | np.ndarray = None,
                                        ) -> None: 
        """
        Launch an interactive conversational loop where the user can ask follow-up questions.

        Args:
            X_instance (pd.DataFrame | np.ndarray): Single-instance data for explanation.
            xai_types (list[str]): List of XAI method names to apply (e.g., ['shap']).
            feature_limit (int, optional): Maximum number of features to return for methods like SHAP.
            X_instance_raw (pd.DataFrame | np.ndarray, optional): Raw feature values if preprocessing was applied.

        Returns:
            None
        """
        # ***** Obtain standardized XAI outputs for the instance *****
        xai_output = self.get_xai_tools(
            X_instance=X_instance, 
            xai_types=xai_types, 
            X_instance_raw=X_instance_raw, 
            feature_limit=feature_limit
        )
        
        # ***** Start the conversational explainer using LLM *****
        self.summarizer.xai_conversational(xai_output)

    def _infer_model_type(self, model: BaseEstimator) -> str:
        """
        Infer model type by checking the final estimator in a pipeline or directly.

        Args:
            model (BaseEstimator): The model or pipeline.

        Returns:
            str: "classifier", "regressor", or "unknown"
        """
        # ***** If it's a pipeline, get the final estimator *****
        if isinstance(model, Pipeline):
            final_estimator = model.steps[-1][1]
        else:
            final_estimator = model

        if isinstance(final_estimator, ClassifierMixin):
            return "classifier"
        elif isinstance(final_estimator, RegressorMixin):
            return "regressor"
        else:
            return "unknown"
    
    def _infer_feature_names(self, model: BaseEstimator, X_train: pd.DataFrame | np.ndarray) -> list: 
        """
        Attempts to infer feature names from model or training data.

        Args:
            model (BaseEstimator): Model or pipeline.
            X_train (pd.DataFrame, np.ndarray, optional): Training data.

        Returns:
            list: Feature names.

        Raises:
            ValueError: If feature names cannot be inferred.
        """
        # ***** If input is DataFrame, just return its column names *****
        if isinstance(X_train, pd.DataFrame):
            return X_train.columns.tolist()
        
        # ***** Get final estimator *****
        if isinstance(X_train, Pipeline): 
            final_estimator = model.steps[-1][1]
        else: 
            final_estimator = model     
        
        # ***** Use feature_names_in_ if available *****
        if hasattr(final_estimator, "feature_names_in_"):
            return list(final_estimator.feature_names_in_)

        # ***** Try to get feature names from ColumnTransformer inside pipeline *****
        if isinstance(model, Pipeline):
            for name, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    try:
                        # ***** Works if get_feature_names_out is implemented *****
                        return step.get_feature_names_out()
                    except AttributeError:
                        # ***** Fallback: manually build names from transformers *****
                        output_features = []
                        for transformer_name, transformer, cols in step.transformers_:
                            if transformer == 'drop':
                                continue
                            # ***** If transformer is a pipeline, take the last step *****
                            if isinstance(transformer, Pipeline):
                                transformer = transformer.steps[-1][1]
                            try:
                                # ***** Get transformed feature names *****
                                names = transformer.get_feature_names_out(cols)
                                output_features.extend(names)
                            except Exception:
                                # ***** Fallback to raw column names *****
                                output_features.extend(cols if isinstance(cols, list) else [cols])
                        return output_features
                    
        # ***** X_train is a NumPy array or similar and names can't be inferred *****
        if X_train is not None and hasattr(X_train, 'shape'):
            warnings.warn("Could not infer feature names. Using generic names (F0, F1, ...)." 
                          "Explicit feature names are required for XAI methods to work properly.")
            # raise ValueError("Feature names could not be inferred. Please pass them explicitly. Explicit feature names are required for XAI methods to work properly.")
            return [f"F{i}" for i in range(X_train.shape[1])]
        
        # ***** Fallback failure case *****
        raise ValueError("Feature names could not be inferred. Please pass them explicitly.")
    
    def _run_shap(self, X_instance: pd.DataFrame | np.ndarray,
                  X_instance_raw: pd.DataFrame | np.ndarray,
                  top_features: int = None, 
                  ) -> dict:
        """
        Compute SHAP explanations for a single instance, handling pipelines or raw estimators,
        and aggregate results into a structured dictionary.

        Args:
            X_instance (pd.DataFrame | np.ndarray): Preprocessed instance (1*n) for model prediction.
            X_instance_raw (pd.DataFrame | np.ndarray): Raw instance values (1*n) before preprocessing.
            top_features (int, optional): Number of top features to include in derived insights. Defaults to all.

        Returns:
            dict: SHAP results including metadata, instance context, base values, per-feature attributions,
                interactions, derived insights, and raw feature attributions.
        """
        shap_results: dict = {}

        # ***** Ensure background data is available for SHAP *****
        if self.X_background_for_xai is None:
            raise ValueError("X_train_for_xai (background data) is required for SHAP.")

        # ***** Determine raw feature names and values from provided raw instance or fall back to X_instance *****
        if X_instance_raw is not None:
            if isinstance(X_instance_raw, pd.DataFrame):
                raw_feature_names = list(X_instance_raw.columns)
                raw_feature_values = X_instance_raw.values[0].tolist()
            else:
                raw_feature_names = list(self.feature_names)
                raw_feature_values = X_instance_raw[0].tolist()
        else:
            # ***** Fallback to treating X_instance as raw *****
            if isinstance(X_instance, pd.DataFrame):
                raw_feature_names = list(X_instance.columns)
                raw_feature_values = X_instance.values[0].tolist()
            else:
                raw_feature_names = list(self.feature_names)
                raw_feature_values = X_instance[0].tolist()

        # ***** If model is a pipeline, separate preprocessor and estimator *****
        if isinstance(self.model, Pipeline):
            preprocessor = Pipeline(self.model.steps[:-1])
            model_only = self.model.steps[-1][1]

            # ***** Transform background data through pipeline *****
            if isinstance(self.X_background_for_xai, pd.DataFrame):
                background_transformed = preprocessor.transform(self.X_background_for_xai)
            else:
                df_bg = pd.DataFrame(self.X_background_for_xai, columns=self.feature_names)
                background_transformed = preprocessor.transform(df_bg)

            # ***** Transform the single instance through pipeline *****
            if isinstance(X_instance, pd.DataFrame):
                instance_transformed = preprocessor.transform(X_instance)
            else:
                df_inst = pd.DataFrame(X_instance, columns=self.feature_names)
                instance_transformed = preprocessor.transform(df_inst)

            # ***** Attempt to retrieve transformed feature names, or fallback to generic names *****
            try:
                transformed_feature_names = preprocessor.get_feature_names_out()
            except Exception:
                warnings.warn(
                    "Could not get transformed feature names from preprocessor. "
                    "Using generic names (F0, F1, ...). Need explicit feature names for XAI."
                )
                n_feat = background_transformed.shape[1]
                transformed_feature_names = [f"F{i}" for i in range(n_feat)]

            explained_feature_level = "direct_model_input"

        else:
            # ***** Model is not a pipeline; treat raw inputs as transformed *****
            model_only = self.model
            if isinstance(self.X_background_for_xai, pd.DataFrame):
                background_transformed = self.X_background_for_xai.values
            else:
                background_transformed = np.asarray(self.X_background_for_xai)

            if isinstance(X_instance, pd.DataFrame):
                instance_transformed = X_instance.values
            else:
                instance_transformed = np.asarray(X_instance)

            transformed_feature_names = raw_feature_names
            explained_feature_level = "raw_user_input"

        # ***** Build a predict function for KernelExplainer compatibility *****
        def predict_fn(X_arr: np.ndarray) -> np.ndarray:
            if isinstance(self.model, Pipeline):
                df_input = pd.DataFrame(X_arr, columns=self.feature_names)
                return self.model.predict(df_input)
            else:
                return model_only.predict(X_arr)

        # ***** Choose appropriate SHAP explainer based on model type *****
        tree_models = (
            DecisionTreeClassifier, DecisionTreeRegressor,
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingRegressor, XGBClassifier, LGBMClassifier
        )
        linear_models = (LinearRegression, LogisticRegression)

        if isinstance(model_only, tree_models):
            explainer = shap.TreeExplainer(model_only, background_transformed)
            shap_values = explainer(instance_transformed, check_additivity=False)

        elif isinstance(model_only, linear_models):
            explainer = shap.LinearExplainer(
                model_only,
                background_transformed,
                feature_perturbation="interventional"
            )
            shap_values = explainer(instance_transformed)

        else:
            explainer = shap.KernelExplainer(predict_fn, background_transformed)
            shap_values = explainer(instance_transformed, check_additivity=False)

        # ***** Gather prediction label and probabilities if available *****
        model_pred: dict[str, Any] = {}
        if hasattr(self.model, "predict_proba"):
            pred_label = self.model.predict(X_instance)
            pred_proba = self.model.predict_proba(X_instance)[0].tolist()
            model_pred["label"] = pred_label[0] if hasattr(pred_label, "__len__") else pred_label
            model_pred["probabilities"] = pred_proba
        else:
            pred_label = self.model.predict(X_instance)
            model_pred["label"] = float(pred_label[0] if hasattr(pred_label, "__len__") else pred_label)
            model_pred["probabilities"] = None

        # ***** Extract base value(s) from the explainer *****
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val_list = base_val.tolist()
        else:
            base_val_list = float(base_val)

        # ***** Extract per-feature SHAP values for the instance *****
        raw_shap_vals = shap_values.values
        if isinstance(raw_shap_vals, np.ndarray) and raw_shap_vals.ndim == 2:
            per_feature_vals = raw_shap_vals[0].tolist()
        else:
            per_feature_vals = np.array(raw_shap_vals[0][0]).tolist()

        # ***** Attempt to compute SHAP interaction values; otherwise set None *****
        try:
            interaction_vals = explainer.shap_interaction_values(instance_transformed)
            if isinstance(interaction_vals, np.ndarray) and interaction_vals.ndim == 3:
                interactions_2d = interaction_vals[0]
            else:
                interactions_2d = np.array(interaction_vals[0][0])
            interactions_list = interactions_2d.tolist()
        except Exception:
            interactions_list = None

        # ***** Identify top-k features by absolute SHAP value *****
        abs_vals = [abs(v) for v in per_feature_vals]
        ranked_idx = np.argsort(abs_vals)[::-1]
        top_k = top_features if top_features is not None else len(transformed_feature_names)
        top_k = min(top_k, len(transformed_feature_names))

        top_k_list: list[dict] = []
        raw_transformed_vals = instance_transformed[0].tolist()
        for rank, idx in enumerate(ranked_idx[:top_k], start=1):
            tname = transformed_feature_names[idx]
            tval = raw_transformed_vals[idx]
            shap_val = per_feature_vals[idx]

            # ***** Map each transformed feature back to its raw feature when possible *****
            raw_feat_name: Optional[str] = None
            raw_feat_value: Optional[Any] = None
            for rf_name, rf_val in zip(raw_feature_names, raw_feature_values):
                if rf_name in tname:
                    raw_feat_name = rf_name
                    raw_feat_value = rf_val
                    break

            top_k_list.append({
                "model_input_name": tname,
                "model_input_value": tval,
                "shap_value": shap_val,
                "rank": rank,
                "original_raw_feature_name": raw_feat_name,
                "original_raw_feature_value": raw_feat_value
            })

        # ***** Prepare force plot data in transformed feature space *****
        force_plot_data: dict[str, Any] = {
            "base_value": base_val_list,
            "shap_values": per_feature_vals,
            "feature_values": raw_transformed_vals,
            "feature_names": self._safe_to_list(transformed_feature_names)
        }

        # ***** Aggregate SHAP values back to raw features *****
        raw_feature_attributions = self._aggregate_to_raw_features(
            per_feature_vals,
            transformed_feature_names,
            raw_feature_names
        )

        # ***** Assemble final SHAP results dictionary *****
        shap_results["metadata"] = {
            "model_class": str(type(self.model)),
            "explainer_type": type(explainer).__name__,
            "explained_feature_level": explained_feature_level
        }
        shap_results["instance_context"] = {
            "feature_names": raw_feature_names,
            "feature_values": raw_feature_values,
            "prediction": model_pred
        }
        shap_results["base_values"] = {"default": base_val_list}
        shap_results["attribution_values"] = {
            "per_feature": per_feature_vals,
            "interactions": interactions_list
        }
        shap_results["derived_insights"] = {
            "top_k_features": top_k_list,
            "plot_data": {"force_plot": force_plot_data}
        }
        shap_results["raw_explainer_output"] = shap_values
        shap_results["raw_feature_attributions"] = raw_feature_attributions

        return shap_results


    def _safe_to_list(self, x: Any) -> list:
        """
        Convert array-like object to a Python list, if possible.

        Args:
            x (Any): An array-like object with a .tolist() method or an iterable.

        Returns:
            list: Converted list from x.
        """
        return x.tolist() if hasattr(x, "tolist") else list(x)


    def _aggregate_to_raw_features(
        self,
        per_feature_vals: list[float],
        transformed_feature_names: list[str],
        raw_feature_names: list[str]
    ) -> dict[str, float]:
        """
        Sum SHAP values of transformed features back into their corresponding raw feature buckets.

        Args:
            per_feature_vals (list[float]): SHAP values for each transformed feature.
            transformed_feature_names (list[str]): Names of features after preprocessing.
            raw_feature_names (list[str]): Original raw feature names.

        Returns:
            dict[str, float]: Aggregated SHAP values mapped to each raw feature.
        """
        raw_feature_attributions: dict[str, float] = {rf: 0.0 for rf in raw_feature_names}

        # ***** For each transformed feature, add its SHAP value to the matching raw feature *****
        for idx, tname in enumerate(transformed_feature_names):
            shap_val = per_feature_vals[idx]
            for raw_feat in raw_feature_names:
                if raw_feat in tname:
                    raw_feature_attributions[raw_feat] += shap_val
                    break

        return raw_feature_attributions