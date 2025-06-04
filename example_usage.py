from engine.xai_explainer import XAIExplainer
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# EXAMPLE USAGE WITH PIPELINE
data = pd.DataFrame({
        'square_feet': [1000, 1200, 1500, 900, 1100],
        'bedrooms': [2, 3, 3, 1, 2],
        'neighborhood': ['A', 'B', 'A', 'C', 'B']
    })
target = pd.Series([250000, 300000, 380000, 200000, 280000])

# Define preprocessing
numeric_features = ['square_feet', 'bedrooms']
categorical_features = ['neighborhood']

preprocessor = ColumnTransformer(transformers=[
   ('num', StandardScaler(), numeric_features),
   ('cat', OneHotEncoder(), categorical_features)
])

# Define pipeline with RandomForest
pipeline = Pipeline(steps=[
   ('preprocessor', preprocessor),
   ('regressor', LinearRegression())
])

# Fit the pipeline
pipeline.fit(data, target)

# Instance to explain
instance_to_explain = pd.DataFrame({
   'square_feet': [1300],
   'bedrooms': [2],
   'neighborhood': ['B']
})

explainer = XAIExplainer(model=pipeline,
                             X_train_for_xai=data)

# One shot for generating summary insights
# result = explainer.summarize_insights(X_instance=instance_to_explain, 
#                                       xai_types=['shap'])
# print(result)

# Conversational style with chat history
# explainer.create_conversational_explainer(X_instance=instance_to_explain, 
#                                        xai_types=['shap'])

# EXAMPLE USAGE WITH NO PIPELINE
X = pd.DataFrame({
    'square_feet': [1000, 1200, 1500, 900, 1100],
    'bedrooms':    [2,    3,    3,    1,    2], 
    'years_built': [1990, 2005, 1985, 2010, 1998]
    })
y = pd.Series([250000, 300000, 380000, 200000, 280000])

# Initialize and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the RandomForestRegressor on the scaled data
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_scaled, y)

# New instance for prediction
instance = pd.DataFrame({'square_feet': [1300], 'bedrooms': [2], 'years_built':[2002]})
instance_scaled = scaler.transform(instance)

explainer_v2 = XAIExplainer(
                        model=rf,
                        X_train_for_xai=X_scaled,
                        feature_names=list(X.columns)
                        )

# One shot for generating summary insights
# result = explainer_v2.summarize_insights(X_instance=instance_scaled, 
#                                       xai_types=['shap'])
# print(result)

# Conversational style with chat history
explainer_v2.create_conversational_explainer(X_instance=instance_scaled, 
                                       xai_types=['shap'])
