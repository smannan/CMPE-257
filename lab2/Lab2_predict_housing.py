
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn_features.transformers import DataFrameSelector


# ## Explore data
# ### Based on 1990 California housing data

# In[ ]:

dataset = pd.read_csv('labII_housing_data.csv')


# In[ ]:

dataset.head()


# ### 20,640 districts with 10 features per district
# ### 9 continuous and 1 categorical feature
# ### ocean_proximity = estimate for how close the district is to the ocean
# ### Continuous data is given in terms of district

# In[ ]:

print ('Rows x features {0}'.format(dataset.shape))


# ### Average house is 30 years old
# ### Average total rooms per district is 2,635
# ### Average population per district is 1,425
# ### Average number of houses per district is 499
# ### Average median income per district is 3.87 -> between 0.5 and 15, not in dollars
# ### Average median house value per district is 206,855 dollars

# In[ ]:

dataset.describe()


# ### Most houses are inland
# ### About the same for near ocean and bay
# ### Five outliers are islands

# In[ ]:

dataset['ocean_proximity'].value_counts()


# ### 207 missing total bedrooms

# In[ ]:

dataset.info()


# ### Data is very tail-heavy -> Numerical values are not evenly distributed into a bell-curve

# In[ ]:

get_ipython().magic(u'matplotlib inline')
dataset.hist(bins=50, figsize=(20, 15))
plt.show()


# plt.hist(dataset['median_income'])
# plt.title('Histogram of Median Income')
# plt.show()

# In[ ]:

# median income is closely related to housing price so we want to ensure
# all incomes are fairly represented in the training set
# to prevent the model from being skewed
dataset['income_cat'] = np.ceil(dataset['median_income'] / 1.5)
dataset['income_cat'].where(dataset['income_cat'] < 5, 5.0, inplace=True)


# In[ ]:

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


# In[ ]:

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# ### Visualize Training Data

# In[ ]:

housing = strat_train_set.copy()


# ### Most houses from the bay or LA with some near Fresno/Bakersfield

# In[ ]:

housing.plot(kind='scatter', x='latitude', y='longitude', alpha=0.1)
plt.show()


# ### Houses are more expensive where population is denser

# In[ ]:

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population",
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
plt.show()


# In[ ]:

corr_matrix = housing.corr()


# ### Median house value is most correlated to median income
# ### No strong correlation with the other numerical features

# In[ ]:

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()


# ### Convert housing metrics to per household
# ### Transform housing measurements to fit domain

# In[ ]:

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# ### There is still a strong correlation to median income
# ### But number of rooms per house is now a better indication of value
# ### As well as total rooms per house vs. total rooms per district
# ### Transforming these metrics will make it easier to learn good predictors of housing value

# In[ ]:

housing.corr()["median_house_value"].sort_values(ascending=False)


# ### Prepare dataset for Machine Learning

# In[ ]:

# remove the variable we are trying to predict from the training data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[ ]:

# Transforms housing metrics in a way that can be plugged into a pipeline
# Transform into more useful variables for predictions
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[ ]:

# Convert categorical feautres into numerical ones that can be used by an algorithm
# Uses one-hot encoding
class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)


# In[ ]:

# create pipelines to automatically chain transformations together
# and combine numerical with categorical
num_attribs = list(housing.columns)
num_attribs.remove("ocean_proximity")
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
     ('selector', DataFrameSelector(num_attribs)),
     # fill missing values with median
     ('imputer', SimpleImputer(strategy="median")),
     ('attribs_adder', CombinedAttributesAdder()),
     # normalize the data so numerical features are in the same ranges   
     ('std_scaler', StandardScaler()),
     ])

cat_pipeline = Pipeline([
     ('selector', DataFrameSelector(cat_attribs)),
     ('label_binarizer', CustomLabelBinarizer()),
     ])

full_pipeline = FeatureUnion(transformer_list=[
     ("num_pipeline", num_pipeline),
     ("cat_pipeline", cat_pipeline),
     ])


# In[ ]:

housing_prepared = full_pipeline.fit_transform(housing)


# In[ ]:

housing_prepared


# In[ ]:

housing_prepared.shape


# ### Train the models

# In[ ]:

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# use an ensemble algorithm to improve accuracy
forest_reg = RandomForestRegressor()
# cross-validate results
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=3)
# use an RMSE scoring to evaluate
# lower scores = better
rmse_scores = np.sqrt(-scores)


# In[ ]:

np.mean(rmse_scores)


# In[ ]:

# use a grid search to find the best parameters for the model
from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[ ]:

feature_importances = grid_search.best_estimator_.feature_importances_


# In[ ]:

# evaluate the data on the testing set - witheld during training and validation
from sklearn.metrics import mean_squared_error

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6


# In[ ]:

final_rmse


# In[ ]:



