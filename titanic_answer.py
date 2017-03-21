# The model to use
from sklearn.ensemble import RandomForestRegressor

# The error metric
from sklearn.metrics import roc_auc_score

# Efficient data structure
import pandas as pd
import matplotlib.pyplot as plt

# Data import
x = pd.read_csv('train.csv')
y = x.pop('Survived')

test = pd.read_csv('test.csv')



# See Numeric data, note: fill age column
# print(x.describe())
x['Age'].fillna(x.Age.median(), inplace=True)

# Get Numeric data, no objects
numeric_variables = list(x.dtypes[x.dtypes != 'object'].index)
print(x[numeric_variables].head())

# Create model, fit x & y, find oob_score
model = RandomForestRegressor(n_estimators=100, oob_score=True)
model.fit(x[numeric_variables], y)
print(model.oob_score_)

# First prediction: only uses numerical data
y_oob = model.oob_prediction_
print("C-Stat: ", roc_auc_score(y, y_oob))


# Show categorical data
def describe_categorical(x):
    """
    Similar to describe but uses object data
    """
    from IPython.display import display
    display(x[x.columns[x.dtypes == 'object']].describe())
describe_categorical(x)

# Removing unnecessary data
x.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)


# cabin has missing data, function to clean
def clean_up(x):
    try:
        return x[0]
    except TypeError:
        return 'None'
x['Cabin'] = x.Cabin.apply(clean_up)

# replacing var with new var with 'na'
categorical_variables = ['Sex', 'Cabin', 'Embarked']
for var in categorical_variables:
    x[var].fillna('na', inplace=True)
    # Creating dummy arrays
    dummies = pd.get_dummies(x[var], prefix=var)
    # drop main var, include dummies
    x = pd.concat([x, dummies], axis=1)
    x.drop([var], axis=1, inplace=True)

# Just in case columns are missing in display
def print_all(x, max_rows=10):
    from IPython.display import display
    display(x)
#print(print_all(x))

model2 = RandomForestRegressor(100, oob_score=True, n_jobs=-1)
model2.fit(x, y)
print("C-Stat: ", roc_auc_score(y, model2.oob_prediction_))

feature_importances = pd.Series(model2.feature_importances_, index=x.columns)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind='barh', figsize=(7,6))


def graph_feature_importances(model2, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
    """
    By Mike Bernico
    Edited by Doug North

    :param model2:
    :param feature:
    :param autoscale:
    :param headroom:
    :param width:
    :param summarized_columns:
    :return:
    """
    if autoscale:
        x_scale = model.feature_importances_.max() + headroom
    else:
        x_scale = 1
    feature_dict = dict(zip(feature_names, model2.feature_importances_))

    if summarized_columns:
        for col_name in summarized_columns:
            sum_value = sum(x for i, x in feature_dict.items() if col_name in i)
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i]
            for i in keys_to_remove:
                feature_dict.pop(i)
    # Could not get dictionary to sort so converted to list
            feature_dict[col_name] = sum_value
    dict_key_list = []
    dict_val_list = []
    for key in feature_dict.keys():
        dict_key_list.append(key)
    for value in feature_dict.values():
        dict_val_list.append(value)

    results = pd.Series(dict_val_list, index=dict_key_list)
    results.sort_values(inplace=True)
    results.plot(kind='barh', figsize=(width, len(results)/4), xlim=(0, x_scale))


print(graph_feature_importances(model2, x.columns, summarized_columns=categorical_variables))

# Test to pick optimal min_sample_leaf
# results = []
# min_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# for min_samples in min_leaf_options:
    # model3 = RandomForestRegressor(n_estimators=1000,
                                   # oob_score=True,
                                   # n_jobs=-1,
                                   # random_state=42,
                                   # max_features="auto",
                                   # min_samples_leaf=min_samples)
    # model3.fit(x,y)
    # print('min_samples: ', min_samples)
    # print('C-Stat: ', roc_auc_score(y, model3.oob_prediction_))
    # results.append(roc_auc_score(y, model3.oob_prediction_))
# pd.Series(results, min_leaf_options).plot()


# Final Model
final_model = RandomForestRegressor(n_estimators=1000,
                                   oob_score=True,
                                   n_jobs=-1,
                                   random_state=42,
                                   max_features="auto",
                                   min_samples_leaf=5)
final_model.fit(x, y)
print('Final C-Stat: ', roc_auc_score(y, final_model.oob_prediction_))






