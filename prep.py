#%%
# Step 1: 
# College Completion Dataset Question:
# Which schools are likely to have a high graduation rate?
# Campus Recruitment Dataset Question:
# Given a student’s academic performance, specialization, and 
# work experience, can we predict whether they will be placed in a job?

#%% [markdown]
## Step 2:
#### Business Metrics:
# College Completion Dataset:
# - Graduation Rate: The percentage of students who graduate within 4 years.
#
# Campus Recruitment Dataset:
# - Placement Rate: The rate of whether a student secures a job.


# %%
## Data Preparation:
import pandas as pd
import numpy as np

# %%
college_completion = pd.read_csv('cc_institution_details.csv')
college_completion.head()
# %%
job_placement = pd.read_csv('https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv')
job_placement.head()

# %% [markdown]
### First, we will focus on the college completion dataset.
# %%
# Correct variable type/class as needed:
college_completion.info()
college_completion['control'] = college_completion['control'].astype('category')
college_completion['level'] = college_completion['level'].astype('category')
college_completion['basic'] = college_completion['basic'].astype('category')
college_completion['state'] = college_completion['state'].astype('category')
#%%
# collapse factor levels as needed:
college_completion['control'].value_counts()
# 3 distinct categories, but we can collapse them 
# into 2 categories: public and private.
college_completion['control'] = college_completion['control'].apply(
    lambda x: 'public' if x == 'Public' else 'private')

college_completion['basic'].value_counts()
# lots of categories, let's collapse them into
# 4: Associate's, Baccalaureate, Master's, and Other.
college_completion['basic'] = college_completion['basic'].apply(
    lambda x: 'Associates' if 'Associates' in x 
              else 'Baccalaureate' if 'Baccalaureate' in x 
              else 'Masters' if 'Masters' in x 
              else 'Other'
)


#%%
# dropping irrelevant columns:
drop_cols_cc = [
    'index', 'unitid','city', 'nicknames', 'site', 
    'vsa_year', 'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first',
    'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first',
    'vsa_grad_after6_first', 'vsa_grad_elsewhere_after6_first',
    'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first',
    'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer',
    'vsa_enroll_after4_transfer', 'vsa_enroll_elsewhere_after4_transfer',
    'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer',
    'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer',
    'similar', 'counted_pct', 'hbcu', 'flagship', 'med_sat_value', 'med_sat_percentile'
]
college_completion = college_completion.drop(columns=drop_cols_cc)
# these columns are do not contribute to 
# the analysis and may introduce noise.
# some have lots of missing values, some
# are identifiers, and some are not relevant.
college_completion.info()

# %%
# One-hot encoding factor variables:
college_completion = pd.get_dummies(college_completion, columns=['control', 'basic', 'state', 'level'], drop_first=True)
# This will create new binary columns for each 
# category in the categorical variables.

#%%
# %%
# Normalize the continuous variables:
college_completion.info()
# The continuous variables are:
continuous_vars = [
    'student_count',
    'fte_value',
    'aid_value',
    'endow_value',
    'pell_value',
    'retain_value',
    'ft_fac_value',
     'cohort_size'
]
# not going to normalize our target variables
# (graduation rates) since they are already
# normalized as percentages.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
college_completion[continuous_vars] = scaler.fit_transform(college_completion[continuous_vars])
# This will standardize the continuous variables to have a 
# mean of 0 and a standard deviation of 1.

# %%
#create target variable if needed:
# We need to define "high graduation rate".
# Let's say a high graduation rate is above 60%.
college_completion['high_grad_rate'] = (college_completion['grad_150_value'] > 60).astype(int)
# %%
# Calculate the prevalence of the target variable:

prevalence = college_completion['high_grad_rate'].sum() / len(college_completion)
print(f"Prevalence of high graduation rate: {prevalence:.2%}")
# %%
# Create the necessary data partitions 
# (Train,Tune,Test):
from sklearn.model_selection import train_test_split

# separate predictors and target variable:
X = college_completion.drop(columns=['grad_150_value', 'high_grad_rate'])
# assuming the model will be classification,
# we only need the binary target variable.
y = college_completion['high_grad_rate']

# First, split off the test set from the train_tune set:
X_train_tune, X_test, y_train_tune, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
# Then, split the train_tune set into train and tune sets:
X_train, X_tune, y_train, y_tune = train_test_split(
    X_train_tune, y_train_tune,
    test_size=0.25,  # 0.25 * 80% = 20% of original
    random_state=42,
    stratify=y_train_tune
)
# Now we have 60% train, 20% tune, and 20% test sets.
print(f"Train set size: {len(X_train)}")
print(f"Tune set size: {len(X_tune)}")
print(f"Test set size: {len(X_test)}")


# %% [markdown]
### Now, we will focus on the campus recruitment dataset.
job_placement.head()
job_placement.info()

# %%
# Correct variable type/class as needed:
# let's look at gender, specialization, and status:
job_placement['gender'].value_counts()
job_placement['specialisation'].value_counts()
job_placement['status'].value_counts()
# they look to all be binary categorical variables,
# so we can convert them to category type:
job_placement['gender'] = job_placement['gender'].astype('category')
job_placement['specialisation'] = job_placement['specialisation'].astype('category')
job_placement['status'] = job_placement['status'].astype('category')

#%%
# # let's look at work experience, degree type
job_placement['workex'].value_counts()
job_placement['degree_t'].value_counts()
# both categorical, let's convert them
job_placement['workex'] = job_placement['workex'].astype('category')
job_placement['degree_t'] = job_placement['degree_t'].astype('category')

#%%
# # let's now look at hsc_b, hsc_s, and ssc_b, 
job_placement['hsc_b'].value_counts()
job_placement['hsc_s'].value_counts()
job_placement['ssc_b'].value_counts()
# these are all categorical variables, let's conver them:
job_placement['hsc_b'] = job_placement['hsc_b'].astype('category')
job_placement['hsc_s'] = job_placement['hsc_s'].astype('category')
job_placement['ssc_b'] = job_placement['ssc_b'].astype('category')



# %%
#collapse factor levels as needed:
# All of our categorical variables have 2 or 3
# categories, so we don't need to collapse any.

#%%
# one-hot encoding factor variables:
categorical_vars = ['gender', 'specialisation', 'status', 'workex', 'degree_t', 'hsc_b', 'hsc_s', 'ssc_b']
job_placement = pd.get_dummies(job_placement, columns=categorical_vars, drop_first=True)

#%%
# normalize the continuous variables:
# The continuous variables are:
continuous_vars_jp = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
job_placement[continuous_vars_jp] = scaler.fit_transform(job_placement[continuous_vars_jp])

#%%
# drop unneeded variables
# let's see if any have lots of missing values:
job_placement.isnull().sum()
# only salary has missing values, so we can
# drop it since it's not a predictor variable:
job_placement = job_placement.drop(columns=['salary'])


# %%
#create target variable if needed
# In this case, employment status is already present.
# We will convert it to binary: placed (1) vs not placed (0).
job_placement['placed'] = (job_placement['status_Placed'] == 1).astype(int)

#%%
#Calculate the prevalence of the target variable:
prevalence_jp = job_placement['placed'].sum() / len(job_placement)
print(f"Prevalence of placement: {prevalence_jp:.2%}")

# %%
# Create the necessary data partitions
# same as before, we will first separate 
# train_tune from test, then split train_tune
# into train and tune sets.

# first, separate predictors and target variables:

X_jp = job_placement.drop(columns=['status_Placed', 'placed'])
y_jp = job_placement['placed']
# First, split off the test set from the train_tune set:
X_train_tune_jp, X_test_jp, y_train_tune_jp, y_test_jp = train_test_split(
    X_jp, y_jp, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_jp
)
# Then, split the train_tune set into train and tune sets:
X_train_jp, X_tune_jp, y_train_jp, y_tune_jp = train_test_split(
    X_train_tune_jp, y_train_tune_jp,
    test_size=0.25,  
    random_state=42,
    stratify=y_train_tune_jp
)
# Now we have 60% train, 20% tune, and 20% test sets.
print(f"Train set size: {len(X_train_jp)}")
print(f"Tune set size: {len(X_tune_jp)}")
print(f"Test set size: {len(X_test_jp)}")

# %%
# Step 3:
# What do your instincts tell you about the data. 
# Can it address your problem, what areas/items are 
# you worried about? 

# After preparing my data, I feel that both datasets are
# reasonably well-suited to address the respective problems.
# However, I do have a couple concerns:
# 1. Feature Relevance: While I've included a variety of features,
#    I'm not entirely sure if all of them are relevant predictors
#    for graduation rates or job placement. Further feature selection
#    techniques may be needed.
# 2. Missing Data: Originally, my research question involved using 
# SAT scores, but due to a high amount of missing values, 
# I had to switch questions. 

