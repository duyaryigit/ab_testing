
#####################################################
# Comparison of AB Test and Conversion of Bidding Methods
#####################################################

#####################################################
# Business Problem
#####################################################

# Facebook recently introduced a new bidding type, "average bidding", as an alternative to the existing bidding type called "maximum bidding".
# One of our clients, bombabomba.com, decided to test this new feature and would like to do an A/B test to see if averagebidding converts more than maximum bidding.
# It is waiting for you to analyze the results of this A/B test.
# The ultimate success criterion for Bombambomba.com is Purchase.
# Therefore, the focus should be on the Purchase metric for statistical testing.

#####################################################
# Dataset Story
#####################################################

# In this data set, which includes the website information of a company, there is information such as the number of advertisements that users see and click,
# as well as earnings information from here. There are two separate data sets, the Control and Test group.
# These datasets are on separate pages of ab_testing.xlsxexcel.
# Maximum Bidding was applied to the control group and Average Bidding was applied to the test group.


# impression: Ad views
# Click: Number of clicks on the displayed ad
# Purchase: Number of products purchased after ads clicked
# Earning: Earnings after purchased products

#####################################################
# Project Tasks
#####################################################

#####################################################
# Task 1: Prepare and Understand Data (Data Understanding)
#####################################################

# 1:  Read the data set ab_testing_data.xlsx consisting of control and test group data. Assign control and test group data to separate variables.


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

dataframe_control = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
dataframe_test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

df_control = dataframe_control.copy()
df_test = dataframe_test.copy()

# 2: Analyze control and test group data.


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df_control)
check_df(df_test)

df_control.describe([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T
df_test.describe([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T

pd.plotting.scatter_matrix(df_control)
plt.show()

pd.plotting.scatter_matrix(df_test)
plt.show()

# 3: After the analysis, combine the control and test group data using the concat method.

df_control["group"] = "control"
df_test["group"] = "test"
df_control.head()
df = pd.concat([df_control,df_test], axis=0,ignore_index=True)
df.head(50)


for col in df.columns:
    sns.boxplot(x="group",y=col,hue="group",data=df)
    plt.show()

df.info()

#####################################################
# Task 2:  Defining the A/B Test Hypothesis
#####################################################

# 1: Define the hypothesis.

# H0 : M1 = M2 (There is no difference between the purchasing averages of the control group and the test group.)
# H1 : M1!= M2 (There is a difference between the purchasing averages of the control group and the test group.)

df_control["Purchase"].mean()
df_test["Purchase"].mean()

# 2: Analyze the purchase (gain) averages for the control and test group

df.groupby("group").agg({"Purchase": "mean"})


#####################################################
# 3: Performing Hypothesis Testing
#####################################################

# 1: Before testing the hypothesis, check the assumptions. These are Assumption of Normality and Homogeneity of Variance.

# Test separately whether the control and test groups comply with the normality assumption, over the Purchase variable.

# Normality Assumption:

# H0: The assumption of normal distribution is provided.
# H1: The assumption of normal distribution is not provided.
# p < 0.05 H0 REJECTION
# p > 0.05 H0 IRREFUTABLE
# Is the assumption of normality according to the test result provided for the control and test groups?
# Interpret the p-values obtained.

test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"]) # Control group
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.5891

# HO cannot be denied. The values of the control group provide the assumption of normal distribution.

shapiro(df.loc[df.index[0:40],"Purchase"])

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"]) # Test group
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.1541

# HO cannot be denied. The values of the control group provide the assumption of normal distribution.

# Variance Homogeneity:
# H0: Variances are homogeneous.
# H1: Variances are not homogeneous.
# p < 0.05 H0 REJECTION
# p > 0.05 H0 IRREFUTABLE
# Test whether the homogeneity of variance is provided for the control and test groups over the Purchase variable.
# Is the assumption of normality provided according to the test result? Interpret the p-values obtained.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.1083
# HO cannot be denied. The values of the Control and Test groups provide the assumption of variance homogeneity.
# Variances are Homogeneous.

# 2nd solution

levene(df.loc[df.index[0:40], "Purchase"],
       df.loc[df.index[40::], "Purchase"])

# 2: Select the appropriate test according to the Normality Assumption and Variance Homogeneity results

# Since the assumptions are provided, an independent two-sample t-test (parametric test) is performed.
# H0: M1 = M2 (There is no statistically significant difference between the control group and test group purchasing averages.)
# H1: M1 != M2 (There is a statistically significant difference between the control group and test group purchasing averages)
# p < 0.05 H0 REJECTION
# p > 0.05 H0 IRREFUTABLE

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 3: Considering the p_value obtained as a result of the test, interpret whether there is a statistically significant difference between
# the purchasing averages of the control and test groups.

'''
p-value=0.3493

HO cannot be denied. There is no statistically significant difference between the purchasing averages of the control and test groups.
'''

##############################################################
# Task 4 : Analysis of Results
##############################################################

# 1: Which test did you use, state the reasons.

'''
It has two assumptions.

1-) Normality Assumption
2-) Variance Homogeneity

In the AB test, first of all, normality assumption check with the shapiro test and variance homogeneity control should be done with Levene test.

Afterwards, if the assumptions are providing, 
the parametric test is performed with 2 independent samples t, and if the normality assumption is not providing, the non-parametric test is performed with mannwhitneyu.

I used t-test(parametric) and p_values were above 0.05.
'''

# 2: Advise the customer according to the test results you have obtained.

'''
You can use both bidding systems because there is no statistically significant difference between the purchasing averages of the control and test groups.
The results obtained before the test do not indicate a clear result.
'''