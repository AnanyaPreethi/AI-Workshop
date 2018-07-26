
# coding: utf-8

# KNN Classification - Machine Learning Model
# 
# The k-nearest-neighbor algorithm is conceptually simple.Given a known set of cases, a new case is classified by majority vote of the K (where , etc.) points nearest to the values of the new case; that is, the nearest neighbors of the new case.

# In[25]:


#Here we are using Iris dataset which is famous for the classification of flowers in statistics
from statsmodels.api import datasets
#Now in the below line, we are read from the already existing dataset called "Iris"
flowers = datasets.get_rdataset("iris")
flowers.data.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']
flowers.data.head()
print(flowers.data)

#Next, you will execute the code in the cell below to show the data types of each column.
#flowers.data.dtypes


# In[7]:


#So now that we know the data types, we will go ahead and determine the number of unique categories, 
#and number of cases for each category, for our label coloumn "Species"
flowers.data['calc'] = 1
flowers.data[['Species', '']].groupby('Species').count()


# In[26]:


#Next, you will create some plots to see how the classes might, or might not, be well separated by the value of the features.
#For now, we will just scatter the 2 variables
get_ipython().magic('matplotlib inline')
def plot_flowers(flowers, col1, col2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lmplot(x = col1, y = col2, 
               data = flowers, 
               hue = "Species", 
               fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Flower species shown by color')
    plt.show()
plot_flowers(flowers.data, 'Petal_Width', 'Petal_Length')
plot_flowers(flowers.data, 'Sepal_Width', 'Sepal_Length')

