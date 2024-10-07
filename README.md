## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df= pd.read_csv("/content/Encoding Data.csv")
df.head()
 ```
![Screenshot 2024-10-07 144954](https://github.com/user-attachments/assets/5b29cfc4-27e6-4fba-a297-12a896cc0be4)
```
df.tail()
```
![Screenshot 2024-10-07 145109](https://github.com/user-attachments/assets/747956e6-b2a9-4c34-8126-6f8f633968cd)

```
df.describe()
```
![Screenshot 2024-10-07 145159](https://github.com/user-attachments/assets/922bfb78-1577-4890-9a86-deb7f407850c)
```
df.info()
```
![Screenshot 2024-10-07 145314](https://github.com/user-attachments/assets/63b13516-262e-4f17-bcfc-92e586a0cdee)
```
df.shape
```
![Screenshot 2024-10-07 145358](https://github.com/user-attachments/assets/a002a417-0574-4b04-8e15-3df85e475f10)
```
df
```
![Screenshot 2024-10-07 145433](https://github.com/user-attachments/assets/f4316a42-3d76-4a69-ba66-1b26831c4e1b)
```
#ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot','Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-10-07 145518](https://github.com/user-attachments/assets/bfe1c9a5-4b81-423f-831c-18e6a443d16f)
```
df['bo2']=oe.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-07 145559](https://github.com/user-attachments/assets/7a9e9a7d-3503-4ee6-b68d-2318f91ef2cc)

```
#label Encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-07 145644](https://github.com/user-attachments/assets/b48a4e3b-e33e-4f11-8cf3-47f478dd603c)
```
#One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False) # sparse has been replaced by sparse_output
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-10-07 145731](https://github.com/user-attachments/assets/602116aa-e007-4908-b641-50feda29e2e8)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-10-07 145819](https://github.com/user-attachments/assets/fe816538-bdb6-4534-a413-fdcc80afb568)
```
!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
import pandas as pd
df= pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-10-07 145902](https://github.com/user-attachments/assets/eb7333e6-ca23-4132-8467-250a64b80b80)
```
#Binary encoder
be= BinaryEncoder() 
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-07 145945](https://github.com/user-attachments/assets/a437fec9-3867-4390-9d81-da5e7f602dd7)
```
#target encoder
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc

```
![Screenshot 2024-10-07 150026](https://github.com/user-attachments/assets/931f22b6-708d-49f2-946e-7b5587d5dcc8)
```
#Feature Transformation
import pandas as pd
from scipy import stats
import numpy as np
df= pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-10-07 150122](https://github.com/user-attachments/assets/d4fa1626-af85-4ce2-93a1-17ce1c7b947d)
```
df.info()
```
![Screenshot 2024-10-07 150219](https://github.com/user-attachments/assets/1337fd80-0977-4d30-b291-503794b02fe3)

```
df.describe()
```
![Screenshot 2024-10-07 150305](https://github.com/user-attachments/assets/9dc092db-28a3-4a6d-a3d1-7f0a0ea15df7)
```
df.size
```
![Screenshot 2024-10-07 150337](https://github.com/user-attachments/assets/f7a541b9-11e4-4b2c-881d-29dcb81c9d66)
```
df.skew()
```
![Screenshot 2024-10-07 150412](https://github.com/user-attachments/assets/35afdc8d-b143-457f-93d2-586d0235fb8a)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-10-07 150450](https://github.com/user-attachments/assets/f832d9e0-94e4-4f2b-be74-9667a2575fe7)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-10-07 150537](https://github.com/user-attachments/assets/4d628627-1053-4238-a71c-55c0653c29a3)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-10-07 150625](https://github.com/user-attachments/assets/94da9e49-dd69-4f0d-ae8f-4d52ee9b8c50)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-10-07 150733](https://github.com/user-attachments/assets/ff9ea8ea-6401-45dc-89d3-306945bdc486)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-07 150818](https://github.com/user-attachments/assets/caa4c3cd-04bb-4701-8404-1cf9c91dae07)
```
df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![Screenshot 2024-10-07 150901](https://github.com/user-attachments/assets/494b89be-21e0-4fc7-b052-aec10d2064ac)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-10-07 150940](https://github.com/user-attachments/assets/499f38e8-3411-4f69-8558-0591642a259c)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])


sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
![Screenshot 2024-10-07 151040](https://github.com/user-attachments/assets/5ecf41f1-9c49-43d4-a11c-b769bfafefd4)
```
df
```
![Screenshot 2024-10-07 151121](https://github.com/user-attachments/assets/89cf10ca-e3bd-4be7-85bc-3dac0e2ff485)

```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-10-07 151159](https://github.com/user-attachments/assets/368e86fe-13b6-4273-bfd3-86e1de995acc)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-10-07 151242](https://github.com/user-attachments/assets/d954e1b7-cd66-4958-8873-83ad3c3a345a)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2024-10-07 151357](https://github.com/user-attachments/assets/498cbd2e-0ad2-4519-9855-c39ceb8b3ec4)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```
![Screenshot 2024-10-07 151448](https://github.com/user-attachments/assets/6c7af891-942c-4e7a-997c-a97e005f647d)



# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
