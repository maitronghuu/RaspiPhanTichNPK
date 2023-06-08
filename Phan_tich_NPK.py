# Thu vien ho tro
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Buoc 1. Thu thap du lieu
data = pd.read_csv('data.csv')
print("Number of Datapoints : ",data.shape[0])
data.head()
# Buoc 2. Xu ly du lieu
x = data.iloc[:,1:-1].values
print("x:")
print(x)
y = data.iloc[:,-1].values
print("y:")
print(y)
le = LabelEncoder()
y = le.fit_transform(y)
print("y1:")
print(y)

# Buoc 3. Xay dung model
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#sc = StandardScaler()
#x_train[:,0:3]=sc.fit_transform(x_train[:,0:3])
#print(x_train)
#x_test[:,0:3]=sc.fit_transform(x_test[:,0:3])
#print(x_test)

model = DecisionTreeClassifier()
mymodel = model.fit(x_train,y_train)

# Buoc 4. Test Du doan ket qua
new_data= np.array([[32.25,78,65]])
print(mymodel.predict(new_data))

# Buoc 5. Danh gia ket qua
while True:
    data = []
    new_data = []
    ni_to = float(input("ni_to: "))
    data.append(ni_to)
    phot_pho = float(input("phot_pho: "))
    data.append(phot_pho)
    ka_li = float(input("ka_li : "))
    data.append(ka_li )
    new_data.append(data)
    if (mymodel.predict(new_data[0:])== 0):
       print("Dat xau")

    else:
        print(mymodel.predict(new_data[0:]))
        print("Dat tot")





