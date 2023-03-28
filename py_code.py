import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import subprocess
import argparse

df = pd.read_csv("C:\\Users\\s123\\Desktop\\iris_dvc\\Iris.csv")

flower_mapping = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2 }
df["Species"] = df["Species"].map(flower_mapping)

X = df.drop('Species', axis = 1)
y = df['Species']

parser = argparse.ArgumentParser()
parser.add_argument('--ccp_alpha', type=float, default=0.0, help='complexity parameter for minimal cost-complexity pruning')
parser.add_argument('--min_sample_split', type=int, default=2, help='min number of sample splits')
parser.add_argument('--criterion', type=str, default='gini', help='gini or entropy')
parser.add_argument('--train_test_size',nargs= '*',type=float,default=[], help='Enter the split of train and test dataset')
args = parser.parse_args()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=args.train_test_size[0], random_state=9)

model_rf = RandomForestClassifier(ccp_alpha=args.ccp_alpha,min_samples_split=args.min_sample_split, criterion=args.criterion)

model_rf.fit(X_train,y_train)

filename = 'C:\\Users\\s123\\Desktop\\test\\models\\'+ str(args.train_test_size[0]) +'_'+ str(args.ccp_alpha)  + '_model_rf'
pickle.dump(model_rf,open(filename,'wb'))

cmd = "git rev-list --count HEAD"
try:
    output = subprocess.check_output(cmd.split()).decode().strip()
    git_count = int(output)
except:
    git_count = 0
accuracy = model_rf.score(X_test,y_test)

params = model_rf.get_params()
df_rf = pd.DataFrame(params, index = [0])
df_rf = df_rf.iloc[:,[1,3,8,10]]
df_rf["Accuracy"]  = accuracy
df_rf["Number of rows"] = len(df)
df_rf.insert(0,'model_name','Random Forest_'+str(git_count))
df_rf = df_rf.reindex(columns = ['model_name','Number of rows','ccp_alpha', 'criterion', 'min_impurity_decrease','min_samples_split', 'Accuracy'])
df_rf.to_csv("C:\\Users\\s123\\Desktop\\iris_dvc\\parameters.csv",mode = 'a', index=False, header = False)