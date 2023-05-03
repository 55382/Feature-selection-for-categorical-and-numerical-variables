#!/usr/bin/env python
# coding: utf-8

# # Projet SNI Quantylix  (ML) réalisé par Hana Ben Ali

# In[2]:



## Importing All Necessary Library Required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[4]:


data = pd.read_excel('Base de données - SNI.xlsx')


# In[70]:



data.head(10)


# In[71]:



df=pd.DataFrame(data)


# In[72]:


df.columns


# In[73]:


df=df.drop(columns=['numtiers', 'Annee'],axis=1)


# In[74]:





data.shape


# In[75]:


#Print the summary statistics of the churn DataFrame
df.describe()


# In[76]:


df.info()


# In[77]:


df.isna().sum()


# In[78]:


df.columns[df.isnull().any()]


# In[79]:


df.shape[0]


# In[80]:


print(df.dtypes)


# After checking our dataset we get to know that there are NA values which are explicitly called 'Modalite vide' for some of the non numerical and date variables .I'll remove these rows as their propotion in the dataset is not higher 65 /1521 unless it will affect the acuuracy of the modelling process.

# In[81]:


# an example for a column that contains values 'modalite variable '
df['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT'].unique()


# In[82]:


# we use drop to remove rows which contain "Modalite vide" for this column
df.drop(df.loc[df['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT']=='Modalite vide'].index, inplace=True)


# In[83]:


#  a little check after removing "Modalite vide" for this column
df['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT'].unique()


# In[84]:


df.shape


# In[85]:


# a for loop for all variables with dtype = object  to remove all rows in dataset that   contains "Modalite vide"
l=['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT', 'DIVERSITE_CLIENTS',
       'DIVERSITE_FOURNISSEURS', 'IMPACT_SOCIAUX_ENVIRONNEMENTAL',
       'NIVEAU_COMPETITIVITE', 'QUALITE_INFORMATION_FINANCIERE', 'REPUTATION',
       'STRUCTUREDUMANAGEMENT', 'SUPPORT', 'POSITIONNEMENTMARCHE',
       'Categorie_juridique', 
       'Secteurs']
for i in l:
    df.drop(df.loc[df[i]=='Modalite vide'].index, inplace=True)
# shape of the data after removing these rows
df.shape


# # Bivariante Analysis

# In[86]:


import scipy.stats as stats
crosstab = pd.crosstab(df['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT'], df['DIVERSITE_CLIENTS'])
crosstab


# In[87]:


get_ipython().system('pip install researchpy')


# In[ ]:





# In[88]:


import researchpy as rp
crosstab, test_results, expected = rp.crosstab(df['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT'], df['QUALITE_INFORMATION_FINANCIERE'],
                                               test= "chi-square",
                                               expected_freqs= True,
                                             prop= "cell")
p_value =test_results['results'][1]

    
#a measure of the strength of the relationship - this is akin to a correlation statistic
Vcramer=test_results['results'][2]
test_results


# In[89]:


import researchpy as rp
crosstab, test_results, expected = rp.crosstab(df['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT'], df['defaut'],
                                               test= "chi-square",
                                               expected_freqs= True,
                                             prop= "cell")
p_value =test_results['results'][1]

    
#a measure of the strength of the relationship - this is akin to a correlation statistic
Vcramer=test_results['results'][2]
test_results


# In[90]:




crosstab, test_results, expected = rp.crosstab(df['QUALITE_INFORMATION_FINANCIERE'], df['defaut'],
                                               test= "chi-square",
                                               expected_freqs= True,
                                             prop= "cell")
p_value =test_results['results'][1]

    
#a measure of the strength of the relationship - this is akin to a correlation statistic
Vcramer=test_results['results'][2]
test_results


# In[91]:




import researchpy as rp
crosstab, test_results, expected = rp.crosstab(df['SUPPORT'], df['POSITIONNEMENTMARCHE'],
                                               test= "chi-square",
                                               expected_freqs= True,
                                              prop= "cell")
p_value =test_results['results'][1]

    
#a measure of the strength of the relationship - this is akin to a correlation statistic
Vcramer=test_results['results'][2]
test_results 


# In[92]:


crosstab, test_results, expected = rp.crosstab(df["DIVERSITE_FOURNISSEURS"], df["DIVERSITE_CLIENTS"],
                                               test= "chi-square",
                                               expected_freqs= True,

                                                      prop= "cell")
p_value =test_results['results'][1]
test_results 


# In[93]:



crosstab, test_results, expected = rp.crosstab(df["DIVERSITE_FOURNISSEURS"], df["IMPACT_SOCIAUX_ENVIRONNEMENTAL"],
                                               test= "chi-square",
                                               expected_freqs= True,

                                                      prop= "cell")
p_value =test_results['results'][1]
test_results 


# ## feature selection on categorical data for a classification predictive modeling

# ### 

# In[94]:


#Cleaning

a = "Diversification tres forte par produits, clients, situation geographique"
b = "Diversification très forte par produits,clients, situation geographique"
c = 'Diversification très forte par produits, clients, situation geographique'
df = df.replace([b,c], a)
df = df.replace("Très grande diversite", "Tres grande diversite")
df = df.replace(["Aucun impact social ou environnemental, soumis e une reglementation","Aucun impact"], "Aucun impact social ou environnemental, soumis à une reglementation")
df = df.replace('Tres bonne', "Très bonne")
df=df.replace('Tres forte concurrence','Très forte concurrence')


# ### Encoding categorical variables

# In[95]:


ordinal_cols_mapping = [{
    "col": "EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT",
    "mapping": {
        'Modalite vide': 0,
        'Inferieure e 5 ans': 1,
        'Entre 5 et 10 ans': 2,
        'Plus de 10 ans': 3,
    }}, {
    "col": "DIVERSITE_CLIENTS",
    "mapping": {
        'Modalite vide': 0,
        'Diversification limitee e un seul client ou un seul produit ou e une seule zone geographique': 1,
        'Forte dependance e quelques clients mais limitee e un seul produit': 2,
        'Bonne diversification par produits mais limitee e une zone geographique ou e quelques client': 3,
        'Diversification tres forte par produits, clients, situation geographique': 4,
    }}, {        
    "col": "DIVERSITE_FOURNISSEURS",
    "mapping": {
        'Modalite vide': 0,
        'Pas de diversite': 1,
        'Diversite insufisante': 2,
        'Diversite moyenne': 3,
        'Tres grande diversite': 4,
    }}, {
    "col": "IMPACT_SOCIAUX_ENVIRONNEMENTAL",
    "mapping": {
        'Aucun impact social ou environnemental, soumis à une reglementation': 0,
        'Impact social ou environnemental potentiel mais reversible': 1,
        'Impact social ou environnemental marginal': 2,
        'Fort impact social ou environnemental irreversible': 3,
    }}, {
    "col": "NIVEAU_COMPETITIVITE",
    "mapping": {
        'Très forte concurrence': 0,
        'Forte presence de produits similaires et competitifs': 1,
        'Quelques competiteurs majeurs identifies': 2,
        'Absence de concurrence (quasi monopole)': 3,
    }}, {
    "col": "QUALITE_INFORMATION_FINANCIERE",
    "mapping": {
        'Etats comptables et financiers peu fiables ou innexistants': 0,
        'Etats comptables et financiers sommaires': 1,
        'Etats comptables et financiers coherents mais non audites': 2,
        'Etats comptables et financiers audites par un cabinet reconnu': 3,
    }}, {
    "col": "REPUTATION",
    "mapping": {
        'Mauvaise': 0,
        'Moyenne': 1,
        'Bonne': 2,
        'Très bonne': 3,
    }}, {
    "col": "STRUCTUREDUMANAGEMENT",
    "mapping": {
        'Pas structure': 0,
        'Peu structure': 1,
        'Moyennement structure': 2,
        'Bien structure': 3,
    }}, {
    "col": "POSITIONNEMENTMARCHE",
    "mapping": {
        'Acteur non significatif': 0,
        'Acteur marginal': 1,
        'Acteur majeur + de 20% de part de marche': 2,
        'Leader': 3,
    }},{
    "col":"Categorie_juridique",
    "mapping":{
        'SA':0,
        'SARL':1,
        'Autres forme juridique':2,

        
        
        
    }},{
    "col":"SUPPORT",
    "mapping":{
        'Absence de support des actionnaires':0,
        'Lettre de confort de la maison mere':1,
        'Support conforme aux attentes':2,
        'Support occasionnel ou insuffisant':3




    }}
    
      

]


# ###### 

# In[96]:


cat_col=["EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT","DIVERSITE_CLIENTS","DIVERSITE_FOURNISSEURS","IMPACT_SOCIAUX_ENVIRONNEMENTAL","NIVEAU_COMPETITIVITE", "QUALITE_INFORMATION_FINANCIERE","REPUTATION","STRUCTUREDUMANAGEMENT","POSITIONNEMENTMARCHE","Categorie_juridique","SUPPORT", "Cote en bourse","Appartenance a un groupe"]
df2=df[cat_col]
cat_col


# In[97]:


get_ipython().system('pip install category_encoders')


# In[98]:



import category_encoders as ce
encoder =ce.OrdinalEncoder (mapping =ordinal_cols_mapping,return_df=True)



df_train =encoder.fit_transform(df2)
df_train .info()#to verify that all categorical variables are encoded


# ### Chi-Squared Feature Selection

# In[99]:




from sklearn.model_selection import train_test_split
# split into input (X) and output (y) variables
X = df[cat_col]
y = np.array(df['defaut']).reshape(-1,1)
 # format all fields as string
X = X.astype(str)

 

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)


# In[100]:


Y=df['defaut']


# In[101]:


from sklearn.feature_selection import SelectKBest, chi2
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(df_train,Y)
X_train_fs = fs.transform(df_train)
X_test_fs = fs.transform(X_test)


# In[102]:


# what are scores for the features
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# This clearly shows that feature 3 might be the most relevant (according to chi-squared) and that perhaps two of the twelve input features are the most relevant.Which are :
#                    
#  3   IMPACT_SOCIAUX_ENVIRONNEMENTAL           
#  4   NIVEAU_COMPETITIVITE

# In[103]:


cols_to_drop =['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT','DIVERSITE_CLIENTS','DIVERSITE_FOURNISSEURS','QUALITE_INFORMATION_FINANCIERE','REPUTATION','STRUCTUREDUMANAGEMENT','POSITIONNEMENTMARCHE','Categorie_juridique','SUPPORT', 'Cote en bourse', 'Appartenance a un groupe']


# In[104]:


df=df.drop(columns=['EXPERIENCE_MANAGEMENT_MOYENNE_DIRIGEANT','DIVERSITE_CLIENTS','DIVERSITE_FOURNISSEURS','QUALITE_INFORMATION_FINANCIERE','REPUTATION','STRUCTUREDUMANAGEMENT','POSITIONNEMENTMARCHE','Categorie_juridique','SUPPORT','Cote en bourse', 'Appartenance a un groupe'],axis=1)


# In[105]:


df.columns


# In[106]:


df=df.drop(columns='NUMTIERS_ANNEE',index=1)


# # Univariante Analysis

# In[107]:


## Extracting all Columns from Lists Comprihensions
L=[fea for fea in df.columns]


# In[108]:


target = df['defaut']


# In[109]:



df.defaut.value_counts(normalize= True).plot(kind="bar")
plt.title("Value counts of the target variable defaut")
plt.xlabel("defaut 1  Non defaut 0")
plt.xticks(rotation=0)
plt.ylabel("pourcentage d'entreprises ")
plt.show()
df.defaut.value_counts(normalize= True).plot(kind="pie")
plt.title("Value counts of the target variable defaut")
plt.xlabel("defaut 1  Non defaut 0")
plt.xticks(rotation=0)
plt.ylabel("pourcentage d'entreprises ")
plt.show()
# percentage of the passengers that survive
df['defaut'].map({0: '', 1: 'defaut'}).value_counts().plot.pie(title='defaut rate',
                                                                autopct='%1.1f%%',
                                                                explode=(0, 0.1),
                                                                shadow=True)


# In this dataset we have the number of companies that didn't have default (0) is much higher than the companies that did default( 1)

# ## numeric variables 

# In[110]:


## Extracting numerical Column by Lists comprihensions
num_col = [fea for fea in  df.columns if df[fea].dtypes !='O' if df[fea].dtypes !='<M8[ns]']
df_numeric = df[num_col]
df_numeric
#remove these variables which are not really to the dataset and remove categorica variables as cote en bouse
#descriptive statistiscs for numerical variables
df_numeric.describe()
df_numeric.columns
df_numeric=df_numeric.drop(['defaut' ,'CHIFFRE_AFFAIRES'],axis=1 )
df_numeric.columns


# In[111]:


## Extracting Categorical Column by Lists comprihensions
cat_col = [fea for fea in df.columns if df[fea].dtypes == 'O']
df_ntnum=df[cat_col]
df_ntnum.describe()
df_ntnum.columns


# In[112]:


## Extracting datetime Column by Lists comprihensions
dat_col = [fea for fea in df.columns if data[fea].dtypes =='<M8[ns]' ]
df_dat=df[dat_col]
df_dat.columns


# ### categorical variables 

# In[113]:


fig, axes = plt.subplots(1, 1, figsize=(12,10))
sns.violinplot(x ='NIVEAU_COMPETITIVITE', y ='defaut', data = df)
plt.show()


# In[114]:


sns.stripplot(x ='Secteurs', y ='defaut', data = df,
              jitter = True, hue ='defaut', dodge = True)


# In[115]:


sns.swarmplot(x ='Secteurs', y ='defaut', data = df)


# # Multivariante Analysis

# In[116]:


## Plotting Heatmap with respect to Correlation between Variable.

sns.heatmap(df.corr(),  annot = True, cmap = 'icefire', linewidths = 0.3)
fig = plt.gcf()
fig.set_size_inches(30,10)
plt.title("Corr between variable", color = 'white', size =10)
plt.show


# In[117]:


#Correlation with output variable >
cor = df.corr()
cor_target = abs(cor["defaut"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>=0.02]
relevant_features


# # Feature SELECTION by correlation for numeric variables

# ## heatmap interpretation 
# When having a look at the second row i.e default, we see how the default is correlated with other features, TRESORIE_NETTE  is the highly correlated with default followed by RESULTAT_NET,AUTO_FINANCEMENT, RESULTAT_EXPLOITATION,CHIFFRE_AFFAIRES ,cote en bourse and
# the others variables seems to be least correlated with default except the two Variables Annee and numtiers that are not really relevant  for this dataset problem ; Hence we will drop them.

# In[118]:


#X= df[['TRESORIE_NETTE','CHIFFRE_AFFAIRES','Cote en bourse','AUTO_FINANCEMENT','EXCEDENT_BRUT_EXPLOITATION','RESULTAT_EXPlOITATION','RESULTAT_NET','FINANCEMENT_PERMANENT','FONDS_DE_ROULEMENT','BESOIN_FONDS_ROULEMENT','CAPITAUX_PROPRES','TRESORIE_NETTE','TOTAL_BILAN','DETTE_FINANCIERE','ACTIF_CIRCULANT','PASSIF_CIRCULANT','TOTAL_ACTIF','TOTAL_PASSIF','DELAI_REGLEMENT_CLIENTS','DELAI_REGLEMENT_FOURNISSEURS','AUTO_FINANCEMENT','FRAIS_FINANCIERS','STOCK']]
X = df[['TRESORIE_NETTE','RESULTAT_NET','AUTO_FINANCEMENT','RESULTAT_EXPlOITATION','CHIFFRE_AFFAIRES']] #independent columns

y = df['defaut' ]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# If these variables are correlated with each other, then we need to keep only one of them and drop the rest. So let us check the correlation of selected features with each other. This can be done either by visually checking it from the above correlation matrix or from the code snippet below.

# In[119]:


print(df[['AUTO_FINANCEMENT','RESULTAT_EXPlOITATION']].corr())
# TRESORIE NETTE & the others
print(df[['TRESORIE_NETTE','RESULTAT_NET']].corr())
print(df[['TRESORIE_NETTE','AUTO_FINANCEMENT']].corr())
print(df[['TRESORIE_NETTE','RESULTAT_EXPlOITATION']].corr())
print(df[['TRESORIE_NETTE','CHIFFRE_AFFAIRES']].corr())


# So , TRESORIE_NETTE and RESULTAT_NET  are correlated 
# And TRESORIE_NETTE and RESULTAT_EXPlOITATION are correlated 

# In[120]:


print(df[['TRESORIE_NETTE','defaut']].corr())
print(df[['RESULTAT_NET','defaut']].corr())
print(df[['RESULTAT_EXPlOITATION' ,'defaut']].corr())


# I'll keep the variable TRESORIE_NETTE because it's more correlated with the target i'll drop the two variables RESULTAT_NET And RESULTAT_EXPLOITATION 
# so we get these Variables TRESORIE_NETTE ,CHIFFRE_AFFAIRES,AUTO_FINANCEMENT,cote en bourse and the target 
# 

# Let's check the independances between them

# In[121]:


print(df[['AUTO_FINANCEMENT','CHIFFRE_AFFAIRES']].corr())


# SO  it's logic that the two variables AUTO_FINANCEMENT,CHIFFRE_AFFAIRES are highly correlated so we will keep the most correlated with target variable defaut which is :AUTO_FINANCEMENT.
# 
# 
# ==> the others variables which are TRESORIE_NETTE, AUTO_FINANCEMENT  are independant 

#  df_new is the new data  that we will use for the modelling step contains 4 variables:
#  
#  TRESORIE_NETTE, AUTO_FINANCEMENT and Cote en bourse and the target defaut

# In[122]:


df_new =df[['TRESORIE_NETTE','AUTO_FINANCEMENT','defaut']]
df_new .head(5)
df=df.drop(columns=['CHIFFRE_AFFAIRES','EXCEDENT_BRUT_EXPLOITATION','RESULTAT_EXPlOITATION','RESULTAT_NET','FINANCEMENT_PERMANENT','FONDS_DE_ROULEMENT','BESOIN_FONDS_ROULEMENT','CAPITAUX_PROPRES','TOTAL_BILAN','DETTE_FINANCIERE','ACTIF_CIRCULANT','PASSIF_CIRCULANT','TOTAL_ACTIF','TOTAL_PASSIF','DELAI_REGLEMENT_CLIENTS','DELAI_REGLEMENT_FOURNISSEURS','FRAIS_FINANCIERS','STOCK'] ,axis=1)


# In[123]:


#Information value Anova for quanti v cremer quali chi2 quali 


# In[124]:


df.columns


# In[125]:


df=df.drop(columns=['DATE_DE_CREATION_TIERS','DATE_DE_CREATION_ENTREP'],axis=1)


# In[126]:


df.columns


# In[127]:


df=df.drop(columns=['Secteurs'] ,axis=1)


# In[128]:


ordinal_col_mapping = [{
    "col": "IMPACT_SOCIAUX_ENVIRONNEMENTAL",
    "mapping": {
        'Aucun impact social ou environnemental, soumis à une reglementation': 0,
        'Impact social ou environnemental potentiel mais reversible': 1,
        'Impact social ou environnemental marginal': 2,
        'Fort impact social ou environnemental irreversible': 3,
    }}, {
    "col": "NIVEAU_COMPETITIVITE",
    "mapping": {
        'Très forte concurrence': 0,
        'Forte presence de produits similaires et competitifs': 1,
        'Quelques competiteurs majeurs identifies': 2,
        'Absence de concurrence (quasi monopole)': 3,
    }}
    ]


# In[129]:


import category_encoders as ce
encoder =ce.OrdinalEncoder (mapping =ordinal_col_mapping,return_df=True)


df_train_2 =encoder.fit_transform(df)
df_train_2 .info()#to verify that all categorical variables are encoded


# In[145]:


df_train_2=df_train_2.drop(columns="defaut",axis=1)


# In[146]:


target=df['defaut']


# In[147]:


from sklearn.model_selection import train_test_split
#Let’s split X and y using Train test split
X_train,X_test,y_train,y_test = train_test_split(df_train_2,target,random_state=42,train_size=0.8)
#get shape of train and test data
print("train size X : ",X_train.shape)
print("train size y : ",y_train.shape)
print("test size X : ",X_test.shape)
print("test size y : ",y_test.shape)


# In[148]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# In[149]:


#import library
from sklearn.linear_model import LogisticRegression
#make instance of model with default parameters except class weight
#as we will add class weights due to class imbalance problem
lr_basemodel =LogisticRegression(class_weight={0:0.1,1:0.9})
# train model to learn relationships between input and output variables
lr_basemodel.fit(X_train,y_train)


# In[150]:


y_pred_basemodel = lr_basemodel.predict(X_test)


# In[151]:


import seaborn as sn
confusion_matrix = pd.crosstab(y_test, y_pred_basemodel, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)


# In[152]:


from sklearn.metrics import f1_score
print("f1 score for base model is : " , f1_score(y_test,y_pred_basemodel,average='weighted'))


# In[153]:


import sklearn.metrics as metrics
accuracy = metrics.accuracy_score(y_test,y_pred_basemodel)
accuracy_percentage = 100 * accuracy
print('Accuracy : ', accuracy)
print("Accuracy Percentage (%) : ", accuracy_percentage)


# In[154]:


from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score
tn, fp = confusion_matrix(y_pred_basemodel,y_test)


# In[155]:


import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score


# In[156]:


#calculate AUC of model
auc = metrics.roc_auc_score(y_test, y_pred_basemodel)


# In[157]:


#print AUC score
print(auc)


# In[158]:


from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred_basemodel)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred_basemodel)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# In[144]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




