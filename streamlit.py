#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
#import relevant libraries
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import tarfile
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

import os

###################################
#from st_aggrid import AgGrid
#from st_aggrid.grid_options_builder import GridOptionsBuilder
#from st_aggrid.shared import JsCode

###################################

#from functionforDownloadButtons import download_button

###################################


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="📦", page_title="K-means Clustering")

c20_, c30_, c31_ = st.columns([1, 1, 6])
# st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/balloon_1f388.png", width=100)
with c30_: st.image(
    "https://community.powerbi.com/oxcrx34285/attachments/oxcrx34285/RVisuals/15/6/clustering.png",
    width=100,
)

with c31_: st.title("K-means Clustering")

#define columns visible on the page
c29, c30, c31 = st.columns([1, 6, 1])

with c30:

    uploaded_file = st.file_uploader(
        "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .xlsx")
        shows = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(shows)

    else:
        st.info(
            f"""
                👆 Upload an .xlsx file first.
                """
        )

        st.stop()

#startCheck = False

rootFolder = st.text_input("Path to Root Folder",value="/Graphs", placeholder="/Users/...").rstrip("/")
nameFolder = st.text_input("Name Cluster Files",value="Clusters", placeholder="Clusters...").rstrip("/")

st.table(shows.head(5))

regColsLooped = pd.DataFrame({'start':[],'end':[]})
colitterator = 0

c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

import random
import string
import streamlit.components.v1 as components
if 'input_keys' not in st.session_state:
    st.session_state.input_keys= []

with c1:
    nameCol = st.text_input("Write Name", value="Looping Columns")
with c2:
    repeats = st.number_input("Repeats in Each Loops", value=3)
with c3:
    loops = st.number_input("Loops to Add", value=4)
with c4:
    if st.button("Add Loops"):#, on_click=ButtonPress):
        for i in range(0,loops):
            st.session_state.input_keys.append(random.choice(string.ascii_uppercase)+str(random.randint(0,999999)))
    if st.button("Remove Last Loop"):
        st.session_state.input_keys = st.session_state.input_keys[:-repeats]
    if st.button("Remove All Loops"):
        st.session_state.input_keys = []


te1, te2 = st.columns([2, 6])

input_values = []
reg_name = []
reg_start = []
reg_end = []
reg_check = []
reg_repeats = []
#input_values = ["11","12","13"]#,"14","15","16","17","18","19"]
#reg_name = ["a","a","a","b","b","b","c","c","c"]
reg_start2 = [79,92,105,83,96,109,120,255,390,190,325,460]
reg_end2 = [82,95,108,92,105,118,190,325,460,207,325+17,460+17]
#reg_check = [False,False,False,False,False,False,False,False,False]
#reg_repeats = [1,2,3,1,2,3,1,2,3]


trueLoop = 0
for i,input_key in enumerate(st.session_state.input_keys):
    #if input_key in
    #input_value = st.text_input("Please input something", key=input_key)
    with te2:
        st.write(nameCol+"_"+str(i))
    with te2:
        globals()['check'+str(i)] = st.checkbox("Impute Categories", key="check"+input_key)
        st.write("Ensure All Loops Have An Identical Number Of Columns")
    for it in range(0,repeats):
        with te2:
            try:
                globals()['colStart'+str(i)], globals()['colEnd'+str(i)] = st.slider("Loop "+str(it),0,len(shows.columns),
                (reg_start2[trueLoop],reg_end2[trueLoop]), key=str(it)+"_"+input_key)
            except:
                globals()['colStart'+str(i)], globals()['colEnd'+str(i)] = st.slider("Loop "+str(it),0,len(shows.columns),
                    (0,5), key=str(it)+"_"+input_key)
            globals()['table'+str(i)] = st.table(shows.iloc[:3,globals()['colStart'+str(i)]:globals()['colEnd'+str(i)]])
        #if buttonTap == True:
        input_values.append(globals()['table'+str(i)])
        reg_name.append(nameCol+"_"+str(i))
        reg_start.append(globals()['colStart'+str(i)])
        reg_end.append(globals()['colEnd'+str(i)])
        reg_check.append(globals()['check'+str(i)])
        reg_repeats.append(it)
        trueLoop = trueLoop+1
    with te2:
        components.html("""<hr style="border: 1px dashed; color:lightgrey;"/>""")

colOverview = pd.DataFrame({'name':reg_name,
                            'repeat':reg_repeats,
                            'start':reg_start,
                            'end':reg_end,
                            'check':reg_check})

if len(reg_check) >9:
    st.write("Loop Overview")
    coverView = st.table(colOverview)


df = shows.copy()
#save original data copy
df_original = df.copy()
df_len = df_original.shape[0]-1
#split relevant occurances to concat later
for n,m in enumerate(colOverview['name']):
    globals()[m+"_"+str(colOverview['repeat'].iloc[n])] = df.iloc[:,colOverview['start'].iloc[n]:colOverview['end'].iloc[n]]

for n,m in enumerate(colOverview['name'].unique()):
    for j in range(1,colOverview.loc[colOverview['name']==m,'repeat'].max()+1):
        globals()[m+"_"+str(colOverview['repeat'].iloc[j])].columns = globals()[m+"_"+str(colOverview['repeat'].iloc[0])].columns
        globals()[m+"_"+str(colOverview['repeat'].iloc[0])] = globals()[m+"_"+str(colOverview['repeat'].iloc[0])].append(globals()[m+"_"+str(colOverview['repeat'].iloc[j])])
    globals()[m+"_"+str(colOverview['repeat'].iloc[0])] = globals()[m+"_"+str(colOverview['repeat'].iloc[0])].reset_index(drop=True)

df = pd.DataFrame({"tempDelCol":range(0,df_len)})
for n,m in enumerate(colOverview['name'].unique()):
    df = df.merge(globals()[m+"_"+str(colOverview['repeat'].iloc[n])], left_index=True, right_index=True )

components.html("""<hr style="border: 1px dashed; color:lightgrey;"/>""")
df = df.drop(columns=["tempDelCol"])

dropcoltxt = st.text_input("Remove Columns Starting With (Seperated By Commas)",placeholder="xQ,...")
dropLst = dropcoltxt.split(",")
dropLst = [i.strip(" ") for i in dropLst]
if (len(dropLst) >= 1) & (dropLst != [""]):
    for io in dropLst:
        df = df.drop(columns=[col for col in df if col.startswith(io)])

dropcoltxt2 = st.text_input("Remove Columns Ending With (Seperated By Commas)",placeholder="oe,...")
dropLst2 = dropcoltxt2.split(",")
dropLst2 = [i.strip(" ") for i in dropLst2]
if (len(dropLst2) >= 1) & (dropLst2 != [""]):
    for io in dropLst2:
        df = df.drop(columns=[col for col in df if col.endswith(io)])

dropcoltxt3 = st.text_input("Remove Columns By Name (Seperated By Commas)",placeholder="Q9_1,...")
dropLst3 = dropcoltxt3.split(",")
dropLst3 = [i.strip(" ") for i in dropLst3]
if (len(dropLst3) >= 1) & (dropLst3 != [""]):
    try:
        df = df.drop(columns=dropLst3)
    except:
        pass

RealView = st.table(df.head(5))


starti, endi = st.slider("Number of Clusters", 0, 100, (9, 14))
endi = endi+1

st.success(
    f"""
        💡 Upload Success!
        """
)

#import subprocess

isFile = os.path.isfile(currentFolder)
if isFile == False:
    os.mkdir(currentFolder)

def clusteringFunc():
    for ix in range(starti,endi):

        numberOfClusters = ix
        currentFolder = rootFolder+"/"+nameFolder+"_"+str(numberOfClusters)
        #currentFolder = rootFolder+nameFolder+"_"+str(numberOfClusters)

        #df = shows.copy()

        #save original data copy
        #df_original = df.copy()

        #df_len = df_original.shape[0]-1

        #split relevant occurances to concat later
#        for n,m in enumerate(colOverview['name']):
#            globals()[m+"_"+str(colOverview['repeat'].iloc[n])] = df.iloc[:,colOverview['start'].iloc[n]:colOverview['end'].iloc[n]]

#        for n,m in enumerate(colOverview['name'].unique()):
#            for j in range(1,colOverview.loc[colOverview['name']==m,'repeat'].max()+1):
#                globals()[m+"_"+str(colOverview['repeat'].iloc[j])].columns = globals()[m+"_"+str(colOverview['repeat'].iloc[0])].columns
#                globals()[m+"_"+str(colOverview['repeat'].iloc[0])] = globals()[m+"_"+str(colOverview['repeat'].iloc[0])].append(globals()[m+"_"+str(colOverview['repeat'].iloc[j])])
#            globals()[m+"_"+str(colOverview['repeat'].iloc[0])] = globals()[m+"_"+str(colOverview['repeat'].iloc[0])].reset_index()

#        df = pd.DataFrame({"tempDelCol":range(0,len(globals()[m+"_"+str(colOverview['repeat'].iloc[0])]))})
#        for n,m in enumerate(colOverview['name'].unique()):
#            df = df.merge(globals()[m+"_"+str(colOverview['repeat'].iloc[n])], left_index=True, right_index=True )

#        df = df.drop(columns=["tempDelCol"])

        #drop flagged columns
#        df = df.drop(columns=[col for col in df if col.startswith('xQ')])
#        df = df.drop(columns=[col for col in df if col.endswith('oe')])
        #df = df.iloc[:,:27]

#        df = df.drop(columns='Q9_1')

        #identify columns that need OneHotEncoding
        cols = pd.DataFrame({'col':df.dtypes.index,'type':df.dtypes.values})
        cols['type'].value_counts()

        text_columns = cols.loc[cols['type']=='object','col'].to_list()
        number_columns = cols.loc[cols['type']!='object','col'].to_list()

        #DataFrame with only the text columns
        df1 = df[text_columns]

        #OneHotEncode the text columns
        cat = OneHotEncoder()
        df2 = cat.fit_transform(df1.fillna("blanc")).toarray()

        #save OneHotEncoded Column names
        labels = cat.get_feature_names(text_columns)

        df3 = pd.DataFrame(df2)
        df3.columns = labels

        #concat to bring back in the numeric columns
        df4 = pd.concat([df[number_columns].reset_index(),df3.reset_index()],axis=1)

        #save cleaned data file
        #df4.iloc[:,1:].to_csv("dataclean_220950_cluster.csv")

        #convert to Numpy array
        data_clean = df4.iloc[:,1:].fillna(0).to_numpy()

        preprocessor = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("pca", PCA(n_components=2, random_state=42)),
            ]
        )

        #init: You’ll use "k-means++" instead of "random" to ensure centroids are initialized
        #with some distance between them. In most cases, this will be an improvement over "random".

        #n_init: You’ll increase the number of initializations to ensure you find a stable solution.

        #max_iter: You’ll increase the number of iterations per initialization to ensure that
        #k-means will converge.

        clusterer = Pipeline(
           [
               (
                   "kmeans",
                   KMeans(
                       n_clusters=numberOfClusters,
                       init="k-means++",
                       n_init=50,
                       max_iter=500,
                       random_state=42,
                   ),
               ),
           ]
        )

        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clusterer", clusterer)
            ]
        )

        pipe.fit(data_clean)

        preprocessed_data = pipe["preprocessor"].transform(data_clean)

        #predicted_labels = pipe["clusterer"]["kmeans"].labels_

        #silhouette_score(preprocessed_data, predicted_labels)

        print(np.var(preprocessed_data, axis=0))

        #explained variance
        df_transform = StandardScaler().fit_transform(data_clean)
        pca_temp = PCA(n_components=2).fit(df_transform)
        new_df = pca_temp.transform(df_transform)
        var_exp = pca_temp.explained_variance_ratio_
        print(var_exp)

        pcadf = pd.DataFrame(
            pipe["preprocessor"].transform(data_clean),
            columns=["Reduced dimension 1", "Reduced dimension 2"],
        )

        pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_

        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(8, 8))

        scat = sns.scatterplot(
            "Reduced dimension 1",
            "Reduced dimension 2",
            s=50,
            data=pcadf,
            hue="predicted_cluster",
            palette="Set2",
        )

        scat.set_title(
            f"Clustering results\n {ix} Clusters"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        #add label
        labs = pcadf.groupby('predicted_cluster').mean().reset_index()
        for i,v in enumerate(labs['predicted_cluster']):
            plt.text(labs['Reduced dimension 1'][i], labs['Reduced dimension 2'][i],
                     labs['predicted_cluster'][i], fontsize = 20)

        plt.savefig(currentFolder+"_ClusteringResultsPCA.png")
        plt.show()

        #number of occurances in each cluster
        #pcadf['predicted_cluster'].value_counts().reset_index().sort_values(by='index')

        df_full = pd.concat([pcadf,df.reset_index()],axis=1)

        df_full.columns = df_full.columns.str.replace("_1","")

        df_full.to_excel(currentFolder+"_PredictedClusters.xlsx")

        d = df_full.copy()

        #question maps

        # Which day of the week did you drink it?
        Q8A = {"1":"Weekday",
               "2":"Weekend"}
        d['Q8A'] = d['Q8A'].astype(str).map(Q8A)

        # When during the day did you drink it?
        Q8B = {"1":"Pre-Breakfast",
               "2":"Breakfast",
               "3":"Mid-morning",
               "4":"Lunch",
               "5":"Mid-afternoon",
               "6":"Dinner",
               "7":"After dinner",
               "98":"Other"}
        d['Q8B'] = d['Q8B'].astype(str).map(Q8B)

        # More specifically, at what time did you drink it?
        #Q9 = {"1":"6am-9am",
        #      "2":"9am-12pm",
        #      "3":"12pm-3pm",
        #      "4":"3pm-6pm",
        #      "5":"6pm-9pm",
        #      "6":"9pm-12am",
        #      "7":"12am–3am",
        #      "8":"3am–6am"}
        #d['Q9'] = d['Q9'].astype(str).map(Q9)

        # Where did you drink it?
        Q10 = {"1":"At my home",
               "2":"At someone else's home",
               "3":"At work/at school/at university",
               "4":"At a restaurant/café/pub",
               "5":"On-the-go",
               "98":"Other"}
        d['Q10'] = d['Q10'].astype(str).map(Q10)

        # Q11 Who were you with, when you drank it?
        d['Q11r1'] = d['Q11r1'].astype(str).str.replace("1","Alone")
        d['Q11r2'] = d['Q11r2'].astype(str).str.replace("1","My partner")
        d['Q11r3'] = d['Q11r3'].astype(str).str.replace("1","Other family")
        d['Q11r4'] = d['Q11r4'].astype(str).str.replace("1","Friends")
        d['Q11r5'] = d['Q11r5'].astype(str).str.replace("1","Colleagues/classmates")
        d['Q11r98'] = d['Q11r98'].astype(str).str.replace("1","Other")

        # Q13 Occasion: Why did you choose to drink non-alcoholic beverages during this occasion?
        d['Q13r1'] = d['Q13r1'].astype(str).str.replace("1","Hungry")
        d['Q13r2'] = d['Q13r2'].astype(str).str.replace("1","Thirsty")
        d['Q13r3'] = d['Q13r3'].astype(str).str.replace("1","Warm up")
        d['Q13r4'] = d['Q13r4'].astype(str).str.replace("1","Cool down")
        d['Q13r5'] = d['Q13r5'].astype(str).str.replace("1","Suppress my appetite")
        d['Q13r6'] = d['Q13r6'].astype(str).str.replace("1","Energy boost")
        d['Q13r7'] = d['Q13r7'].astype(str).str.replace("1","Satisfy a craving")
        d['Q13r8'] = d['Q13r8'].astype(str).str.replace("1","Relax")
        d['Q13r9'] = d['Q13r9'].astype(str).str.replace("1","Celebrate")
        d['Q13r10'] = d['Q13r10'].astype(str).str.replace("1","Indulge/treat myself")
        d['Q13r11'] = d['Q13r11'].astype(str).str.replace("1","Stay focused/engaged")
        d['Q13r12'] = d['Q13r12'].astype(str).str.replace("1","My routine")
        d['Q13r13'] = d['Q13r13'].astype(str).str.replace("1","Feel comforted")
        d['Q13r14'] = d['Q13r14'].astype(str).str.replace("1","Cope with stress")
        d['Q13r15'] = d['Q13r15'].astype(str).str.replace("1","Take a break")
        d['Q13r16'] = d['Q13r16'].astype(str).str.replace("1","Connect with others")
        d['Q13r98'] = d['Q13r98'].astype(str).str.replace("1","Other")

        # Q14 Occasion: What factors were most important to you in choosing a beverage?
        d['Q14r1'] = d['Q14r1'].astype(str).str.replace("1","Tasty")
        d['Q14r2'] = d['Q14r2'].astype(str).str.replace("1","Healthy")
        d['Q14r3'] = d['Q14r3'].astype(str).str.replace("1","Hydrating")
        d['Q14r4'] = d['Q14r4'].astype(str).str.replace("1","Hot")
        d['Q14r5'] = d['Q14r5'].astype(str).str.replace("1","Cold")
        d['Q14r6'] = d['Q14r6'].astype(str).str.replace("1","Shareable")
        d['Q14r7'] = d['Q14r7'].astype(str).str.replace("1","New/different")
        d['Q14r8'] = d['Q14r8'].astype(str).str.replace("1","A familiar product/brand")
        d['Q14r9'] = d['Q14r9'].astype(str).str.replace("1","Affordable")
        d['Q14r10'] = d['Q14r10'].astype(str).str.replace("1","Easy to drink")
        d['Q14r11'] = d['Q14r11'].astype(str).str.replace("1","Easy to prepare")
        d['Q14r12'] = d['Q14r12'].astype(str).str.replace("1","Sustainably packaged")
        d['Q14r13'] = d['Q14r13'].astype(str).str.replace("1","High quality")
        d['Q14r14'] = d['Q14r14'].astype(str).str.replace("1","Able to keep me focused")
        d['Q14r15'] = d['Q14r15'].astype(str).str.replace("1","Energizing")
        d['Q14r16'] = d['Q14r16'].astype(str).str.replace("1","Relaxing")
        d['Q14r98'] = d['Q14r98'].astype(str).str.replace("1","Other")

        d = d.replace("0","")

        d['Q11'] = d[['Q11r1','Q11r2','Q11r3','Q11r4','Q11r5','Q11r98']
                    ].stack().groupby(level=0).apply(lambda x: x.unique().tolist())
        q11 = d.groupby(['predicted_cluster'])['Q11'].sum()
        q11 =pd.DataFrame(q11)

        d['Q13'] = d[['Q13r1','Q13r2','Q13r3','Q13r4','Q13r5','Q13r6','Q13r7','Q13r8','Q13r9',
                      'Q13r10','Q13r11','Q13r12','Q13r13','Q13r14','Q13r15','Q13r16','Q13r98']
                    ].stack().groupby(level=0).apply(lambda x: x.unique().tolist())
        q13 = d.groupby(['predicted_cluster'])['Q13'].sum()
        q13 =pd.DataFrame(q13)

        d['Q14'] = d[['Q14r1','Q14r2','Q14r3','Q14r4','Q14r5','Q14r6','Q14r7','Q14r8','Q14r9','Q14r10','Q14r11','Q14r12','Q14r13','Q14r14','Q14r15','Q14r16','Q14r98']
                    ].stack().groupby(level=0).apply(lambda x: x.unique().tolist())
        q14 = d.groupby(['predicted_cluster'])['Q14'].sum()
        q14 =pd.DataFrame(q14)

        q8a = pd.DataFrame(d.groupby(['predicted_cluster'])['Q8A'].value_counts()
                          ).rename(columns={'Q8A':'Q8A_value_counts'}).reset_index(
                          ).rename(columns={'Q8A':'Q8A_WHEN'})
        q8b = pd.DataFrame(d.groupby(['predicted_cluster'])['Q8B'].value_counts()
                          ).rename(columns={'Q8B':'Q8B_value_counts'}).reset_index(
                          ).rename(columns={'Q8B':'Q8B_WHEN'})
        #q9 =  pd.DataFrame(d.groupby(['predicted_cluster'])['Q9'].value_counts()
        #                  ).rename(columns={'Q9':'Q9_value_counts'}).reset_index(
        #                  ).rename(columns={'Q9':'Q9_WHAT TIME'})
        q10 = pd.DataFrame(d.groupby(['predicted_cluster'])['Q10'].value_counts()
                          ).rename(columns={'Q10':'Q10_value_counts'}).reset_index(
                          ).rename(columns={'Q10':'Q10_WHERE'})
        #q11 = pd.DataFrame(d.groupby(['predicted_cluster'])['Q11'].value_counts()
        #                  ).rename(columns={'Q11':'Q11_value_counts'}).reset_index(
        #                  ).rename(columns={'Q11':'Q11_WHO'})

        #Join Q11 response options
        q11_clusterCount = pd.DataFrame({"predicted_cluster":[],"Q11_value_counts":[]})
        for i,v in enumerate(q11['Q11']):
            temp = pd.DataFrame(v).replace("",np.nan).value_counts()
            te = pd.DataFrame({'Q11_value_counts':temp})
            te['predicted_cluster'] = i
            q11_clusterCount = pd.concat([q11_clusterCount,te],axis=0)

        q11_clusterCount = q11_clusterCount.reset_index()

        q11_clusterCount = q11_clusterCount.rename(columns={'index':'Q11_WHO'})

        q11_clusterCount['Q11_WHO'] = q11_clusterCount['Q11_WHO'].astype(str).str.replace(",",""
                                          ).str.replace("\(",""
                                          ).str.replace("\)",""
                                          ).str.replace("'","")

        q11_clusterCount[['predicted_cluster','Q11_value_counts']
                     ] = q11_clusterCount[['predicted_cluster','Q11_value_counts']].astype(int)

        q11 = q11_clusterCount.copy()

        #Join Q13 response options
        clusterCounts = pd.DataFrame({"predicted_cluster":[],"Q13_value_counts":[]})
        for i,v in enumerate(q13['Q13']):
            temp = pd.DataFrame(v).replace("",np.nan).value_counts()
            te = pd.DataFrame({'Q13_value_counts':temp})
            te['predicted_cluster'] = i
            clusterCounts = pd.concat([clusterCounts,te],axis=0)

        clusterCounts = clusterCounts.reset_index()

        clusterCounts = clusterCounts.rename(columns={'index':'Q13_WHY'})

        clusterCounts['Q13_WHY'] = clusterCounts['Q13_WHY'].astype(str).str.replace(",",""
                                          ).str.replace("\(",""
                                          ).str.replace("\)",""
                                          ).str.replace("'","")

        clusterCounts[['predicted_cluster','Q13_value_counts']
                     ] = clusterCounts[['predicted_cluster','Q13_value_counts']].astype(int)

        #Join Q14 response options
        q14_clusterCount = pd.DataFrame({"predicted_cluster":[],"Q14_value_counts":[]})
        for i,v in enumerate(q14['Q14']):
            temp = pd.DataFrame(v).replace("",np.nan).value_counts()
            te = pd.DataFrame({'Q14_value_counts':temp})
            te['predicted_cluster'] = i
            q14_clusterCount = pd.concat([q14_clusterCount,te],axis=0)

        q14_clusterCount = q14_clusterCount.reset_index()

        q14_clusterCount = q14_clusterCount.rename(columns={'index':'Q14_WHAT'})

        q14_clusterCount['Q14_WHAT'] = q14_clusterCount['Q14_WHAT'].astype(str).str.replace(",",""
                                          ).str.replace("\(",""
                                          ).str.replace("\)",""
                                          ).str.replace("'","")

        q14_clusterCount[['predicted_cluster','Q14_value_counts']
                     ] = q14_clusterCount[['predicted_cluster','Q14_value_counts']].astype(int)

        q14 = q14_clusterCount.copy()

        #calculate total percentage distributions
        q8atotal = q8a.groupby('Q8A_WHEN').sum()
        q8atotal['Q8A_total%'] = q8atotal['Q8A_value_counts']/q8atotal['Q8A_value_counts'].sum()

        q8btotal = q8b.groupby('Q8B_WHEN').sum()
        q8btotal['Q8B_total%'] = q8btotal['Q8B_value_counts']/q8btotal['Q8B_value_counts'].sum()

        #q9total = q9.groupby('Q9_WHAT TIME').sum()
        #q9total['Q9_total%'] = q9total['Q9_value_counts']/q9total['Q9_value_counts'].sum()

        q10total = q10.groupby('Q10_WHERE').sum()
        q10total['Q10_total%'] = q10total['Q10_value_counts']/q10total['Q10_value_counts'].sum()

        q11total = q11.groupby('Q11_WHO').sum()
        q11total['Q11_total'] = d['index'].count()
        q11total['Q11_total%'] = q11total['Q11_value_counts']/d['index'].count()

        clusterCounts_total = clusterCounts.groupby('Q13_WHY').sum()
        clusterCounts_total['Q13_total'] = d['index'].count()
        clusterCounts_total['Q13_total%'] = clusterCounts_total['Q13_value_counts']/d['index'].count()

        q14total = q14.groupby('Q14_WHAT').sum()
        q14total['Q14_total'] = d['index'].count()
        q14total['Q14_total%'] = q14total['Q14_value_counts']/d['index'].count()

        #calculate relative percentages
        q8a_temp = q8a.groupby('predicted_cluster')['Q8A_value_counts'].sum().rename('Q8A_total')
        q8a = q8a.merge(q8a_temp.reset_index(),on='predicted_cluster',how='left')
        q8a['Q8A_%'] = q8a['Q8A_value_counts']/q8a['Q8A_total']

        q8b_temp = q8b.groupby('predicted_cluster')['Q8B_value_counts'].sum().rename('Q8B_total')
        q8b = q8b.merge(q8b_temp.reset_index(),on='predicted_cluster',how='left')
        q8b['Q8B_%'] = q8b['Q8B_value_counts']/q8b['Q8B_total']

        #q9_temp = q9.groupby('predicted_cluster')['Q9_value_counts'].sum().rename('Q9_total')
        #q9 = q9.merge(q9_temp.reset_index(),on='predicted_cluster',how='left')
        #q9['Q9_%'] = q9['Q9_value_counts']/q9['Q9_total']

        q10_temp = q10.groupby('predicted_cluster')['Q10_value_counts'].sum().rename('Q10_total')
        q10 = q10.merge(q10_temp.reset_index(),on='predicted_cluster',how='left')
        q10['Q10_%'] = q10['Q10_value_counts']/q10['Q10_total']

        #q11_temp = q11.groupby('predicted_cluster')['Q11_value_counts'].sum().rename('Q11_total')
        q11_temp = d.groupby('predicted_cluster')['index'].count().rename('Q11_total').reset_index()
        q11 = q11.merge(q11_temp.reset_index(),on='predicted_cluster',how='left')
        q11['Q11_%'] = q11['Q11_value_counts']/q11['Q11_total']

        #clusterCounts_temp = clusterCounts.groupby('predicted_cluster')['Q13_value_counts'].sum().rename('Q13_total')
        clusterCounts_temp = d.groupby('predicted_cluster')['index'].count().rename('Q13_total').reset_index()
        clusterCounts = clusterCounts.merge(clusterCounts_temp.reset_index(),on='predicted_cluster',how='left')
        clusterCounts['Q13_%'] = clusterCounts['Q13_value_counts']/clusterCounts['Q13_total']

        q14_temp = d.groupby('predicted_cluster')['index'].count().rename('Q14_total').reset_index()
        q14 = q14.merge(q14_temp.reset_index(),on='predicted_cluster',how='left')
        q14['Q14_%'] = q14['Q14_value_counts']/q14['Q14_total']

        d2 = d.merge(df_original[['uuid']].reset_index(),how='left',on='index')

        d3 = d2[['index','predicted_cluster','uuid']]
        d3 = d3.reset_index()
        d3.loc[d3['level_0']<=df_len,'Occurrence'] = 1
        d3.loc[(d3['level_0']>df_len)&(d3['level_0']<=(df_len*2)+1),'Occurrence'] = 2
        d3.loc[(d3['level_0']>(df_len*2)+1),'Occurrence'] = 3

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(currentFolder+'_Summary.xlsx', engine='xlsxwriter')
        tots = pd.concat([q8atotal.drop(columns=['predicted_cluster']).reset_index(),
                   q8btotal.drop(columns=['predicted_cluster']).reset_index(),
                   #q9total.drop(columns=['predicted_cluster']).reset_index(),
                   q10total.drop(columns=['predicted_cluster']).reset_index(),
                   q11total.drop(columns=['predicted_cluster']).reset_index(),
                   clusterCounts_total.drop(columns=['predicted_cluster']).reset_index(),
                   q14total.drop(columns=['predicted_cluster']).reset_index()
                             ],axis=1)
        tots.to_excel(writer, sheet_name='Total percentages')
        for i in q8a['predicted_cluster'].unique():
            temp = pd.concat([q8a[q8a['predicted_cluster']==i].drop(columns=['predicted_cluster']).reset_index(drop=True),
                              q8b[q8b['predicted_cluster']==i].drop(columns=['predicted_cluster']).reset_index(drop=True),
                              #q9[q9['predicted_cluster']==i].drop(columns=['predicted_cluster']).reset_index(drop=True),
                              q10[q10['predicted_cluster']==i].drop(columns=['predicted_cluster']).reset_index(drop=True),
                              q11[q11['predicted_cluster']==i].drop(columns=['predicted_cluster']).reset_index(drop=True),
                              clusterCounts[clusterCounts['predicted_cluster']==i].drop(columns=['predicted_cluster']).reset_index(drop=True),
                              q14[q14['predicted_cluster']==i].drop(columns=['predicted_cluster']).reset_index()
                             ],axis=1)

            temp.to_excel(writer, sheet_name='Cluster '+str(i))
        d3.drop(columns=['level_0']).to_excel(writer, sheet_name='ID&Cluster Overview')
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
    return st.success("💡 Clustering Complete!")


st.button(label="Run Clustering",on_click=clusteringFunc)

import shutil
myfile = shutil.make_archive("myfile", 'zip', "/Graphs")

with open("myfile.zip", "rb") as fp:
    btn = st.download_button(
        label="Download ZIP",
        data=fp,
        file_name="myfile.zip",
        mime="application/zip"
    )

###

#st.subheader("Filtered data will appear below 👇 ")

#st.text("")

#c29, c30, c31 = st.columns([1, 1, 2])

#with c29:

#    CSVButton = download_button(
#        df,
#        "File.csv",
#        "Download to CSV",
#    )

#with c30:
#    CSVButton = download_button(
#        df,
#        "File.csv",
#        "Download to TXT",
#    )


# In[ ]:
