import base64
import io
from time import sleep
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Div,AutocompleteInput,FileInput,PreText,Range1d,FactorRange,Plot, VBar,LabelSet
#from bokeh.models.widgets import Spacer
from bokeh.layouts import column, row
import pandas as pd
import requests
import json
import numpy as np

# manque incorporaison df de base avec df récupéré qui remplace df de base mais avec df de sélecgtion à rajouté en rouge dans les distributions
from feature_cleaning import nettoyage
print('debut')


# Load some sample data
df = pd.read_csv("application_test.csv")
df = df.sample(150,random_state=42).copy()
df_f = pd.read_csv('shap_glob_val_sup_0.csv')
data_f = df_f[df_f['feature_importance']>0.001]
#fig = pd.read_csv()  # faire sur model prediction ipynb et exporter dataset #feature_importance_global des variables d'entrainement
dict_credit = {0: 'remboursé',
               1: 'non remboursé'}

nettoyage(df)
print('Nombre de valeurs manquantes : ',df.isna().sum().sum())
loading_div = [0]
#loading_div[0] = Div(text="Waiting for file...", width=200, height=50)
decoded=[0]
value_input=[0]

#####################################################  Différents widgets   ###############################################################

# source = ColumnDataSource(df)
# ,align ='center'on ne met rien au départ # 100005 feature_importance = 0 pk ?

flag_upload = [0] # s'il ya eu chargement ou non
dot_flag = [0]

text = PreText(text=("Veuillez charger un ID svp"), styles={'font-size': '15pt', 'color': 'red'})
source = ColumnDataSource(data={'feature_importance': [0],
                                'colonnes': [0],
                                'names': [0]})  # graph sans rien au départ

# columns = [TableColumn(field=col, title = col) for col in df.columns]
file_input = FileInput(accept=".csv", width=400, height=50)
df_retrieved = [[],0] #df_retrieved[0]= une ligne sélectionnée par rapport à id
                                                            #df_retrieved[1] = prediction, feature importance locale, nom des variables, score

completion_list = [0]
if len(df_retrieved[0]) == 0:
    # liste des identifiants des clients
    completion_list[0] = [str(id) for id in df["SK_ID_CURR"].tolist()]

auto_complete_input = AutocompleteInput(completions=completion_list[0], description='ex 101099',
                                        placeholder="Veuillez saisir l\'id client par exemple 101099...",
                                        min_width=300,
                                        restrict=True)  # configuration liste auto suggestion seule

#####################################################  graphiques declarations   ###############################################################

# Create a ColumnDataSource object to hold the data
source_f = ColumnDataSource(dict(x=data_f['feature_importance'], y=data_f['nom_variables']))
source_fl = ColumnDataSource(dict(x=[0],y=[0]))
source_s = ColumnDataSource(dict(x=[0], y=[0]))
source_s_2 = ColumnDataSource(dict(x=[0], y=[0]))
source_b = ColumnDataSource(dict(x=[0], y=[0]))
source_b_2 = ColumnDataSource(dict(x=[0], y=[0]))
source_g = ColumnDataSource(dict(x=[0], y=[0]))
source_g_2 = ColumnDataSource(dict(x=[0], y=[0]))

TOOLTIPS = [
    ("index", "$index"),
    ("Importance de la variable", "@x"),  # v3
    ("Nom de la variable", "@y")]
TOOLTIPS_2 = [
    ("index", "$index"),
    ("variable X", "@x"),  # v3
    ("Variable Y", "@y")]

TOOLTIPS_3 = [
    ("index", "$index"),
    ("Index des valeurs", "@x"),  # v3
    ("Valeurs", "@y")]



## Add a bar # variables categorielles et variable continues 2
f= figure(x_range=(min(data_f['feature_importance'])-0.005,min(data_f['feature_importance'])+0.05),
          y_range=(data_f['nom_variables']),width=500, height=900,tooltips=TOOLTIPS)
f.hbar(y='x', right='y',left=0,height=0.4,source=source_f)

fl= figure(x_range=(-1,1),y_range=([]),width=500, height=800,tooltips=TOOLTIPS)
fl.hbar(y='x', right='y',left=0.05,height=0.4,source=source_fl)

# Add a scatterplot => int/int or int/float variables
s= figure(width=400, height=400,tooltips=TOOLTIPS_2, sizing_mode="stretch_both")
s.scatter(x='x', y='y', source=source_s)
s_dot = [s.circle(name='s_dot')]

# Add a scatterplot => int int or float variables
s_2= figure(width=400, height=400,tooltips=TOOLTIPS_2, sizing_mode="stretch_both")
s_2.scatter(x='x', y='y', source=source_s_2)
s_2_dot = [s_2.circle(name='s_2_dot')]
# Add a bar # variables categorielles et variable continues
#b= Plot(title=None, width=400, height=400,min_border=0, tooltips=TOOLTIPS_2)
b = figure(x_range=[],width=400, height=400,tooltips=TOOLTIPS_2, sizing_mode="stretch_both")
b.vbar(x='x', top='y', source=source_b, width=0.5)
b_dot = [b.circle(name='b_dot')]

## Add a bar # variables categorielles et variable continues 2
b_2 = figure(x_range=[],width=400, height=400,tooltips=TOOLTIPS_2, sizing_mode="stretch_both")
b_2.vbar(x='x', top='y', source=source_b_2, width=0.5)
b_2_dot = [b_2.circle(name='b_2_dot')]

#add graph with distribution
g=figure(width=400, height=400,tooltips=TOOLTIPS_3, sizing_mode="stretch_both")
g.scatter(x='x', y='y', source=source_g)
dot = [g.circle(name='dot')]
#add graph with distribution
g_2=figure(x_range=[],width=400, height=400,tooltips=TOOLTIPS_3, sizing_mode="stretch_both")
g_2.vbar(x='x', top='y', source=source_g_2, width=0.5)
dot_2 =[g_2.circle(name='dot_2')]

#####################################################  shap   ###############################################################
#shap valeurs globales 

#pd.read_csv('data_shap_x_test') ##### a modifier
#source_f.data = dict()

#source_f.data = dict(x=df_f['nom_variables'],y=df_f['feature_importance'])# a changer
f.hbar(y='y', right="x", left=0, height=0.4, source=source_f)
f.ygrid.grid_line_color = None
f.yaxis.axis_label_text_font_size = "1pt"
f.yaxis.major_label_text_font_size = "5pt"
f.xaxis.axis_label = "feature importance globale"
f.outline_line_color = None


fl.hbar(y='y', right="x", left=0, height=0.4, source=source_fl)
source_fl.data = dict(x=[], y=[])# a changer
fl.ygrid.grid_line_color = None
fl.yaxis.axis_label_text_font_size = "1pt"
fl.yaxis.major_label_text_font_size = "5pt"
fl.xaxis.axis_label = "feature importance locale"
fl.outline_line_color = None
#####################################################  callback widgets   ###############################################################

########################################################## Select 1 #############################################################

def update_select_1(attr, old, new):
    # Update the x and y values of the ColumnDataSource object based on the selected variable
    print('select_1')
    x_val = x_select.value
    y_val = y_select.value
    print('x_val =',x_val)
    print(df[x_val].value_counts(dropna=False))
    print('y_val =',y_val)
    print(df[y_val].value_counts(dropna=False))
    
    #selection multi index
    if ((df[x_val].nunique()<=10)&(df[x_val].nunique()!=1))& ((df[y_val].nunique()<=3) &(df[y_val].nunique()!=1)): # x10 catégories, y maximum 3 # variables num transfo catégorielles/ plusieurs categ x et y 
        # si une valeur dans id ou chargement a été sélectionnée
        print('categorique multi index')
        teest_ = df.groupby([x_val,y_val]).size() 
        source_c = ColumnDataSource(data={'x':[(str(i[0]), str(i[1])) for i in teest_.index],'y':teest_.values})
        c = figure(x_range=FactorRange(*[(str(i[0]), str(i[1])) for i in teest_.index]),height=400,tooltips=TOOLTIPS_2,width=400,
           sizing_mode="stretch_both")
        c_dot = [c.circle(name='c_dot')]
        if len(df_retrieved[0]) != 0:
            c.renderers.remove(c_dot[0])
            df_teest_ = teest_.reset_index()
            index = df_teest_[(df_teest_[x_val] == df_retrieved[0][x_val].values[0]) & (df_teest_[y_val] == df_retrieved[0][y_val].values[0])][[x_val, y_val]].values[0]
            index_str = tuple(str(el)for el in index) # sous forme multi index
            c_dot[0] = c.circle(x=[index_str], y=teest_[index[0], index[1]]/2,size=10, color='red',name='c_dot')  # red_dot
            print('ajout point rouge')
            
        c.vbar(x='x', top='y', source=source_c, width=0.5)
        c.xaxis.axis_label = x_val
        c.yaxis.axis_label = y_val
        labels = LabelSet(x='x', y='y', text='y', level='glyph',
                         x_offset=0, y_offset=0, source=source_c,
                         text_color='black',text_align='center')
        c.add_layout(labels)
        layout.children[1] = c
        print(5)
        
    elif (df[x_val].dtype != 'object') & (df[y_val].dtype != 'object'):  #
        
        if len(df_retrieved[0]) != 0 : #si une valeur dans id ou chargement a été sélectionnée
            s.renderers.remove(s_dot[0])
            s_dot[0] = s.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val],
                                size=10, color='red', name='s_dot')  # red_dot
            print('ajout point rouge')
        
        source_s.data['x'] = df[x_val]  # nouvelles colonnes chargées
        source_s.data['y'] = df[y_val]  # nouvelles colonnes chargées  # nouvelles colonnes chargées 
        s.xaxis.axis_label = x_val
        s.yaxis.axis_label = y_val
        layout.children[1]=s
        print('scatter_chosen')
    #selection vbar plot x = objet
    elif ((df[x_val].dtype == 'object') & (df[y_val].dtype == 'object')) | ((df[x_val].dtype == 'object') & ((df[y_val].dtype != 'object') & (df[y_val].nunique()<=2))):#(df[x_val].dtype == 'object') & (df[y_val].dtype == 'object'):
        print('vbar_chosen')
        # si une valeur dans id ou chargement a été sélectionnée
        if dot_flag[0] != 0:
            b.renderers.remove(b_dot[0])
            b_dot[0] = b.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val], size=10, color='red', name='b_dot')  # red_dot
        x_abs = df.groupby([x_val]).count()[y_val].index
        y_ord = df.groupby([x_val]).count()[y_val].values 
        #changement des valeurs 
        source_b.data['x']=x_abs
        source_b.data['y']=y_ord
        b.x_range.factors = x_abs
        b.y_range = FactorRange(str(y_ord))
        b.xaxis.axis_label = x_val
        b.yaxis.axis_label = y_val
        labels = LabelSet(x='x', y='y', text='y', level='glyph',
                         x_offset=0, y_offset=0, source=source_b,
                         text_color='black',text_align='center')
        b.add_layout(labels)
        layout.children[1]=b 
           
    # x est  une variable de type float ou int       
    else : 
        print('else', 'x_val type :' ,df[x_val].dtype,'\n',
                      'y_val type : ',df[y_val].dtype)
        print(dot_flag[0])

        # si une valeur dans id ou chargement a été sélectionnée
        if  dot_flag[0] != 0:
            b.renderers.remove(b_dot[0])
            b_dot[0] = b.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val],
                                size=10, color='red', name='b_dot')  # red dot
            print('ajout point rouge')
        
        is_numeric = True
        # si une valeur dans id ou chargement a été sélectionnée
        if (df[y_val].dtype)=='object' : #cnt children x et flowg own realty
            is_numeric = False
    
        x_abs = df.groupby([x_val]).mean(numeric_only=is_numeric)[y_val].index
        y_ord = df.groupby([x_val]).mean(numeric_only=is_numeric)[y_val].values.round(decimals=2).tolist()
        print(y_ord, max(y_ord))
        source_b.data['x']=x_abs
        source_b.data['y']=y_ord
        b.x_range.factors = x_abs
        start_ = 0
        if dot_flag[0]==0 :
            if df[y_val].mean()>0 : #variable négatif et positive
                end_ = max(y_ord)*1.05
            elif df[y_val].mean()<0 :
                end_ = min(y_ord)*1.05 # si valeur négative la plus petite est la valeur la plus importante 
        elif  dot_flag[0]==1 :
            if (df[y_val].mean()>0) & (max(y_ord)>df_retrieved[0][y_val].values[0]) : #variable négatif et positive
                end_ = max(y_ord)*1.05
            elif (df[y_val].mean()<0) & (min(y_ord)<df_retrieved[0][y_val].values[0]):
                end_ = min(y_ord)*1.05 # si valeur négative la plus petite est la valeur la plus importante 
            elif (df[y_val].mean() < 0) & (min(y_ord) > df_retrieved[0][y_val].values[0]):
                end_ =  df_retrieved[0][y_val].values[0]
            elif (df[y_val].mean() > 0) & (max(y_ord) < df_retrieved[0][y_val].values[0]):
                end_ =  df_retrieved[0][y_val].values[0]
                      
        b.y_range.start = start_
        b.y_range.end = end_
        b.xaxis.axis_label = x_val
        b.yaxis.axis_label = y_val
        labels = LabelSet(x='x', y='y', text='y', level='glyph',
                          x_offset=0, y_offset=0,source = source_b,
                          text_color='black',text_align='center')
        b.add_layout(labels)
        layout.children[1]=b 
#################################################### Select 2 #############################################################
def update_select_2(attr, old, new):
    # Update the x and y values of the ColumnDataSource object based on the selected variable
    print('test_avant_changement_2')
    x_val = x_select_2.value
    y_val = y_select_2.value
    print('x_val =',x_val)
    print(df[x_val])
    print('y_val =',y_val)
    print(df[y_val])
##################################################### Select 2 : graph modifications #############################################################   
    if (df[x_val].nunique()<=10) & (df[y_val].nunique()<=3): # x10 catégories, y maximum 3 # variables num transfo catégorielles/ plusieurs categ x et y 
        # si une valeur dans id ou chargement a été sélectionnée
        print('categorique multi index')
        teest_ = df.groupby([x_val,y_val]).size() 
        source_c_2 = ColumnDataSource(data={'x':[(str(i[0]), str(i[1])) for i in teest_.index],'y':teest_.values})
        c_2 = figure(x_range=FactorRange(*[(str(i[0]), str(i[1])) for i in teest_.index]),height=400,tooltips=TOOLTIPS_2,width=400,
           sizing_mode="stretch_both")
        c_2_dot = [c_2.circle(name='c_2_dot')]
        if len(df_retrieved[0]) != 0:
            c_2.renderers.remove(c_2_dot[0])
            df_teest_ = teest_.reset_index()
            index = df_teest_[(df_teest_[x_val] == df_retrieved[0][x_val].values[0]) & (df_teest_[y_val] == df_retrieved[0][y_val].values[0])][[x_val, y_val]].values[0]
            index_str = tuple(str(el)for el in index) # sous forme multi index
            c_2_dot[0] = c_2.circle(x=[index_str], y=teest_[index[0], index[1]]/2,size=10, color='red',name='c_2_dot')  # red_dot
            print('ajout point rouge')
            
        c_2.vbar(x='x', top='y', source=source_c_2, width=0.5)
        c_2.xaxis.axis_label = x_val
        c_2.yaxis.axis_label = y_val
        labels = LabelSet(x='x', y='y', text='y', level='glyph',
                         x_offset=0, y_offset=0, source=source_c_2,
                         text_color='black',text_align='center')
        c_2.add_layout(labels)
        layout.children[3] = c_2
        print(5)
        
    elif (df[x_val].dtype != 'object') & (df[y_val].dtype != 'object'):  #
        
        if len(df_retrieved[0]) != 0 : #si une valeur dans id ou chargement a été sélectionnée
            s_2.renderers.remove(s_2_dot[0])
            s_2_dot[0] = s_2.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val],
                                size=10, color='red', name='s_2_dot')  # red_dot
            print('ajout point rouge')
        
        source_s_2.data['x'] = df[x_val]  # nouvelles colonnes chargées
        source_s_2.data['y'] = df[y_val]  # nouvelles colonnes chargées  # nouvelles colonnes chargées 
        s_2.xaxis.axis_label = x_val
        s_2.yaxis.axis_label = y_val
        layout.children[3]=s_2
        print('scatter_chosen 2')
    #selection vbar plot x = objet
    elif ((df[x_val].dtype == 'object') & (df[y_val].dtype == 'object')) | ((df[x_val].dtype == 'object') & ((df[y_val].dtype != 'object') & (df[y_val].nunique()<=2))):#(df[x_val].dtype == 'object') & (df[y_val].dtype == 'object'):
        print('vbar_chosen 2')
        # si une valeur dans id ou chargement a été sélectionnée
        if dot_flag[0] != 0:
            b_2.renderers.remove(b_2_dot[0])
            b_2_dot[0] = b_2.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val], size=10, color='red', name='b_2_dot')  # red_dot
        x_abs = df.groupby([x_val]).count()[y_val].index
        y_ord = df.groupby([x_val]).count()[y_val].values 
        #changement des valeurs 
        source_b_2.data['x']=x_abs
        source_b_2.data['x']=y_ord
        b_2.x_range.factors = x_abs
        b_2.y_range = FactorRange(str(y_ord))
        b_2.xaxis.axis_label = x_val
        b_2.yaxis.axis_label = y_val
        labels = LabelSet(x='x', y='y', text='y', level='glyph',
                         x_offset=0, y_offset=0, source=source_b_2,
                         text_color='black',text_align='center')
        b_2.add_layout(labels)
        layout.children[3]=b_2 
           
    # x est  une variable de type float ou int       
    else : 
        print('else', 'x_val type :' ,df[x_val].dtype,'\n',
                      'y_val type : ',df[y_val].dtype)
        print(dot_flag[0])

        # si une valeur dans id ou chargement a été sélectionnée
        if  dot_flag[0] != 0:
            b_2.renderers.remove(b_2_dot[0])
            b_2_dot[0] = b_2.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val],
                                size=10, color='red', name='b_2_dot')  # red dot
            print('ajout point rouge')
        
        is_numeric = True
        # si une valeur dans id ou chargement a été sélectionnée
        if (df[y_val].dtype)=='object' : #cnt children x et flowg own realty
            is_numeric = False
    
        x_abs = df.groupby([x_val]).mean(numeric_only=is_numeric)[y_val].index
        y_ord = df.groupby([x_val]).mean(numeric_only=is_numeric)[y_val].values.round(decimals=2).tolist()
        print(y_ord, max(y_ord))
        source_b_2.data=dict(x=x_abs,y=y_ord)
        b_2.x_range.factors = x_abs
        start_ = 0
        if dot_flag[0]==0 :
            if df[y_val].mean()>0 : #variable négatif et positive
                end_ = max(y_ord)*1.05
            elif df[y_val].mean()<0 :
                end_ = min(y_ord)*1.05 # si valeur négative la plus petite est la valeur la plus importante 
        elif  dot_flag[0]==1 :
            if (df[y_val].mean()>0) & (max(y_ord)>df_retrieved[0][y_val].values[0]) : #variable négatif et positive
                end_ = max(y_ord)*1.05
            elif (df[y_val].mean()<0) & (min(y_ord)<df_retrieved[0][y_val].values[0]):
                end_ = min(y_ord)*1.05 # si valeur négative la plus petite est la valeur la plus importante 
            elif (df[y_val].mean() < 0) & (min(y_ord) > df_retrieved[0][y_val].values[0]):
                end_ =  df_retrieved[0][y_val].values[0]
            elif (df[y_val].mean() > 0) & (max(y_ord) < df_retrieved[0][y_val].values[0]):
                end_ =  df_retrieved[0][y_val].values[0]
                
        b_2.y_range.start = start_
        b_2.y_range.end = end_
        b_2.xaxis.axis_label = x_val
        b_2.yaxis.axis_label = y_val
        labels = LabelSet(x='x', y='y', text='y', level='glyph',
                          x_offset=0, y_offset=0,source = source_b_2,
                          text_color='black',text_align='center')
        b_2.add_layout(labels)
        layout.children[3]=b_2
#    #selection  scatterplot #que des valeurs continues
#    if (df[x_val].dtype != 'object') & (df[y_val].dtype != 'object'):  # sélection graphique
#        if  dot_flag[0] != 0:  #si une valeur dans id ou chargement a été sélectionnée
#            s_2.renderers.remove(s_2_dot[0])
#            s_2_dot[0]=s_2.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val], size=10, color='red',name='s_2_dot') # red_dot
#            
#        source_s_2.data['x'] = df[x_val]  # nouvelles colonnes chargées
#        source_s_2.data['y'] = df[y_val]  # nouvelles colonnes chargées 
#        s_2.xaxis.axis_label = x_val
#        s_2.yaxis.axis_label = y_val
#        layout.children[3]=s_2
#        print('scatter_chosen_2')
#        
#    #selection vbar plot x = objet COUNT
#    elif (df[x_val].dtype == 'object') & (df[y_val].dtype == 'object') :
#        if  dot_flag[0] != 0:  #si une valeur dans id ou chargement a été sélectionnée
#            b_2.renderers.remove(b_2_dot[0])
#            b_2_dot[0]=b_2.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val], size=10, color='red',name='b_2_dot') # red_dot
#            print('ajout point rouge')
#            
#        x_abs = df.groupby([x_val]).count()[y_val].index
#        y_ord = df.groupby([x_val]).count()[y_val].values
#        y_ord = (y_ord.round(decimals=2)).tolist()
#        print('vbar_chosen2_count', 'x', x_abs, 'y', y_ord)
#        #changement des valeurs 
#        new_data=dict()
#        new_data['x']=x_abs
#        new_data['y']=y_ord
#        source_b_2.data=new_data
#        print('vbar_data',new_data)
#        b_2.x_range.factors = x_abs
#        b_2.y_range = FactorRange(y_val)
#        b_2.xaxis.axis_label = x_val
#        b_2.yaxis.axis_label = y_val
#        labels = LabelSet(x='x', y='y', text='y', level='glyph',
#                         x_offset=0, y_offset=0, source=source_b_2,
#                         text_color='black',text_align='center')
#        b_2.add_layout(labels)
#        layout.children[3]=b_2
#    # x est  une variable de type float ou int    MEAN   
#    else : #erreur id avec name_type_suite et dsays birth
#        print('else', 'x_val type :' ,df[x_val].dtype,'\n',
#                      'y_val type : ',df[y_val].dtype)
#        print(dot_flag[0])
#        if dot_flag[0] != 0: #si une valeur dans id ou chargement a été sélectionnée
#            b_2.renderers.remove(b_2_dot[0])
#            b_2_dot[0]= b_2.circle(x=df_retrieved[0][x_val], y=df_retrieved[0][y_val], size=10, color='red',name='b_2_dot') # red_dot
#            print("ajout point rouge")
#        is_numeric = True # par défaut y est un nombre
#        # si une valeur dans id ou chargement a été sélectionnée
#        if (df[y_val].dtype)=='object' : #cnt children x et flowg own realty
#            is_numeric = False
#        
#        x_abs = df.groupby([x_val]).mean(numeric_only=is_numeric)[y_val].index
#        y_ord = df.groupby([x_val]).mean(numeric_only=is_numeric)[y_val].values
#        y_ord = (y_ord.round(decimals=2)).tolist()
#        print('vbar_chosen2_mean', 'x', x_abs, 'y', y_ord)
#           
#        #changement des valeurs 
#        new_data=dict()
#        new_data['x']=x_abs
#        new_data['y']=y_ord
#        source_b_2.data=new_data
#        print('vbar_data',new_data)
#        b_2.x_range.factors = x_abs
#        b_2.y_range = FactorRange(y_val)
#        b_2.xaxis.axis_label = x_val
#        b_2.yaxis.axis_label = y_val
#        labels = LabelSet(x='x', y='y', text='y', level='glyph',
#                         x_offset=0, y_offset=0, source=source_b_2,
#                         text_color='black',text_align='center')
#        b_2.add_layout(labels)
#        layout.children[3]=b_2
#

 
 #graphique distribution  monovariée à rajouter point de comparaison   
def update_graph(attr, old, new):
    graph_val = graph_select.value #nom dela variable sélectionné
    new_data=dict()
    
    if df[graph_val].dtype !='object':
        if dot_flag[0] != 0:
            g.renderers.remove(dot[0])
            print(df_retrieved[0][graph_val])
            dot[0]=g.circle(x=round(len(df[graph_val])/2), y=df_retrieved[0][graph_val], size=10, color='red', name='dot') # se positionne au milieu du graph
            print('ajout point rouge')

        new_data['x']= [i for i in range (len(df[graph_val]))]
        new_data['y']= df[graph_val]
        source_g.data['x'] = new_data['x']
        source_g.data['y'] = new_data['y']
        
        layout.children[5]=g
        
    elif df[graph_val].dtype =='object':
        if dot_flag[0] != 0:
            print( df_retrieved[0][graph_val].values)
            print('affiche les renderers',g_2.renderers)
            g_2.renderers.remove(dot_2[0])
            val = [idx for idx, val in enumerate(df[graph_val].value_counts().index) if val == df_retrieved[0][graph_val].values][0] #index de la valeur du graph qui correspond a la variable de l'observation
            print('val', val, 'y', df[graph_val].value_counts().values[val])
            df_retrieved[0][graph_val]#valeur de l'observation
            dot_2[0] = g_2.circle(x=df_retrieved[0][graph_val],
                                  y=max(df[graph_val].value_counts().values)/2,
                                  size=10, color='red',name='dot_2')  # red_dot
            print('ajout point rouge')
        
            new_data['x'] = df[graph_val].value_counts().index #x = nom des valeurs
            new_data['y'] = df[graph_val].value_counts().values
            source_g_2.data['x'] = new_data['x']
            source_g_2.data['y'] = new_data['y']
            g_2.x_range = FactorRange(*[val for val in df[graph_val].unique()])
            g_2.y_range = Range1d(0,max(df[graph_val].value_counts().values)*1.05)
            g_2.xaxis.axis_label = 'index'
            g_2.yaxis.axis_label = graph_val
            labels = LabelSet(x='x', y='y', text='y', level='glyph',
                         x_offset=0, y_offset=0, source=source_g_2,
                         text_color='black',text_align='center')
            g_2.add_layout(labels)
            layout.children[5]=g_2
        
        
    print(graph_val,df[graph_val])

def update_text_loading(attr, old, new):
    text.text=("Veuillez patienter la prédiction est en cours...")
    decoded[0] = base64.b64decode(new)
    curdoc().add_next_tick_callback(upload_data)
    
# FileInput => df => cleaned_df => post => feature_importance & prediction & score
def upload_data():  # charger et envoyer df fastapi nettoyage
    #loading_div[0] = Div(text="File is being loaded...", width=200, height=50)
    flag_upload[0] = 1
    url = "https://docker-projet-7.azurewebsites.net/uploadfile/"
    print("dataset has been uploaded succesfully")
    f = io.BytesIO(decoded[0])
    df_retrieved[0] = pd.read_csv(f) # df importé
    print(df_retrieved[0]['SK_ID_CURR'])
    id = df_retrieved[0]['SK_ID_CURR'].values
    nettoyage(df_retrieved[0])
    if df_retrieved[0].shape[0]==1: #si une seule ligne 
        dot_flag[0]=1   
        df_retrieved[0].to_csv('df_cleaned.csv', index=False)
        files = {'file': open('df_cleaned.csv', 'rb')}
        res = requests.post(url, files=files)
        print('fichier_recup',res.json())
        df_retrieved[1] = res.json()
        text.text = 'Le crédit sera {} pour l\'id {} et le score est de {}'.format(dict_credit[df_retrieved[1]['prediction']],
                                                                                   id[0],
                                                                                   str(df_retrieved[1]['score'])[:6])  # 100057
        #df_filtered = pd.Series(df_retrieved[1]['feature_importance'])[abs(pd.Series(df_retrieved[1]['feature_importance'])) > 0.0001]
        #colonnes = [df_retrieved[1]['nom_colonnes'][col_index] for col_index in range(len(df_retrieved[1]['nom_colonnes'])) if col_index in df_filtered.index ]
        
        df_filtered = pd.DataFrame({'feature_importance':df_retrieved[1]['feature_importance'],
                                    'nom_colonnes': df_retrieved[1]['nom_colonnes']})  # df avec feature importance et colonne
        df_filtered = df_filtered[abs(df_filtered['feature_importance'])> 0.0001]
        df_filtered.sort_values(ascending=True, by='feature_importance')
        
        new_data = dict(x=df_filtered['feature_importance'],
                           y=df_filtered['nom_colonnes'])
    
        print('f_filtré',len(df_filtered),'anc_f', len(df_retrieved[1]['feature_importance']),'col filtrés',len(df_filtered['feature_importance']))
        source_fl.data = new_data
        fl.y_range.factors = df_filtered['nom_colonnes']
        range_x = Range1d(-(min(round(df_filtered['feature_importance']))*0.05),
                          max(round(df_filtered['feature_importance']))*0.05)
        fl.x_range = range_x
    
    print(df_retrieved[0])
    
#################################################### #Auto complete  #############################################################
def update_text_id (attr, old, new):
    text.text=("Veuillez patienter la prédiction est en cours...")
    value_input[0] = new  # valeur dans le champs
    curdoc().add_next_tick_callback(update_id)

def update_id():  # attrname, old, new => obligatoire pour ce composant
    global res
    # pour récupérer la variable pour le graph
    print('update_id_in_progress')
    # res = requests.post(url+value_input) #requete a fastapi
    print(value_input[0])

    if flag_upload[0] == 0:# demo ou aucune donnéee rajoutée
        url =  "https://docker-projet-7.azurewebsites.net/predict/"
        res = requests.post(url+str(value_input[0]))
        print('contenu apres envoi '+str(value_input[0]), res._content)
        
        df_retrieved[0]=df[df['SK_ID_CURR'] == int(value_input[0])] #ligne d'info d'un id 
        ## vérifier si colonnes différentes de 0 et rajouter
        df_retrieved[1] = json.loads(res._content.decode('utf-8'))  # data_retrieved
        new_data = dict()  # cd pour instancier en un seul coup avec les nouvelles données sinon erreur de longueur des colonnes
        # premiere ligne
        df_filtered = pd.DataFrame({'feature_importance':df_retrieved[1]['feature_importance'],
                                    'nom_colonnes': df_retrieved[1]['nom_colonnes']})  # df avec feature importance et colonne
        df_filtered = df_filtered[abs(df_filtered['feature_importance'])> 0.0001]
        df_filtered.sort_values(ascending=True, by='feature_importance')
        
        new_data = dict(x=df_filtered['feature_importance'],
                           y=df_filtered['nom_colonnes'])
    
        print('f_filtré',len(df_filtered),'anc_f', len(df_retrieved[1]['feature_importance']),'col filtrés',len(df_filtered['feature_importance']))
        source_fl.data = new_data
        fl.y_range.factors = df_filtered['nom_colonnes']
        range_x = Range1d(-(min(round(df_filtered['feature_importance']))*0.05),
                          max(round(df_filtered['feature_importance']))*0.05)
        fl.x_range = range_x
        
        text.text = 'Le crédit sera {} pour l\'id {} et le score est de {}'.format(
        dict_credit[df_retrieved[1]['prediction']], value_input[0],str(df_retrieved[1]['score'])[:6])  # 100057
        dot_flag[0]=1
#####################################################  select   ###############################################################
x_select = Select(title='X-Axis Variable:',
                  options=[val  for val in df.columns if val not in  ['SK_ID_CURR','FLAG_MOBIL']], value='NAME_TYPE_SUITE')
y_select = Select(title='Y-Axis Variable:',
                  options=[val  for val in df.select_dtypes(exclude='object').columns if val not in  ['SK_ID_CURR','FLAG_MOBIL']],value='DAYS_REGISTRATION')
x_select_2 = Select(title='X-Axis Variable:',
                  options=[val  for val in df.columns if val not in  ['SK_ID_CURR','FLAG_MOBIL']], value='NAME_TYPE_SUITE')
y_select_2 = Select(title='Y-Axis Variable:',
                  options=[val  for val in df.select_dtypes(exclude='object').columns if val not in  ['SK_ID_CURR','FLAG_MOBIL']],value='DAYS_REGISTRATION')
graph_select = Select(title='Distribution de la variable:',
                  options=[val  for val in df.columns if val not in  ['SK_ID_CURR','FLAG_MOBIL']],value='DAYS_REGISTRATION')
#####################################################  widgets callback config   ###############################################################
# Bind the callback function to the Select widgets
x_select.on_change('value', update_select_1)
y_select.on_change('value', update_select_1)
x_select_2.on_change('value', update_select_2)
y_select_2.on_change('value', update_select_2)
graph_select.on_change('value', update_graph)
auto_complete_input.on_change('value', update_text_id)#new
file_input.on_change('value', update_text_loading)#new
#####################################################  layout   ###############################################################
# Create a layout for the widgets
#loading_div[0] = Div(text="Loading file...", width=200, height=50)

# create an empty Div widget
spacer = Div(text="<div style='height: 50px'></div>")
#spacer = Spacer(width=1, height=50)
inputs = row(x_select, y_select)
inputs_1 = row(x_select_2, y_select_2)
layout = column([inputs,
                b,
                inputs_1,
                b_2,
                graph_select,
                g],sizing_mode='stretch_both')

print('fin')

# Add the widgets to the current document
curdoc().add_root(row(column([text,
                        file_input,
                        auto_complete_input,
                        fl,
                        f]),layout))
#curdoc().add_root(row(column(layout),column(graph_select,spacer,g))) back up  si erreur
