import streamlit as st
from streamlit_ketcher import st_ketcher
import numpy as np
import pandas as pd
from rdkit import Chem
# from rdkit.Chem import rdFingerprintGenerator
# from rdkit.Chem.Draw import rdDepictor
# import time
# rdDepictor.SetPreferCoordGen(True)

from SMILESX import main
from SMILESX import loadmodel
from SMILESX import inference
from SMILESX import interpret

import os


st.title('Streamlit + SMILESX :rocket:')

choice = st.radio('Do you have a list of SMILES to predict?',('No','Yes'))

if choice =='Yes':
    uploaded_file = st.file_uploader('Upload your data', type = ['.csv','.xlsx'])

    if not uploaded_file:
        st.stop()
    if uploaded_file.name[-4:] == '.csv':
            data_file = pd.read_csv(uploaded_file)
    elif uploaded_file.name[-5:] == '.xlsx':
        data_excel_file = pd.ExcelFile(uploaded_file)
        data_file = pd.read_excel(data_excel_file, sheet_name = data_excel_file.sheet_names[0])
        st.session_state['data_file'] = data_file
    for i in range(len(data_file)):
            try:
                data_value = data_file.iloc[i][0]
                test = Chem.MolFromSmiles(data_value)
                test_2 = Chem.MolToSmiles(test)
            except (RuntimeError, TypeError, NameError):
                st.error("Make sure that all your data are SMILES, line %s"%(i+2) +" '" +data_value +" ' " + " is not a SMILES.")
                st.stop()
    column1, column2 = st.columns(2)

    with column1:
        st.write('Your list :')
        st.dataframe(data_file)
    with column2:
        smile_input = st.selectbox('Select the one to predict',options=data_file)

    st.session_state['smile_input']=smile_input
    res  = st_ketcher(st.session_state['smile_input'])

if choice=='No':
    if 'smiles' not in st.session_state:
        molecule = st.text_input(label="Molecule", value='CC(=O)Nc1ccc(O)cc1', placeholder='example CC=O')
    else:
        molecule = st.text_input(label="Molecule", value=st.session_state['smiles'], placeholder='example CC=O')

    res = st_ketcher(molecule)
    st.markdown(f"Smile code: ``{res}``")
    st.session_state['smiles']=res

### SMILEX part ###
st.header('Prediction on Hydration free energy (kcal/mol)')

st.write('The SMILES that will be used for the prediction is :\n\n %s'%res)

list_properties = ['Hydration free energy','Polar Area','logP']
st.selectbox('Choose the properties to predict :',list_properties)

col1, col2 = st.columns(2)

with col1:
    fingerprint_list = np.array(['Morgan','AtomPair','RDKitFP','TopologicalTorsion'])
    finger_choice = st.radio('Choose your FingerPrint Generator',fingerprint_list)
    st.write(finger_choice)

with col2:
    prediction_list = np.array(['ElasticNet', 'RandomForestRegressor', 'XGBRegressor','SMILESX'])
    prediction_choice = st.radio('Choose your Prediction model',prediction_list)
    st.write(prediction_choice)

mol = Chem.MolFromSmiles(res)

smiles_button = st.button(label = 'Launch the prediction')
if smiles_button:
    data_path = "/Users/c.bajan/Visual Code Projects/JSME_SMILE/data/FreeSolv_SAMPL.csv"
    data_name = 'FreeSolv'
    data_label = 'Hydration free energy'
    data_units = 'kcal/mol'
    data = pd.read_csv(data_path)

    model = loadmodel.LoadModel(data_name=data_name,
                                augment=False,
                                gpu_ind=0,
                                use_cpu=True,
                                return_attention=True)

    preds = inference.infer(model=model,
                            data_smiles=[Chem.MolToSmiles(mol)],
                            augment=False,
                            check_smiles=True,
                            log_verbose=True)
    st.session_state['preds'] = preds

    inter = interpret.interpret(model=model,
                                smiles=preds['SMILES'],
                                pred=preds[['mean', 'sigma']],
                                check_smiles=True,
                                log_verbose=True)
    st.session_state['path1']='./outputs/FreeSolv/Can/Interpret/2D_Interpretation_0.png'
    st.session_state['path2']='./outputs/FreeSolv/Can/Interpret/1D_Interpretation_0.png'
    st.session_state['path3']='./outputs/FreeSolv/Can/InterpretTemporal_Relative_Distance_smiles_0.png'
if 'preds' in st.session_state: 
    st.write('Prediction on Hydration free energy (kcal/mol)')
    st.write(st.session_state['preds'])
    st.image(st.session_state['path1'])
    st.image(st.session_state['path2'])
    st.write('Temporal relative distance display the influence of each element of the SMLIE to the prediction. \n\n'
    'When the value is closer to zero it means that this element help for the prediction.')
    st.image(st.session_state['path3']) 
    

# ### Extracted from the streamlit_test.py, need to adapt it when better understanding of trained model.
# def analyze_function(prediction_choice, data, target, crossval, lim_feature, k_num=3,random_state = 0, n_estimators = 100):
        
#             if prediction_choice == 'ElasticNet':
#                 regressor = ElasticNet(random_state = 0)
#             elif prediction_choice == 'RandomForestRegressor':
#                 regressor = RandomForestRegressor(n_estimators = n_estimators, random_state = 0)
#             elif prediction_choice == 'XGBRegressor':
#                 regressor = XGBRegressor(n_estimators = n_estimators, seed = 0)
                
#             def cross_val_est_fn(clf, x, y, cv):
#                 predictions = cross_val_predict(estimator = clf, 
#                                                 X = x, 
#                                                 y = y, 
#                                                 cv = cv)
#                 validation_arrays = cross_validate(estimator = clf, 
#                                                   X = x, 
#                                                   y = y, 
#                                                   cv = cv, 
#                                                   scoring = scorer, 
#                                                   return_estimator = True)   

#                 test_mae, test_mse, estimator = validation_arrays['test_mae'], validation_arrays['test_mse'], validation_arrays['estimator']

#                 return predictions, -test_mae, np.sqrt(-test_mse), estimator
            
#             ####
#             #### Need to change that to adapt it 
#             feature_columns = np.where(data.columns == target)[0][0]
#             ####
#             ####

#             clf = make_pipeline(StandardScaler(), regressor)

#             if crossval=='LeaveOneOut':
#                 crossvalidation = LeaveOneOut()
#             elif crossval=='K-Fold':
#                 crossvalidation = KFold(n_splits=k_num, shuffle=True, random_state=0)

#             mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
#             mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
#             scorer = {'mae': mae_scorer, 'mse': mse_scorer}
            
#             # Prediction with cross validation using LeaveOneOut or K-fold and the method selected by user
#             pred_, test_mae_, test_rmse_, est_ = cross_val_est_fn(clf = clf, x = data.iloc[:,0:lim_feature], 
#                                                                   y = data.iloc[:,feature_columns], 
#                                                                   cv = crossvalidation)
#             return pred_, test_mae_, test_rmse_, est_





