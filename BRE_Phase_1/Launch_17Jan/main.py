# Comment this out in VM
import functions_framework

import json
import pandas as pd
import numpy as np
import datetime
from datetime import datetime

# Comment this out in VM
from google.cloud import storage

def split_name(name):
        return pd.Series(name)

def data_parse_Inq(Dict):
    """
    Function to convert dictionary file to pandas dataframe.
    """
    ## 1st Key Dataframe
    df_Bur_Pull_ID = pd.DataFrame(Dict['InquiryResponseHeader'])

    ## 2nd Key Dataframe
    df_Bur_Inq_Info = pd.DataFrame(Dict['InquiryRequestInfo'])
    Phn_Det = pd.json_normalize(df_Bur_Inq_Info['InquiryPhones'])
    df_Bur_Inq_Info = pd.concat([df_Bur_Inq_Info,Phn_Det],axis=1)
    df_Bur_Inq_Info.drop(columns=['InquiryPhones'],inplace=True)

    ## Combining Dataframe
    df_Bur = pd.concat([df_Bur_Pull_ID,df_Bur_Inq_Info],axis=1)
    
    return df_Bur

def data_parse_grid(Dict, df_Bur):
    
    ## Account History Grid
    Acct_Hist_Grid_JSON = Dict['CCRResponse']['CIRReportDataLst'][0]['CIRReportData']['RetailAccountDetails']
    Acct_Data = pd.DataFrame({"Del_Grid":Acct_Hist_Grid_JSON})

    ## Normalizing the Grid Data
    Acct_Data = pd.json_normalize(Acct_Data['Del_Grid']).copy()

    ## Splitting the 48 month Grid variable grid
    Acct_Dt_48Mon_His_All_Grid = Acct_Data['History48Months'].apply(split_name).copy()
    Acct_Data.drop(columns='History48Months', inplace=True)

    ## Exploding the 48 Month History Grid
    Acct_Data_All_Grid_Nor = pd.DataFrame()
    for col in Acct_Dt_48Mon_His_All_Grid.columns:
        df_temp = pd.json_normalize(Acct_Dt_48Mon_His_All_Grid[col])
        df_temp.columns = [f"{col_temp}_{col}" for col_temp in df_temp.columns]
        Acct_Data_All_Grid_Nor = pd.concat([Acct_Data_All_Grid_Nor,df_temp], axis=1).copy()

    ## Merging data to original data
    Acct_Data = pd.concat([Acct_Data,Acct_Data_All_Grid_Nor],axis=1).copy()

    ## Repeat Data
    n = Acct_Data.shape[0]
    columns = [col for col in df_Bur.columns]
    df_Bur = pd.DataFrame(np.repeat(df_Bur.values, n, axis=0))
    df_Bur.columns = columns

    ## Concatenating the data
    df_Bur = pd.concat([df_Bur,Acct_Data],axis=1).copy()

    return df_Bur

def Data_Rename_ETC(df_Bur):   
    """
    This function changes the format of the data to the one required for policy making.
    """
    
    df_Bur_col = [col for col in df_Bur.columns]
    
    ## Renaming if variable is present
    if 'HighCredit' in df_Bur_col:
        df_Bur.rename(columns={'HighCredit':'HIGHCREDIT'}, inplace=True)
    if 'CreditLimit' in df_Bur_col:
        df_Bur.rename(columns={'CreditLimit':'CREDITLIMIT'}, inplace=True)
    if 'SanctionAmount' in df_Bur_col:
        df_Bur.rename(columns={'SanctionAmount' : 'SANCTIONAMOUNT'}, inplace=True)
    if 'InterestRate' in df_Bur_col:
        df_Bur.rename(columns={'InterestRate' : 'INTERESTRATE'}, inplace=True)
    if 'WriteOffAmount' in df_Bur_col:
        df_Bur.rename(columns={'WriteOffAmount' : 'WRITEOFFAMOUNT'}, inplace=True)
    
    ## Assigning features based on which variable is present
    if ('HighCredit' not in df_Bur_col) and ('SanctionAmount' in df_Bur_col):
        df_Bur['HIGHCREDIT'] = df_Bur['SANCTIONAMOUNT'].copy()
    elif ('HighCredit' not in df_Bur_col) and ('CreditLimit' in df_Bur_col):
        df_Bur['HIGHCREDIT'] = df_Bur['CREDITLIMIT'].copy()
    elif ('HighCredit' not in df_Bur_col):
        df_Bur['HIGHCREDIT'] = np.nan
        
    ## Assigning Features if 2 sources are present
    if ('HighCredit' in df_Bur_col) and ('SanctionAmount' in df_Bur_col):
        df_Bur['HIGHCREDIT'] = np.where(df_Bur['HIGHCREDIT'].isnull(),df_Bur['SANCTIONAMOUNT'],df_Bur['HIGHCREDIT']).copy()
        
    ## Creating Feature values if not present in the data
    if 'InterestRate' not in df_Bur_col:
        df_Bur['INTERESTRATE'] = np.nan
    if 'WriteOffAmount' not in df_Bur_col:
        df_Bur['WRITEOFFAMOUNT'] = np.nan
        
    ## Creating extra features
    df_Bur['SUITFILEDSTATUS'] = df_Bur['SuitFiledStatus_0']
    df_Bur['DAYS_PAST_DUE'] = df_Bur['PaymentStatus_0']

    ## Account Unique ID is not present
    df_Bur['ACCT_UNIQ_ID'] = [i for i in range(df_Bur.shape[0])]

    ## Interest Rate is not present in the D2C
    ## What is TERMS_FREQUENCY
    ## We will have to create Written off amount variable
    rename_dict = {
    'ReportOrderNO' : 'REPORTNUMBER',
    'CustRefField' : 'REFERENCE_NO',
    'AccountNumber' : 'ACCT_NUMBER',
    'AccountStatus' : 'ACCOUNTSTATUS',
    'AccountType' : 'ACCOUNTTYPE',
    'AssetClassification' : 'ASSETCLASSIFICATION',
    'Balance' : 'BALANCE',
    'DateClosed' : 'DATECLOSED',
    'DateOpened' : 'DATEOPENED',
    'DateReported' : 'DATEREPORTED',
    'InstallmentAmount' : 'INSTALLMENTAMOUNT',
    'LastPayment' : 'LASTPAYMENT',
    'LastPaymentDate': 'LASTPAYMENTDATE',
    'OwnershipType':'OWNERSHIPTYPE',
    'PastDueAmount':'PASTDUEAMOUNT',
    'RepaymentTenure':'REPAYMENTTENURE',
    'TermFrequency':'TERMFREQUENCY',
    'ClientID':'CLIENT_ID'}

    df_Bur.rename(columns=rename_dict, inplace=True)
    
    ## Adding Masking columns
    n_col_hist_max = len([col for col in df_Bur.columns if "SuitFiledStatus" in col])
    mask_col = [f"{col_type}_{i}" for col_type in ("key","PaymentStatus","SuitFiledStatus","AssetClassificationStatus") for i in range(n_col_hist_max,49,1)]
    df_Bur[mask_col] = np.nan

    ## Renaming columns
    rename_dict = {f"{old}_{i}":f"{new}_{i+1}" for i in range(49) for old,new in zip(['PaymentStatus','SuitFiledStatus','AssetClassificationStatus'],['ACCT_ST','SUIT_FILED_ST','ASSET_CL_ST'])}
    df_Bur.rename(columns=rename_dict, inplace=True)

    return df_Bur

def Data_Rename_NTC(df_Bur):
    rename_dict = {
        'ReportOrderNO' : 'REPORTNUMBER',
        'CustRefField' : 'REFERENCE_NO',
        'ClientID':'CLIENT_ID',
    }

    df_Bur.rename(columns=rename_dict, inplace=True)
    return df_Bur

def Bur_Status_Flag(df_Bur,data_json):
    
    ## Cases tagged as NTC Cases
    if 'Error' in data_json['CCRResponse']['CIRReportDataLst'][0].keys():
        df_Error = pd.DataFrame(data_json['CCRResponse']['CIRReportDataLst'][0]['Error'], index=[0])
        df_Bur = pd.concat([df_Bur,df_Error], axis=1)
        df_Bur['Bur_Hist_Sta'] = 'NTC'
        Bur_Hist_Flag = 'NTC'
    else:
        Bur_Hist_Flag = 'ETC'
        df_Bur['Bur_Hist_Sta'] = 'ETC'
    return (df_Bur, Bur_Hist_Flag)

def Score_Bur(data_json, df_Bur):
    ## Getting Available Keys
    Avail_Keys = data_json['CCRResponse']['CIRReportDataLst'][0]['CIRReportData'].keys()
    
    ## If Score is 
    if 'ScoreDetails' in Avail_Keys:
        Score = int(data_json['CCRResponse']['CIRReportDataLst'][0]['CIRReportData']['ScoreDetails'][0]['Value'])
        df_Bur['Score'] = Score
    else:
        df_Bur['Score'] = 99999  
    return df_Bur

def Inq_data(json_data,df_Bur):
    Avail_Keys = json_data['CCRResponse']['CIRReportDataLst'][0]['CIRReportData'].keys()

    if 'EnquirySummary' in Avail_Keys:
        Dict_Enq = json_data['CCRResponse']['CIRReportDataLst'][0]['CIRReportData']['EnquirySummary']
        df_Inq_Sum = pd.DataFrame(Dict_Enq, index=[0])

    ## Repeat Data
    n = df_Bur.shape[0]
    columns = [col for col in df_Inq_Sum.columns]
    df_Inq_Sum = pd.DataFrame(np.repeat(df_Inq_Sum.values, n, axis=0))
    df_Inq_Sum.columns = columns
    
    ## Concatenating the data
    df_Bur = pd.concat([df_Bur,df_Inq_Sum],axis=1)
    return df_Bur

def Enq_df(json_data):
    ## Available Keys 
    Avail_Keys = json_data['CCRResponse']['CIRReportDataLst'][0]['CIRReportData'].keys()

    ## Enquiries Dataframe
    if 'Enquiries' in Avail_Keys:
        Enq_dt_json = json_data['CCRResponse']['CIRReportDataLst'][0]['CIRReportData']['Enquiries']
        df_Enq = pd.DataFrame(Enq_dt_json)
        return df_Enq
    else:
        return None
    
## Cases with invalid input
def Invalid_Input_Cases(json_data, df_Bur):
    
    Avail_keys_json = json_data.keys()
    if 'Error' in Avail_keys_json:
        df_Error = pd.DataFrame(json_data['Error'],index=[0])
        df_Bur = pd.concat([df_Bur,df_Error],axis=1)
        df_Bur['Input_Detail_Type'] = 'Invalid_Detail'
        Input_Detail_Type = 'Invalid_Detail'
            
    else:
        df_Bur['Input_Detail_Type'] = 'Valid_Detail'
        Input_Detail_Type = 'Valid_Detail'
#     display(df_Bur_test)

    return (df_Bur,Input_Detail_Type)

def var_treatment(df_Bur):
    
    ## 1. Account Status Features
    ## Replacing the Dataset
    Raw_feat_Acct_St_list = [f"ACCT_ST_{i}" for i in range(1,49,1)]
    
    ## Treatment of the data
    df_Bur[Raw_feat_Acct_St_list] = df_Bur[Raw_feat_Acct_St_list].fillna('-16').copy()
    Replace_Mapping = {'000':'0', '01+':'1', '30+':'2', '60+':'3', '90+':'4', '120+':'5', '180+':'6', '360+':'7', '540+':'8', '720+':'9', 'PWOS':'10', 'RGM':'11', 'SET':'12', 'WOF':'13', 'BK':'14', 'RES':'15', 'SPM':'16', 'WDF':'17', 'SF':'18', 'CLSD':'-1', '*':'-2', 'INAC':'-3', 'ADJ':'-4', 'STD':'-5', 'LOSS':'-6', 'RNC':'-7', 'LNSB':'-8', 'FPD':'-9', 'SUB':'19', 'DBT':'20', 'DIS':'-12', 'LAND':'-13', 'CUF':'21', 'DEC':'-15', '*':'-16', 'NEW':'-17' }
    
    ## Replacing non required values with np.nan
    Acct_St_Cd = [cd for cd in Replace_Mapping.keys()] + ['-16']
    df_Bur.loc[:,Raw_feat_Acct_St_list] = df_Bur.loc[:,Raw_feat_Acct_St_list].where(df_Bur[Raw_feat_Acct_St_list].isin(Acct_St_Cd), '-16' ).copy()
    
    ## Replacing codes with numeric values
    Replace_Dict = {f"ACCT_ST_{i}": Replace_Mapping for i in range(1,49,1)}
    df_Bur.replace(Replace_Dict,inplace=True)
    
    ## Converting data to numeric
    df_Bur[Raw_feat_Acct_St_list] = df_Bur[Raw_feat_Acct_St_list].astype(int).copy()
    Replace_Dict = {f"ACCT_ST_{i}":{-16:np.nan} for i in range(1,49,1)}
    df_Bur.replace(Replace_Dict,inplace=True)

    
    ## 2. Suit Filed Status Features
    ## Feature Names
    Raw_feat_Suit_St_list = [f"SUIT_FILED_ST_{i}" for i in range(1,49,1)]

    ## Treatment of the data
    df_Bur[Raw_feat_Suit_St_list] = df_Bur[Raw_feat_Suit_St_list].fillna('-1').copy()
    Replace_Mapping = {'NS':'-3', 'SF':'1', 'TP':'9', 'DI':'10', 'ED':'11', '*':'-2'}
    
    ## Replacing non required values with np.nan
    Suit_St_Cd = [cd for cd in Replace_Mapping.keys()] + ['-1']
    df_Bur.loc[:,Raw_feat_Suit_St_list] = df_Bur.loc[:,Raw_feat_Suit_St_list].where(df_Bur[Raw_feat_Suit_St_list].isin(Suit_St_Cd), '-1' ).copy()
    
    ## Replacing codes with numeric values mappings
    Replace_Dict = {f"SUIT_FILED_ST_{i}": Replace_Mapping for i in range(1,49,1)}
    df_Bur.replace(Replace_Dict, inplace=True)

    ## Converting case to not available case
    df_Bur[Raw_feat_Suit_St_list] = df_Bur[Raw_feat_Suit_St_list].astype(int).copy()
    Replace_Dict = {f"SUIT_FILED_ST_{i}":{-1:np.nan} for i in range(1,49,1)}
    df_Bur.replace(Replace_Dict,inplace=True)
    
    
    ## 3.Asset Classification Status Features
    ## Feature Names
    Raw_feat_Asset_St_list = [f"ASSET_CL_ST_{i}" for i in range(1,49,1)]

    ## Treatment of the data
    df_Bur[Raw_feat_Asset_St_list] = df_Bur[Raw_feat_Asset_St_list].fillna('-1').copy()
    Replace_Mapping = {'STD':'1', 'SUB':'2', 'DBT':'3', 'LOSS':'4', 'SMA':'5', 'NPA':'6', 'SMA 0':'7', 'SMA 1':'8', 'SMA 2':'9', 'DBT 1':'10', 'DBT 2':'11', 'DBT 3':'12', '*':'-1' }

    ## Replacing non required values with np.nan
    Asset_St_Cd = [cd for cd in Replace_Mapping.keys()] + ['-1']
    df_Bur.loc[:,Raw_feat_Asset_St_list] = df_Bur.loc[:,Raw_feat_Asset_St_list].where(df_Bur[Raw_feat_Asset_St_list].isin(Asset_St_Cd), '-1' ).copy()

    ## Replacing codes with numeric values mappings
    Replace_Dict = {f"ASSET_CL_ST_{i}": Replace_Mapping for i in range(1,49,1)}
    df_Bur.replace(Replace_Dict, inplace=True)

    ## Converting case to not available case
    df_Bur[Raw_feat_Asset_St_list] = df_Bur[Raw_feat_Asset_St_list].astype(int).copy()
    Replace_Dict = {f"ASSET_CL_ST_{i}":{-1:np.nan} for i in range(1,49,1)}
    df_Bur.replace(Replace_Dict,inplace=True)
    
    return df_Bur

def Data_Cleaning(df_Bur):
    
    Avail_Cols = [col for col in df_Bur]
    
    if 'DATEOPENED' not in Avail_Cols:
        df_Bur['DATEOPENED'] = np.nan
    if 'DATECLOSED' not in Avail_Cols:
        df_Bur['DATECLOSED'] = np.nan
    if 'DATEREPORTED' not in Avail_Cols:
        df_Bur['DATEREPORTED'] = np.nan
    
    # ## Changing Date Foramt
    df_Bur['DATEOPENED_N'] = pd.to_datetime(df_Bur['DATEOPENED'], errors='coerce', dayfirst=True)
    df_Bur['DATECLOSED_N'] = pd.to_datetime(df_Bur['DATECLOSED'], errors='coerce', dayfirst=True)
    df_Bur['DATEREPORTED_N'] = pd.to_datetime(df_Bur['DATEREPORTED'], errors='coerce', dayfirst=True)
    df_Bur['WRITEOFFAMOUNT'] = df_Bur['WRITEOFFAMOUNT'].astype(float).copy()
    df_Bur['BALANCE'] = df_Bur['BALANCE'].astype(float).copy()
    
    df_Bur.drop(columns=['DATEOPENED','DATECLOSED','DATEREPORTED'], inplace=True)
    df_Bur = df_Bur[~df_Bur['DATEOPENED_N'].isnull()].reset_index(drop=True).copy()
    
    # Calculating Tenure as Close Date - Open Date
    df_Bur['Tenure_Clo_Ope'] = (df_Bur['DATECLOSED_N'].dt.year*12+df_Bur['DATECLOSED_N'].dt.month)-(df_Bur['DATEOPENED_N'].dt.year*12-df_Bur['DATEOPENED_N'].dt.month)
    df_Bur['Tenure_Rep_Ope'] = (df_Bur['DATEREPORTED_N'].dt.year*12+df_Bur['DATEREPORTED_N'].dt.month)-(df_Bur['DATEOPENED_N'].dt.year*12-df_Bur['DATEOPENED_N'].dt.month)
    ## Creating Tenure Available Values
    df_Bur['Tenure_Avai'] = np.where(df_Bur['Tenure_Clo_Ope'].isnull(),df_Bur['Tenure_Rep_Ope'],df_Bur['Tenure_Clo_Ope']).copy()
    df_Bur["Tenure_Avai"] = df_Bur["Tenure_Avai"].astype('int').copy()
    
    return df_Bur

def json_serialize(df_Bur, df_Enq): 
    
    Avail_Cols = [col for col in df_Bur.columns]
    
    if 'DATEOPENED_N' not in Avail_Cols:
        df_Bur['DATEOPENED_N'] = np.nan
        df_Bur['DATEOPENED_N'] = pd.to_datetime(df_Bur['DATEOPENED'], errors='coerce', dayfirst=True)
    if 'DATECLOSED_N' not in Avail_Cols:
        df_Bur['DATECLOSED_N'] = np.nan
        df_Bur['DATECLOSED_N'] = pd.to_datetime(df_Bur['DATECLOSED_N'], errors='coerce', dayfirst=True)
    if 'DATEREPORTED_N' not in Avail_Cols:
        df_Bur['DATEREPORTED_N'] = np.nan
        df_Bur['DATEREPORTED_N'] = pd.to_datetime(df_Bur['DATEREPORTED_N'], errors='coerce', dayfirst=True)
    
    ## Converting Datetime Data to String in Base Data
    df_Bur['DATEOPENED_N'] = df_Bur['DATEOPENED_N'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_Bur['DATECLOSED_N'] = df_Bur['DATECLOSED_N'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_Bur['DATEREPORTED_N'] = df_Bur['DATEREPORTED_N'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    ## Converting Datetime Data to String in Enquiry Data
    if df_Enq is not None:
        df_Enq['Date_N'] = df_Enq['Date_N'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df_Bur, df_Enq

def Last_Month_Feat(df_Bur):
    
    ## Creating Time Frame Features with Raw variable list
    Raw_feat_list = []
    Raw_feat_flat_list = []
    for raw_var_code in ["SUIT_FILED_ST","ACCT_ST","ASSET_CL_ST"]:
        Raw_feat_list.append([f'{raw_var_code}_{i}' for i in range(1,49,1)])
        Raw_feat_flat_list = Raw_feat_flat_list + [f'{raw_var_code}_{i}' for i in range(1,49,1)]
        
    ## Creating Time Frame Features with Named variable list
    Named_feat_list = []
    Named_feat_flat_list = []
    for named_var_code in ['SuitSt','AcctSt','AssetSt']:
        Named_feat_list.append([f'{named_var_code}_L{i}M' for i in range(0,49,1)])
        Named_feat_flat_list = Named_feat_flat_list + [f'{named_var_code}_L{i}M' for i in range(0,49,1)]
        
    ## Getting the month and the year
    month = datetime.now().month
    year = datetime.now().year

    ## All accounts will be closed or last reported before the current date.
    df_Bur['Non_Avail_Per'] = (year*12+month) - (df_Bur['DATEREPORTED_N'].dt.year*12+df_Bur['DATEREPORTED_N'].dt.month)
    
    ## Initiating the named features
    df_Bur[Named_feat_flat_list] = np.nan

    # Code for dataset creation
    df_Bur_A = df_Bur[df_Bur["Non_Avail_Per"]<=48].reset_index(drop=True).copy()
    df_Bur_NA = df_Bur[df_Bur["Non_Avail_Per"]>48].reset_index(drop=True).copy()
    
    # Code for dataset creation
    for non_avai_per in df_Bur_A["Non_Avail_Per"].unique():
        for var_type, var_code in zip(Raw_feat_list,Named_feat_list):
            df_Bur_A.loc[df_Bur_A["Non_Avail_Per"]==non_avai_per,var_code[non_avai_per:48]] = df_Bur_A.loc[df_Bur_A["Non_Avail_Per"]==non_avai_per,var_type[0:48-non_avai_per]].values
    
    ## Dropping Raw Features
    df_Bur_A.drop(columns=Raw_feat_flat_list,inplace=True)
    df_Bur_NA.drop(columns=Raw_feat_flat_list,inplace=True)
    
    df_Bur = pd.concat([df_Bur_A,df_Bur_NA]).copy()
    df_Bur.reset_index(inplace=True,drop=True)
    
    return df_Bur

def Feat_Treat_DPD(df_Bur):
    ## DPD Feature
    ## Replacing the Dataset
    Raw_feat_DPD_list = ['DAYS_PAST_DUE']

    ## Treatment of the data
    df_Bur[Raw_feat_DPD_list] = df_Bur[Raw_feat_DPD_list].fillna('-16').copy()
    Replace_Mapping = {'000':'0', '01+':'1', '30+':'2', '60+':'3', '90+':'4', '120+':'5', '180+':'6', '360+':'7', '540+':'8', '720+':'9', 'PWOS':'10', 'RGM':'11', 'SET':'12', 'WOF':'13', 'BK':'14', 'RES':'15', 'SPM':'16', 'WDF':'17', 'SF':'18', 'CLSD':'-1', '*':'-2', 'INAC':'-3', 'ADJ':'-4', 'STD':'-5', 'LOSS':'-6', 'RNC':'-7', 'LNSB':'-8', 'FPD':'-9', 'SUB':'19', 'DBT':'20', 'DIS':'-12', 'LAND':'-13', 'CUF':'21', 'DEC':'-15', '*':'-16', 'NEW':'-17' }

    ## Replacing non required values with np.nan
    Acct_DPD_Cd = [cd for cd in Replace_Mapping.keys()] + ['-16']
    df_Bur.loc[:,Raw_feat_DPD_list] = df_Bur.loc[:,Raw_feat_DPD_list].where(df_Bur[Raw_feat_DPD_list].isin(Acct_DPD_Cd), '-16' ).copy()

    ## Replacing codes with numeric values
    Replace_Dict = {"DAYS_PAST_DUE": Replace_Mapping}
    df_Bur.replace(Replace_Dict,inplace=True)

    ## Converting data to numeric
    df_Bur[Raw_feat_DPD_list] = df_Bur[Raw_feat_DPD_list].astype(int).copy()
    Replace_Dict = {"DAYS_PAST_DUE":{-16:np.nan}}
    df_Bur.replace(Replace_Dict,inplace=True)
    
    return df_Bur

def Policy_Feat(df_Bur):
    ## Data Available Policy
    df_Bur['Perf_Avail'] = (df_Bur["Non_Avail_Per"]<=48)

    ## Getting the month and the year
    month = datetime.now().month
    year = datetime.now().year

    ## Derog Policy Creiterias
    df_Bur['Acct_MOB'] = (year*12+month) - (df_Bur['DATEOPENED_N'].dt.year*12+df_Bur['DATEOPENED_N'].dt.month)
    Derog_Pol_Cond1 = (df_Bur['Acct_MOB']<=48) & (df_Bur['WRITEOFFAMOUNT']>0)
    Derog_Pol_Cond2 = (df_Bur[[f'AcctSt_L{i}M' for i in range(49)]].max(axis=1,skipna=True)>=5)
    Derog_Pol_Cond3 = (df_Bur[[f'AssetSt_L{i}M' for i in range(49)]].max(axis=1,skipna=True)>=4)
    Derog_Pol_Cond4 = (df_Bur[[f'SuitSt_L{i}M' for i in range(49)]].max(axis=1,skipna=True)>=0)
    Derog_Pol_Cond = Derog_Pol_Cond1 | Derog_Pol_Cond2 | Derog_Pol_Cond3 | Derog_Pol_Cond4
    df_Bur['Derog_Pol'] = Derog_Pol_Cond

    ## Active Accounts
    df_Bur['Act_Acct_Pol'] = (df_Bur['BALANCE']>0) & (df_Bur['DAYS_PAST_DUE']>0)

    ## DPD Ever Policies
    df_Bur['DPD90_Pol_L12M'] = df_Bur[[f'AcctSt_L{i}M' for i in range(1,13,1)]].max(axis=1,skipna=True)>=4
    df_Bur['DPD60_Pol_L12M'] = df_Bur[[f'AcctSt_L{i}M' for i in range(1,13,1)]].max(axis=1,skipna=True)>=3
    df_Bur['DPD30_Pol_L6M'] = df_Bur[[f'AcctSt_L{i}M' for i in range(1,7,1)]].max(axis=1,skipna=True)>=2
    df_Bur['DPD30_Pol_L12M'] = df_Bur[[f'AcctSt_L{i}M' for i in range(1,13,1)]].max(axis=1,skipna=True)>=2

    ## DPD Count Policies
    df_Bur['DPD30_Pol_l12m_cnt'] = (df_Bur[[f'AcctSt_L{i}M' for i in range(1,13,1)]]>=2).sum(axis=1)
    df_Bur['DPD30_Pol_l6m_cnt'] = (df_Bur[[f'AcctSt_L{i}M' for i in range(1,7,1)]]>=2).sum(axis=1)
    df_Bur['DPD30_Pol_l3m_cnt'] = (df_Bur[[f'AcctSt_L{i}M' for i in range(1,4,1)]]>=2).sum(axis=1)

    ## Score Policy
    # df_Bur.groupby('REPORTNUMBER')['SCORE'].min()==df_Bur.groupby('REPORTNUMBER')['SCORE'].max()
    df_Bur['Sco_Pol'] = (df_Bur['Score'] >= 700)
    
    return df_Bur

def Enq_Cnt(df_Bur, df_Enq):
    
    if df_Enq is not None:
        ## Changing Data Type
        df_Enq['Date_N'] = pd.to_datetime(df_Enq['Date'], errors='coerce')  

        ## Taking month and year data
        month = datetime.now().month
        year = datetime.now().year

        ## Enquiries in last 3 months
        df_Enq['INQ_Flag'] = ((year*12+month) - (df_Enq['Date_N'].dt.year*12+df_Enq['Date_N'].dt.month))<=3
        df_Enq_Cnt = df_Enq['INQ_Flag'].sum()

        ## Assigning number of inquiries
        df_Bur['INQ_Flag'] = df_Enq_Cnt
    else:
        df_Bur['INQ_Flag'] = 0
    
    return df_Bur

def Bur_Pol_Agg(df_Bur):
    
    ## Aggregating Acount Level Data
    df_User = df_Bur.groupby(['REPORTNUMBER']).agg(
                                            Sum_Perf_Avail = ('Perf_Avail',sum),
                                            Cnt_Derog_Accts = ('Derog_Pol',sum), 
                                            Cnt_OverDue_Accts= ('Act_Acct_Pol',sum), 
                                            Cnt_DPD90_L12M_Accts=('DPD90_Pol_L12M',sum),
                                            Cnt_DPD60_L12M_Accts=('DPD60_Pol_L12M',sum),
                                            Cnt_DPD30_L6M_Accts=('DPD30_Pol_L6M',sum),
                                            Cnt_DPD30_L12M_Accts=('DPD30_Pol_L12M',sum),     
                                            Sum_DPD_Ins_L12M=('DPD30_Pol_l12m_cnt',sum),
                                            Sum_DPD_Ins_L6M=('DPD30_Pol_l6m_cnt',sum),
                                            Sum_DPD_Ins_L3M=('DPD30_Pol_l3m_cnt',sum),
                                            Max_Inq_L3M = ('INQ_Flag',max),
                                            Max_Score = ('Sco_Pol',max),
                                            Cnt_Accts = ('ACCT_NUMBER','count')).copy()
    
    ## Creating Decision Features
    df_User['Sum_Perf_Avail_Dec'] = (df_User['Sum_Perf_Avail']>0)[0]
    df_User['Cnt_Derog_Accts_Dec'] = (df_User['Cnt_Derog_Accts']==0)[0]
    df_User['Cnt_OverDue_Accts_Dec'] = (df_User['Cnt_OverDue_Accts']<=1)[0]
    df_User['Cnt_DPD90_L12M_Accts_Dec'] = (df_User['Cnt_DPD90_L12M_Accts']<=0)[0]
    df_User['Cnt_DPD60_L12M_Accts_Dec'] = (df_User['Cnt_DPD60_L12M_Accts']<=1)[0]
    df_User['Cnt_DPD30_L6M_Accts_Dec'] = (df_User['Cnt_DPD30_L6M_Accts']<=1)[0]
    df_User['Cnt_DPD30_L12M_Accts_Rt_Dec'] = ((df_User['Cnt_DPD30_L12M_Accts']/df_User['Cnt_Accts'])<=0.33)[0]
    df_User['Sum_DPD_Ins_L12M_Dec'] = (df_User['Sum_DPD_Ins_L12M']<=5)[0]
    df_User['Sum_DPD_Ins_L6M_Dec'] = (df_User['Sum_DPD_Ins_L6M']<=3)[0]
    df_User['Sum_DPD_Ins_L3M_Dec'] = (df_User['Sum_DPD_Ins_L3M']<=0)[0]
    df_User['Max_Inq_L3M_Dec'] = (~(df_User['Max_Inq_L3M']>10))[0]
    df_User['Max_Score_Dec'] = (df_User['Max_Score']==1)[0]
    
    ## Non Rejects Population count
    Non_Rej = ((df_User['Sum_Perf_Avail']>0) & (df_User['Cnt_Derog_Accts']==0) & (df_User['Cnt_OverDue_Accts']<=1) & (df_User['Cnt_DPD90_L12M_Accts']<=0) & (df_User['Cnt_DPD60_L12M_Accts']<=1) & (df_User['Cnt_DPD30_L6M_Accts']<=1) & ((df_User['Cnt_DPD30_L12M_Accts']/df_User['Cnt_Accts'])<=0.33) & (df_User['Sum_DPD_Ins_L12M']<=5) & (df_User['Sum_DPD_Ins_L6M']<=3) & (df_User['Sum_DPD_Ins_L3M']<=0) & (~(df_User['Max_Inq_L3M']>10)) & (df_User['Max_Score']==1))
    
    ## Accept Reject Reason
    Accpt_Ind = Non_Rej[0]
    
    return df_User, Accpt_Ind

def NTC_Cond_2(df_Bur):
    
    ## Current Month and Year
    month = datetime.now().month
    year = datetime.now().year
    
    ## MOB Calculation and Number of Accounts
    df_Bur['DATEOPENED_N'] = pd.to_datetime(df_Bur['DATEOPENED'], errors='coerce', dayfirst=True)
    df_Bur['Acct_MOB'] = (year*12+month) - (df_Bur['DATEOPENED_N'].dt.year*12+df_Bur['DATEOPENED_N'].dt.month)
    MOB = df_Bur['Acct_MOB'].max()
    No_Accts = df_Bur.shape[0]
    
    ## NTC Criteria
    if (MOB < 6) & (No_Accts<=1):
        Bur_Hist_Flag = 'NTC'
        NTC_Type = 'NTC_New_User'
    else:
        Bur_Hist_Flag = 'ETC'
        NTC_Type = None
    
    return Bur_Hist_Flag, NTC_Type

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    # Comment Out While Testing
    type_var =  type(request)
    request_json = request.get_json(silent=True)
    request_args = request.args
    # Get the uploaded file
    uploaded_file = request.files.get('file')
    # Get the file content
    file_content = uploaded_file.read().decode('utf-8')

    # Type of file_content
    file_con_type = type(file_content)

    # Parse the JSON
    json_data = json.loads(file_content)
    json_data_type = type(json_data)

#     ## Comment out in cloud function
#     json_data = request
#     json_data_type = type(json_data)

    # if request_json and 'name' in request_json:
    #     name = request_json['name']
    # elif request_args and 'name' in request_args:
    #     name = request_args['name']
    # else:
    #     name = 'World'
    
    ## Creating the data frame from json Inquiry Info
    df_Bur = data_parse_Inq(json_data)
    
    ## Checking for Invalid Input Provided by user
    df_Bur, Input_Detail_Type = Invalid_Input_Cases(json_data,df_Bur)

    # ## Testing for ETC & NTC Flag
    if Input_Detail_Type == 'Valid_Detail':
        df_Bur, Bur_Hist_Flag = Bur_Status_Flag(df_Bur,json_data)
    else:
        Bur_Hist_Flag = 'NA'
    
    if (Bur_Hist_Flag == 'ETC'):
        ## Creating the 48 months history grid data frame from dict and combining with previous data
        df_Bur = data_parse_grid(json_data,df_Bur)
        ## Converting the format of the data
        df_Bur = Data_Rename_ETC(df_Bur)
    else:
        df_Bur = Data_Rename_NTC(df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') :
        Bur_Hist_Flag, NTC_Type  = NTC_Cond_2(df_Bur)
    else:
        NTC_Type = 'NTC_No_Accts'
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Score_Bur(json_data, df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Inq_data(json_data,df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Enq = Enq_df(json_data)
    else:
        df_Enq = None
    
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = var_treatment(df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Feat_Treat_DPD(df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Data_Cleaning(df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Last_Month_Feat(df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Policy_Feat(df_Bur)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Enq_Cnt(df_Bur, df_Enq)
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_User, Accpt_Ind = Bur_Pol_Agg(df_Bur)
    elif Bur_Hist_Flag == 'NTC':
        df_User = None
        Accpt_Ind = True
    else:
        df_User = None
        Accpt_Ind = None
        
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur, df_Enq = json_serialize(df_Bur, df_Enq)
    
    df_Bur_Dict = df_Bur.to_dict('dict')
    type_df_Bur_Dict = type(df_Bur_Dict)
    
    if df_Enq is not None:
        df_Enq_Dict = df_Enq.to_dict('dict')
        type_df_Enq_Dict = type(df_Enq_Dict)
    else:
        df_Enq_Dict = None
    
    if df_User is not None:
        df_User_Dict = df_User.to_dict('dict')
    else:
        df_User_Dict = None
        
    if Accpt_Ind is not None:
        Str_Accpt_Ind = str(Accpt_Ind)
    else:
        Str_Accpt_Ind = None
        
    ## Creating JSON Output
    df_Bur_Out = {
        'Bur_Hist_Sta' : Bur_Hist_Flag,
        'Bur_Data' : df_Bur_Dict,
        'Enq_Data' : df_Enq_Dict,
        'df_User' : df_User_Dict,
        'Accpt_Ind' : Str_Accpt_Ind
    }
    
    ## Comment this out in VM
    # Define your GCS bucket and folder path
    bucket_name = 'fs_user_bre_ml'
    gcs_folder_path = 'BRE_Setup_Testing/API_Response/'

    # Initialize the GCS client
    storage_client = storage.Client()

    # Store the JSON to GCS
    file_name = f"{gcs_folder_path}processed_file_{uploaded_file.filename}"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
            
    # Upload the JSON as a string to GCS
    blob.upload_from_string(json.dumps(df_Bur_Out), content_type='application/json')

#     # return f"The file content data type is {json_data_type} and file content is {json_data}"
    return json.dumps(df_Bur_Out), 200