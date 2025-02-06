# Comment this out in VM
import functions_framework

import json
import pandas as pd
import numpy as np
import datetime
import math
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
    
    ## Dropping seq variable if present
    available_df_bur_columns = [col for col in df_Bur.columns]
    
    if 'seq' in available_df_bur_columns:
        df_Bur.drop(columns=['seq'],inplace=True)
        
    ## Converting phonetype columns datatype from list to string
    avail_col = [col for col in df_Bur.columns]
    
    if 'PhoneType' in avail_col:
        df_Bur['PhoneType'] = df_Bur['PhoneType'].str[0]
    
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
    if 'SuitFiledStatus' in df_Bur_col:
        df_Bur.rename(columns={'SuitFiledStatus' : 'SUITFILEDSTATUS'}, inplace=True)
    
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
    if 'DateOpened' not in df_Bur_col:
        df_Bur['DATEOPENED'] = np.nan
    if 'SuitFiledStatus' not in df_Bur_col:
        df_Bur['SUITFILEDSTATUS'] = df_Bur['SuitFiledStatus_0']
        
    ## Creating extra features
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
    
    ## Case when Bureau Data is empty
    ## This case will happen when dataframe is empty
    if df_Bur.shape[0] == 0:
        df_User = None
        Accpt_Ind = 0
        return df_User, Accpt_Ind
    
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
    
    ## Reset Index
    df_User.reset_index(inplace=True)
    
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
    if ((MOB < 6) | (np.isnan(MOB))) & (No_Accts<=1):
        Bur_Hist_Flag = 'NTC'
        NTC_Type = 'NTC_New_User'
    else:
        Bur_Hist_Flag = 'ETC'
        NTC_Type = None
    
    return Bur_Hist_Flag, NTC_Type

def multiplier_calculation(offline_bnpl_score, min_monthly_income, max_monthly_income): 
    multiplier_grid = {
            'A': {1: 1.15, 2: 1.15, 3: 1.25, 4: 1.50},
            'B': {1: 1.00, 2: 1.00, 3: 1.15, 4: 1.25},
            'C': {1: 0.80, 2: 1.00, 3: 1.15, 4: 1.15},
            'D': {1: 0.80, 2: 0.80, 3: 1.00, 4: 1.15},
            'E': {1: 0.75, 2: 0.80, 3: 1.00, 4: 1.00},
            'F': {1: 0.75, 2: 0.75, 3: 0.80, 4: 1.00},
            'G': {1: 0.65, 2: 0.65, 3: 0.75, 4: 0.75}
        }
    
    # Determine the bucket based on the offline_bnpl_score
    if offline_bnpl_score >= 900:
        score_bucket = 'A'
    elif offline_bnpl_score >= 800:
        score_bucket = 'B'
    elif offline_bnpl_score >= 700:
        score_bucket = 'C'
    elif offline_bnpl_score >= 600:
        score_bucket = 'D'
    elif offline_bnpl_score >= 500:
        score_bucket = 'E'
    elif offline_bnpl_score >= 400:
        score_bucket = 'F'
    else:
        score_bucket = 'G'
        
    # Determine the variable range index based on the given range
    if (min_monthly_income == 0) & (max_monthly_income == 0):
        income_index = 1
    elif (min_monthly_income == 0) & (max_monthly_income == 10000):
        income_index = 1
    elif (min_monthly_income == 10000) & (max_monthly_income == 25000):
        income_index = 2
    elif (min_monthly_income == 25000) & (max_monthly_income == 30000):
        income_index = 2
    elif (min_monthly_income == 30000) & (max_monthly_income == 40000):
        income_index = 3
    elif (min_monthly_income == 40000) & (max_monthly_income == 50000):
        income_index = 3
    elif (min_monthly_income == 50000) & (max_monthly_income is None):
        income_index = 4
    else:
        income_index = 1
        
    multiplier = multiplier_grid[score_bucket][income_index]
    
    return multiplier

def max_capping_calculation(offline_bnpl_score, min_monthly_income, max_monthly_income):
    # Define the updated grid of income values
    max_capping_grid = {
        'A': {1: 1000, 2: 2000, 3: 3000, 4: 4000},
        'B': {1: 750, 2: 1500, 3: 2000, 4: 3000},
        'C': {1: 750, 2: 1000, 3: 1500, 4: 2500},
        'D': {1: 750, 2: 1000, 3: 1000, 4: 2500},
        'E': {1: 500, 2: 750, 3: 1000, 4: 1500},
        'F': {1: 500, 2: 500, 3: 750, 4: 1000},
        'G': {1: 500, 2: 500, 3: 500, 4: 750}
    }
    
    # Determine the bucket based on the offline_bnpl_score
    if offline_bnpl_score >= 900:
        score_bucket = 'A'
    elif offline_bnpl_score >= 800:
        score_bucket = 'B'
    elif offline_bnpl_score >= 700:
        score_bucket = 'C'
    elif offline_bnpl_score >= 600:
        score_bucket = 'D'
    elif offline_bnpl_score >= 500:
        score_bucket = 'E'
    elif offline_bnpl_score >= 400:
        score_bucket = 'F'
    else:
        score_bucket = 'G'
        
    # Determine the income range index based on the given monthly income ranges
    if (min_monthly_income == 0) & (max_monthly_income == 0):
        income_index = 1  # Default index for 'No income'
    elif (min_monthly_income == 0) & (max_monthly_income == 10000):
        income_index = 1  # Income range: <10
    elif (min_monthly_income == 10000) & (max_monthly_income == 25000):
        income_index = 2  # Income range: 10-30
    elif (min_monthly_income == 25000) & (max_monthly_income == 30000):
        income_index = 2  # Income range: 10-30
    elif (min_monthly_income == 30000) & (max_monthly_income == 40000):
        income_index = 3  # Income range: 30-50
    elif (min_monthly_income == 40000) & (max_monthly_income == 50000):
        income_index = 3  # Income range: 30-50
    elif (min_monthly_income == 50000) & (max_monthly_income is None):
        income_index = 4  # Income range: 50+
    else:
        income_index = 1  # Default index for other cases
    
    # Retrieve the corresponding income value from the grid
    max_capping = max_capping_grid[score_bucket][income_index]
    
    return max_capping

def min_capping_calculation(offline_bnpl_score, min_monthly_income, max_monthly_income):
    # Define the updated min capping grid
    min_capping_grid = {
        'A': {1: 300, 2: 500, 3: 750, 4: 1000},
        'B': {1: 300, 2: 500, 3: 750, 4: 1000},
        'C': {1: 300, 2: 500, 3: 750, 4: 750},
        'D': {1: 300, 2: 300, 3: 500, 4: 500},
        'E': {1: 300, 2: 300, 3: 500, 4: 500},
        'F': {1: 300, 2: 300, 3: 300, 4: 300},
        'G': {1: 300, 2: 300, 3: 300, 4: 300}
    }
    
    # Determine the bucket based on the offline_bnpl_score
    if offline_bnpl_score >= 900:
        score_bucket = 'A'
    elif offline_bnpl_score >= 800:
        score_bucket = 'B'
    elif offline_bnpl_score >= 700:
        score_bucket = 'C'
    elif offline_bnpl_score >= 600:
        score_bucket = 'D'
    elif offline_bnpl_score >= 500:
        score_bucket = 'E'
    elif offline_bnpl_score >= 400:
        score_bucket = 'F'
    else:
        score_bucket = 'G'
        
    # Determine the income range index based on the given monthly income ranges
    if (min_monthly_income == 0) & (max_monthly_income == 0):
        income_index = 1  # Default index for 'No income'
    elif (min_monthly_income == 0) & (max_monthly_income == 10000):
        income_index = 1  # Income range: <10
    elif (min_monthly_income == 10000) & (max_monthly_income == 25000):
        income_index = 2  # Income range: 10-30
    elif (min_monthly_income == 25000) & (max_monthly_income == 30000):
        income_index = 2  # Income range: 10-30
    elif (min_monthly_income == 30000) & (max_monthly_income == 40000):
        income_index = 3  # Income range: 30-50
    elif (min_monthly_income == 40000) & (max_monthly_income == 50000):
        income_index = 3  # Income range: 30-50
    elif (min_monthly_income == 50000) & (max_monthly_income is None):
        income_index = 4  # Income range: 50+
    else:
        income_index = 1  # Default index for other cases
    
    # Retrieve the corresponding min capping value from the grid
    min_capping = min_capping_grid[score_bucket][income_index]
    
    return min_capping

def limit_calculation(employee_data,ftpl_data):
    min_monthly_income = employee_data['min_monthly_income']
    max_monthly_income = employee_data['max_monthly_income']
    offline_bnpl_score = ftpl_data['offline_bnpl_score']
    offline_bnpl_limit = ftpl_data['offline_bnpl_limit']

    multiplier = multiplier_calculation(offline_bnpl_score, min_monthly_income, max_monthly_income)
    max_capping = max_capping_calculation(offline_bnpl_score, min_monthly_income, max_monthly_income)
    min_capping = min_capping_calculation(offline_bnpl_score, min_monthly_income, max_monthly_income)

    bnpl_limit_bre = offline_bnpl_limit*multiplier

    if bnpl_limit_bre > max_capping:
        bnpl_limit_bre = max_capping
    if bnpl_limit_bre < min_capping:
        bnpl_limit_bre = min_capping
        
    bnpl_limit_bre = (math.ceil(bnpl_limit_bre/100)*100)

    return bnpl_limit_bre

def applying_feature_schema_bureau_data(df_Bur):
    feature_schema = ['CLIENT_ID',
     'REFERENCE_NO',
     'meesho_user_id',
     'REPORTNUMBER',
     'ProductCode',
     'SuccessCode',
     'Date',
     'Time',
     'InquiryPurpose',
     'FirstName',
     'seq',
     'PhoneType',
     'Number',
     'Input_Detail_Type',
     'ErrorCode',
     'ErrorDesc',
     'Bur_Hist_Sta',
     'ACCT_NUMBER',
     'Institution',
     'ACCOUNTTYPE',
     'OWNERSHIPTYPE',
     'BALANCE',
     'PASTDUEAMOUNT',
     'Open',
     'HIGHCREDIT',
     'LASTPAYMENTDATE',
     'CREDITLIMIT',
     'ACCOUNTSTATUS',
     'source',
     'LASTPAYMENT',
     'SANCTIONAMOUNT',
     'REPAYMENTTENURE',
     'TERMFREQUENCY',
     'INSTALLMENTAMOUNT',
     'Reason',
     'ASSETCLASSIFICATION',
     'key_0',
     'key_1',
     'key_2',
     'key_3',
     'key_4',
     'key_5',
     'key_6',
     'key_7',
     'key_8',
     'key_9',
     'key_10',
     'key_11',
     'key_12',
     'key_13',
     'key_14',
     'key_15',
     'key_16',
     'key_17',
     'key_18',
     'key_19',
     'key_20',
     'key_21',
     'key_22',
     'key_23',
     'key_24',
     'key_25',
     'key_26',
     'key_27',
     'key_28',
     'key_29',
     'key_30',
     'key_31',
     'key_32',
     'key_33',
     'key_34',
     'key_35',
     'key_36',
     'key_37',
     'key_38',
     'key_39',
     'key_40',
     'key_41',
     'key_42',
     'key_43',
     'key_44',
     'key_45',
     'key_46',
     'key_47',
     'key_48',
     'INTERESTRATE',
     'WRITEOFFAMOUNT',
     'SUITFILEDSTATUS',
     'DAYS_PAST_DUE',
     'ACCT_UNIQ_ID',
     'ACCT_ST_49',
     'SUIT_FILED_ST_49',
     'ASSET_CL_ST_49',
     'DATEOPENED_N',
     'Acct_MOB',
     'Score',
     'Purpose',
     'Total',
     'Past30Days',
     'Past12Months',
     'Past24Months',
     'DATECLOSED_N',
     'DATEREPORTED_N',
     'Tenure_Clo_Ope',
     'Tenure_Rep_Ope',
     'Tenure_Avai',
     'Non_Avail_Per',
     'SuitSt_L0M',
     'SuitSt_L1M',
     'SuitSt_L2M',
     'SuitSt_L3M',
     'SuitSt_L4M',
     'SuitSt_L5M',
     'SuitSt_L6M',
     'SuitSt_L7M',
     'SuitSt_L8M',
     'SuitSt_L9M',
     'SuitSt_L10M',
     'SuitSt_L11M',
     'SuitSt_L12M',
     'SuitSt_L13M',
     'SuitSt_L14M',
     'SuitSt_L15M',
     'SuitSt_L16M',
     'SuitSt_L17M',
     'SuitSt_L18M',
     'SuitSt_L19M',
     'SuitSt_L20M',
     'SuitSt_L21M',
     'SuitSt_L22M',
     'SuitSt_L23M',
     'SuitSt_L24M',
     'SuitSt_L25M',
     'SuitSt_L26M',
     'SuitSt_L27M',
     'SuitSt_L28M',
     'SuitSt_L29M',
     'SuitSt_L30M',
     'SuitSt_L31M',
     'SuitSt_L32M',
     'SuitSt_L33M',
     'SuitSt_L34M',
     'SuitSt_L35M',
     'SuitSt_L36M',
     'SuitSt_L37M',
     'SuitSt_L38M',
     'SuitSt_L39M',
     'SuitSt_L40M',
     'SuitSt_L41M',
     'SuitSt_L42M',
     'SuitSt_L43M',
     'SuitSt_L44M',
     'SuitSt_L45M',
     'SuitSt_L46M',
     'SuitSt_L47M',
     'SuitSt_L48M',
     'AcctSt_L0M',
     'AcctSt_L1M',
     'AcctSt_L2M',
     'AcctSt_L3M',
     'AcctSt_L4M',
     'AcctSt_L5M',
     'AcctSt_L6M',
     'AcctSt_L7M',
     'AcctSt_L8M',
     'AcctSt_L9M',
     'AcctSt_L10M',
     'AcctSt_L11M',
     'AcctSt_L12M',
     'AcctSt_L13M',
     'AcctSt_L14M',
     'AcctSt_L15M',
     'AcctSt_L16M',
     'AcctSt_L17M',
     'AcctSt_L18M',
     'AcctSt_L19M',
     'AcctSt_L20M',
     'AcctSt_L21M',
     'AcctSt_L22M',
     'AcctSt_L23M',
     'AcctSt_L24M',
     'AcctSt_L25M',
     'AcctSt_L26M',
     'AcctSt_L27M',
     'AcctSt_L28M',
     'AcctSt_L29M',
     'AcctSt_L30M',
     'AcctSt_L31M',
     'AcctSt_L32M',
     'AcctSt_L33M',
     'AcctSt_L34M',
     'AcctSt_L35M',
     'AcctSt_L36M',
     'AcctSt_L37M',
     'AcctSt_L38M',
     'AcctSt_L39M',
     'AcctSt_L40M',
     'AcctSt_L41M',
     'AcctSt_L42M',
     'AcctSt_L43M',
     'AcctSt_L44M',
     'AcctSt_L45M',
     'AcctSt_L46M',
     'AcctSt_L47M',
     'AcctSt_L48M',
     'AssetSt_L0M',
     'AssetSt_L1M',
     'AssetSt_L2M',
     'AssetSt_L3M',
     'AssetSt_L4M',
     'AssetSt_L5M',
     'AssetSt_L6M',
     'AssetSt_L7M',
     'AssetSt_L8M',
     'AssetSt_L9M',
     'AssetSt_L10M',
     'AssetSt_L11M',
     'AssetSt_L12M',
     'AssetSt_L13M',
     'AssetSt_L14M',
     'AssetSt_L15M',
     'AssetSt_L16M',
     'AssetSt_L17M',
     'AssetSt_L18M',
     'AssetSt_L19M',
     'AssetSt_L20M',
     'AssetSt_L21M',
     'AssetSt_L22M',
     'AssetSt_L23M',
     'AssetSt_L24M',
     'AssetSt_L25M',
     'AssetSt_L26M',
     'AssetSt_L27M',
     'AssetSt_L28M',
     'AssetSt_L29M',
     'AssetSt_L30M',
     'AssetSt_L31M',
     'AssetSt_L32M',
     'AssetSt_L33M',
     'AssetSt_L34M',
     'AssetSt_L35M',
     'AssetSt_L36M',
     'AssetSt_L37M',
     'AssetSt_L38M',
     'AssetSt_L39M',
     'AssetSt_L40M',
     'AssetSt_L41M',
     'AssetSt_L42M',
     'AssetSt_L43M',
     'AssetSt_L44M',
     'AssetSt_L45M',
     'AssetSt_L46M',
     'AssetSt_L47M',
     'AssetSt_L48M',
     'Perf_Avail',
     'Derog_Pol',
     'Act_Acct_Pol',
     'DPD90_Pol_L12M',
     'DPD60_Pol_L12M',
     'DPD30_Pol_L6M',
     'DPD30_Pol_L12M',
     'DPD30_Pol_l12m_cnt',
     'DPD30_Pol_l6m_cnt',
     'DPD30_Pol_l3m_cnt',
     'Sco_Pol',
     'INQ_Flag',
     'Recent',
     'CollateralType',
     'CollateralValue',
     'LastName',
     'CustomFields']
    
    columns_df_bureau = [col for col in df_Bur.columns]
    columns_in_feature_schema = [col for col in columns_df_bureau if col in feature_schema]
    columns_not_in_feature_schema = [col for col in feature_schema if col not in columns_df_bureau]
    
    df_bureau_following_schema = df_Bur[columns_in_feature_schema].copy()
    
    # Concatenate the new columns with the original DataFrame
    new_columns_df = pd.DataFrame(np.nan, index=df_bureau_following_schema.index, columns=columns_not_in_feature_schema)
    df_bureau_following_schema = pd.concat([df_bureau_following_schema, new_columns_df], axis=1)
    
    df_bureau_following_schema.columns = [col.lower() for col in df_bureau_following_schema.columns]
    df_bureau_following_schema.replace({np.nan: None},inplace=True)
    
    return df_bureau_following_schema

def applying_feature_schema_enquiry_data(df_Enq):
    
    if df_Enq is not None:
        feature_schema = ['seq',
         'Institution',
         'Date',
         'Time',
         'RequestPurpose',
         'Amount',
         'Date_N',
         'INQ_Flag']

        columns_df_enquiry = [col for col in df_Enq.columns]
        columns_in_feature_schema = [col for col in columns_df_enquiry if col in feature_schema]
        columns_not_in_feature_schema = [col for col in feature_schema if col not in columns_df_enquiry]

        df_enquiry_following_schema = df_Enq[columns_in_feature_schema].copy()

        # Concatenate the new columns with the original DataFrame
        new_columns_df = pd.DataFrame(np.nan, index=df_enquiry_following_schema.index, columns=columns_not_in_feature_schema)
        df_enquiry_following_schema = pd.concat([df_enquiry_following_schema, new_columns_df], axis=1)

        df_enquiry_following_schema.columns = [col.lower() for col in df_enquiry_following_schema.columns]
        df_enquiry_following_schema.replace({np.nan: None},inplace=True)
    
        return df_enquiry_following_schema
    else:
        return None

def applying_feature_schema_user_aggregated_policy_data(df_User):
    
    if df_User is not None:
        feature_schema = ['REPORTNUMBER',
             'Sum_Perf_Avail',
             'Cnt_Derog_Accts',
             'Cnt_OverDue_Accts',
             'Cnt_DPD90_L12M_Accts',
             'Cnt_DPD60_L12M_Accts',
             'Cnt_DPD30_L6M_Accts',
             'Cnt_DPD30_L12M_Accts',
             'Sum_DPD_Ins_L12M',
             'Sum_DPD_Ins_L6M',
             'Sum_DPD_Ins_L3M',
             'Max_Inq_L3M',
             'Max_Score',
             'Cnt_Accts',
             'Sum_Perf_Avail_Dec',
             'Cnt_Derog_Accts_Dec',
             'Cnt_OverDue_Accts_Dec',
             'Cnt_DPD90_L12M_Accts_Dec',
             'Cnt_DPD60_L12M_Accts_Dec',
             'Cnt_DPD30_L6M_Accts_Dec',
             'Cnt_DPD30_L12M_Accts_Rt_Dec',
             'Sum_DPD_Ins_L12M_Dec',
             'Sum_DPD_Ins_L6M_Dec',
             'Sum_DPD_Ins_L3M_Dec',
             'Max_Inq_L3M_Dec',
             'Max_Score_Dec']

        columns_df_aggregated_policies = [col for col in df_User.columns]
        columns_in_feature_schema = [col for col in columns_df_aggregated_policies if col in feature_schema]
        columns_not_in_feature_schema = [col for col in feature_schema if col not in columns_df_aggregated_policies]

        df_aggreagated_policies_following_schema = df_User[columns_in_feature_schema].copy()

        # Concatenate the new columns with the original DataFrame
        new_columns_df = pd.DataFrame(np.nan, index=df_aggreagated_policies_following_schema.index, columns=columns_not_in_feature_schema)
        df_aggreagated_policies_following_schema = pd.concat([df_aggreagated_policies_following_schema, new_columns_df], axis=1)

        df_aggreagated_policies_following_schema.columns = [col.lower() for col in df_aggreagated_policies_following_schema.columns]
        df_aggreagated_policies_following_schema.replace({np.nan: None},inplace=True)
        
        return df_aggreagated_policies_following_schema
    else:
        return None

# Comment this out in VM
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

    ## Comment this out in VM
    ## Extracting the json file in the form of dictionary
    data_dict = request.get_json()
    
#     ## Comment in cloud function
#     data_dict = request
    
    # Changing the format of the ftpl dataset
    ftpl_data_scylla = data_dict["ftpl_features"]

    ftpl_features_list = []
    for key, values in ftpl_data_scylla.items():
        ftpl_features_list = ftpl_features_list+values

    ftpl_data = {}

    for dict_feature_label in ftpl_features_list:
        ftpl_data[dict_feature_label['label']]=float(dict_feature_label['value'])
        
    ## Reading the user_id variable
    meesho_user_id = data_dict["user_id"]

    # Separating the datasets
    json_data = data_dict["bureau_data"]
    employee_data = data_dict["employment_details"]
    
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
        
    ## Checking NTC condition of new user
    if (Bur_Hist_Flag == 'ETC') :
        Bur_Hist_Flag, NTC_Type  = NTC_Cond_2(df_Bur)
    else:
        NTC_Type = 'NTC_No_Accts'
    
    ## Getting the Score of the User
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Score_Bur(json_data, df_Bur)
    
    ## Creating the Summary of Enquiry Data
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Inq_data(json_data,df_Bur)
    
    ## Getting the Enquiry Data
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Enq = Enq_df(json_data)
    else:
        df_Enq = None
    
    ## Changing code to integer for delinquency buckets, suit files status and assest class variables
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = var_treatment(df_Bur)
    
    ## Changing code to integer delinquency buckets for Days Past Due Variable
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Feat_Treat_DPD(df_Bur)
    
    ## Creating utility variables and changing varaible format
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Data_Cleaning(df_Bur)
    
    ## Changing the frame of reference to as of today for the grid features
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Last_Month_Feat(df_Bur)
    
    ## Creating account level policy features
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Policy_Feat(df_Bur)
    
    ## Creating enquiry count feature as policy metric
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur = Enq_Cnt(df_Bur, df_Enq)
    
    ## Creating user level policy features
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_User, Accpt_Ind = Bur_Pol_Agg(df_Bur)
    elif Bur_Hist_Flag == 'NTC':
        df_User = None
        Accpt_Ind = True
    else:
        df_User = None
        Accpt_Ind = None
        
    ## Appending meesho user id to the Bureau Dataset
    df_Bur["meesho_user_id"] = meesho_user_id
        
    ## Limit Calculation
    bnpl_limit_bre = limit_calculation(employee_data,ftpl_data)
    
    ## Converting features to format which can be stored as json
    if (Bur_Hist_Flag == 'ETC') | (NTC_Type == 'NTC_New_User'):
        df_Bur, df_Enq = json_serialize(df_Bur, df_Enq)
        
    ## Making consisitent schema for the processed bureau data 
    df_Bur = applying_feature_schema_bureau_data(df_Bur)
    df_Enq = applying_feature_schema_enquiry_data(df_Enq)
    df_User = applying_feature_schema_user_aggregated_policy_data(df_User)
    
    ## Converting pandas dataframe to dictionary
    df_Bur_Dict = df_Bur.to_dict('dict')
    type_df_Bur_Dict = type(df_Bur_Dict)
    
    ## Converting enquiry dataframe to dictionary
    if df_Enq is not None:
        df_Enq_Dict = df_Enq.to_dict('dict')
        type_df_Enq_Dict = type(df_Enq_Dict)
    else:
        df_Enq_Dict = None
    
    ## Converting user level policy aggregated features to json
    if df_User is not None:
        df_User_Dict = df_User.to_dict('dict')
    else:
        df_User_Dict = None
    
    ## Changing accept indicator format from boolean to string
    if Accpt_Ind is not None:
        Str_Accpt_Ind = str(Accpt_Ind)
    else:
        Str_Accpt_Ind = None
        
    ## Creating JSON Output
    df_Bur_Out = {
        'accept_ind' : Str_Accpt_Ind,
        'bureau_history_status' : Bur_Hist_Flag,
        'bnpl_limit' : bnpl_limit_bre,
        'analytics_data' : {
            'bur_processed_bnpl_data' : df_Bur_Dict,
            'enquiry_bnpl_data' : df_Enq_Dict,
            'user_aggregated_policy_data' : df_User_Dict}
    }
    
#     ## Comment this out in VM
#     # Define your GCS bucket and folder path
#     bucket_name = 'fs_user_bre_ml'
#     gcs_folder_path = 'BRE_Setup_Testing/API_Response/'

#     # Initialize the GCS client
#     storage_client = storage.Client()

#     # Store the JSON to GCS
#     file_name = f"{gcs_folder_path}processed_file_{uploaded_file.filename}"
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(file_name)
            
#     # Upload the JSON as a string to GCS
#     blob.upload_from_string(json.dumps(df_Bur_Out), content_type='application/json')

#     # return f"The file content data type is {json_data_type} and file content is {json_data}"
    return df_Bur_Out, 200