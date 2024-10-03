import gradio as gr
import pandas as pd
import numpy as np
import tempfile 
from io import StringIO
from huggingface_hub import HfApi, Repository
import os
from datetime import datetime
from tqdm import tqdm

pd.set_option("future.no_silent_downcasting", True)

api = HfApi()

def upload_file_to_hf(df,final_name,repo_id='laurentperrierus/lp_nielsen_data_cleansing'):

    HF_TOKEN = os.getenv('HF_TOKEN')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Unidentified_UPCS_name_changes_proposition_")
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    api.upload_file(
        path_or_fileobj=temp_file.name,
        path_in_repo=final_name,
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN
    )

def update_the_truth(truth, 
                     add_to_truth_df_state, 
                     upcs_to_modify_df_state, 
                     upcs_to_delete_df_state):
    
    # Delete UPCs that must be removed from truth
    if upcs_to_delete_df_state is not None:
        lst_of_upcs_to_delete = upcs_to_delete_df_state['UPC'].unique().tolist()
        truth_minus_deleted = truth[~truth['wine_upc'].isin(lst_of_upcs_to_delete)]
    else:
        truth_minus_deleted = truth.copy()

    # Modify the UPCs to modify
    if upcs_to_modify_df_state is not None:
        lst_of_upcs_to_modify = upcs_to_modify_df_state['UPC'].unique().tolist()
        truth_modified = truth_minus_deleted [~truth_minus_deleted ['wine_upc'].isin(lst_of_upcs_to_modify)]
        upcs_to_modify_df = upcs_to_modify_df_state.rename(columns={'UPC':'wine_upc',
                                                    'Item Names':'item_desc_truth',
                                                     'Brand Families':'brand_family_truth',
                                                     'Sizes':'size_truth',
                                                     'Colors':'color_truth',
                                                     'Prestige':'prestige_truth',
                                                     'ARP':'arp_truth',
                                                     'ARP LY':'arp_ly_truth',
                                                     'Units LY':'Units YA',
                                                     '%ACV Reach Where Dist NON ALCOHOLIC':'Max Item %ACV Reach Where Dist WINE',
                                                     '%ACV Reach Where Dist YA NON ALCOHOLIC': 'Max Item %ACV Reach YA Where Dist WINE',
                                                     '$ Per Point of ACV':'$ Per Point of Max Item ACV'
                                                   })
        upcs_to_modify_df_int = upcs_to_modify_df.drop(columns=['# Stores Last Period', '# Stores Selling Last Period','$ per Store Selling Last Period', 'ACV Calc', 'GT'])
        truth_modified_final = pd.concat([truth_modified, upcs_to_modify_df_int], ignore_index=True)
    else:
        truth_modified_final = truth_minus_deleted.copy()

    # Add the newly identified UPCs to the truth
    if add_to_truth_df_state is not None:
        add_to_truth_df_state.columns = add_to_truth_df_state.columns.str.strip()
        truth_modified_final.columns = truth_modified_final.columns.str.strip()
        df_to_concat_int = add_to_truth_df_state.rename(columns={'UPC':'wine_upc',
                                                        'Item Names':'item_desc_truth',
                                                        'Brand Families':'brand_family_truth',
                                                        'Sizes':'size_truth',
                                                        'Colors':'color_truth',
                                                        'Prestige':'prestige_truth',
                                                        'ARP':'arp_truth',
                                                        'ARP LY':'arp_ly_truth',
                                                        'Units LY':'Units YA',
                                                        '%ACV Reach Where Dist NON ALCOHOLIC':'Max Item %ACV Reach Where Dist WINE',
                                                        '%ACV Reach Where Dist YA NON ALCOHOLIC': 'Max Item %ACV Reach YA Where Dist WINE',
                                                        '$ Per Point of ACV':'$ Per Point of Max Item ACV'
                                                    })
        df_to_concat_final = df_to_concat_int.drop(columns=['# Stores Last Period', '# Stores Selling Last Period','$ per Store Selling Last Period', 'ACV Calc', 'GT'])
        new_truth = pd.concat([truth_modified_final,df_to_concat_final],ignore_index=True)
    else:
        new_truth = truth_modified_final.copy()
    new_truth_final = return_new_truth_in_right_format(new_truth)
    now = datetime.now()
    formatted_datetime = now.strftime("%m_%d_%Y_%H_%M_%S")
    name_of_old_truth = str('truth_' + str(formatted_datetime))
    upload_file_to_hf(truth,name_of_old_truth)
    upload_file_to_hf(new_truth_final,'current_truth.csv')

def change_red_to_white(row, colors='Colors'):
    
    color = row[colors].strip()

    if color == 'RED':
        new_color = 'WHITE'
    else:
        new_color = color
    
    return new_color

def return_new_truth_in_right_format(df):

    def turn_back_arp_to_dolls(row):

        if '$$' in str(row):
            row_str = str(row)
            row_dolls = row_str.split('$',1)[1]
        elif '$' in str(row):
            row_dolls = str(row)
        elif str(row).lower() !='nan':
            row_dolls = '$' + str(row)
        else:
            row_dolls =''
    
        return row_dolls

    df.columns = df.columns.str.strip()
    df_copy = df.copy()
    df_copy['arp_truth'] = df['arp_truth'].astype(str)
    df_copy['arp_ly_truth'] = df['arp_ly_truth'].astype(str)
    df_copy.loc[:,'arp_truth'] = df['arp_truth'].apply(turn_back_arp_to_dolls)
    df_copy.loc[:,'arp_ly_truth'] = df['arp_ly_truth'].apply(turn_back_arp_to_dolls)

    def turn_to_int(x):
        if pd.isna(x):
            return ''
        else:
            return int(float(x))
    
    def format_percentage(x):
        if pd.isna(x):
            return ''
        elif isinstance(x, str) and '%' in x:
            return x
        else:
            x_float = float(x)
            return '{:.2f}%'.format(x_float * 100)
    
    df_copy['wine_upc'] = df['wine_upc'].apply(lambda x: turn_to_int(x))
    df_copy['# Stores'] = df['# Stores'].apply(lambda x: turn_to_int(x))
    df_copy['# Stores Selling'] = df['# Stores Selling'].apply(lambda x: turn_to_int(x))
    df_copy['$'] = df['$'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['$ YA'] = df['$ YA'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['9 Liter Equivs'] = df['9 Liter Equivs'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['9 Liter Equivs YA'] = df['9 Liter Equivs YA'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['Units'] = df['Units'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['Units YA'] = df['Units YA'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['$ per Store Selling'] = df['$ per Store Selling'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['$ Per Point of Max Item ACV'] = df['$ Per Point of Max Item ACV'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['Max Item %ACV Reach Where Dist WINE'] = df_copy['Max Item %ACV Reach Where Dist WINE'].apply(lambda x: format_percentage(x))
    df_copy['Max Item %ACV Reach YA Where Dist WINE'] = df_copy['Max Item %ACV Reach YA Where Dist WINE'].apply(lambda x: format_percentage(x))

    df_final = df_copy.fillna('')
    
    return df_final
   
def identify_known_upcs(file):

    # Define truth and input dataframes
    truth = truth_data_prep(pd.read_csv('current_truth.csv'))
    input_df = pd.read_csv(process_csv(file))
    input_df.columns = input_df.columns.str.strip()
    
    # PREPARE INPUT DF
    input_df_beautiful = input_df.copy()
    input_df_beautiful_int = data_prep_for_all_input_dfs(input_df_beautiful,additional_column=None,check=True)
    input_df_beautiful_final = return_df_in_right_format(input_df_beautiful_int)
    
    # Prepare input data
    input_identified_upcs, input_unidentified_upcs = split_df_into_knwon_and_unknown_upcs(input_df_beautiful_int, truth)

    # Compute number of rows with identified UPCs
    num_of_identified_rows = input_identified_upcs.shape[0]
    num_of_identified_upcs = len(input_identified_upcs['wine_upc'].unique().tolist())
    
    # Correct rows of identified UPCs
    corrected_identified_upcs = return_identified_upcs_df(input_identified_upcs,truth)
    corrected_identified_upcs_int = corrected_identified_upcs.copy()
    corrected_identified_upcs_int.rename(columns={'wine_upc':'UPC',
                                                    'item_desc_input':'Item Names',
                                                    'prestige_input':'Prestige',
                                                    'brand_family_input':'Brand Families',
                                                    'color_input':'Colors',
                                                    'size_input':'Sizes'},inplace=True
                                                    )
    corrected_identified_upcs_final = return_df_in_right_format(corrected_identified_upcs_int)
    corrected_identified_upcs_unique = corrected_identified_upcs_final.drop_duplicates(subset=['UPC'])
    temp_file_11 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Identified_UPCs_from_truth_")
    corrected_identified_upcs_unique_for_csv = corrected_identified_upcs_unique.copy()
    corrected_identified_upcs_unique_for_csv['Modification Status'] = ''
    corrected_identified_upcs_unique_for_csv = corrected_identified_upcs_unique_for_csv[['Modification Status'] + corrected_identified_upcs_unique.columns.tolist()]
    corrected_identified_upcs_unique_for_csv.to_csv(temp_file_11.name, index=False)

    upc_message = f"""# Identified UPCs
    
    We have already seen the UPCs of **{num_of_identified_rows} rows**: this corresponds to **{num_of_identified_upcs} UPCs**.
            
    Please review the corresponding rows that have been fully filled using the truth.
            
    If you decide to modify information for some of these UPCs and submit your own correction:
        - Do not forget to **change the format of the "UPC" column to integer** as soon as you open the CSV.
        - For each UPC where you make a change, please indicate the following in the 'Modification Status' column:
            - Use "modified" if you want to modify the UPC
            - Use "deleted" if you delete the UPC from the truth. This means the information currently available is insufficient to correctly identify the wine associated with that UPC.
        
    The truth will be updated accordingly."""
            
    return (
        input_df_beautiful_final, # input_csv_df_state
        gr.update(value=upc_message, visible=True), # upc_message
        corrected_identified_upcs_final, # known_upcs_df_state
        gr.update(value=corrected_identified_upcs_unique, visible=True), # identified_upcs_df
        input_unidentified_upcs, # unidentified_names_df_state
        gr.update(visible=True), # proceed_with_this_truth_button
        gr.update(visible=True), # modify_the_truth_button
        gr.update(interactive=False), # submit_input_csv_button
        gr.update(visible=True, value=temp_file_11.name))  # identified_upcs_csv


def proceed_with_identified_upcs_fn(unidentified_names_df_state):

    # Try to match unknown UPC names with known names
    corrected_unknown_upcs = identify_name_of_unknown_upc(unidentified_names_df_state)
    corrected_unknown_upcs.rename(columns={'wine_upc':'Nielsen Item Names ID',
                                            'identified_name':'Suggested name modification',
                                            'item_desc_input':'Nielsen Item Names',
                                            'prestige_input':'Prestige',
                                            'brand_family_input':'Brand Families',
                                            'color_input':'Colors',
                                            'size_input':'Sizes'},inplace=True
                                )

    identified_names_df = corrected_unknown_upcs[corrected_unknown_upcs['Suggested name modification']!='']
    unidentified_names_df = corrected_unknown_upcs[corrected_unknown_upcs['Suggested name modification']=='']
    unidentified_names_df.to_csv('unidentified_names_df.csv',index=False)
    
    if identified_names_df.shape[0] > 0:
        identified_names_df_final = return_df_in_right_format(identified_names_df)
        identified_names_df_styled = highlight_name_changes(identified_names_df_final)
        
        unidentified_names_df_int = unidentified_names_df.copy()
        unidentified_names_df_int.drop(columns=['Suggested name modification'],inplace=True)
        unidentified_names_df_final = return_df_in_right_format(unidentified_names_df_int)
        
        temp_file_2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Unidentified_UPCS_name_changes_proposition_")
        identified_names_df_final.to_csv(temp_file_2.name, index=False)
        
        name_identification_message = f"""
        # Name modifications for unknown UPCs

        Among the rows with unknown UPCs, we propose the following **{identified_names_df_final.shape[0]} name modifications** in the column 'Suggested name modification'
        - If all modifications look correct, click on 'Accept all suggested modifications' - these rows will be added to the truth and used in future tasks as the ground truth for these UPCS.
        - If not, download the output and do the following:
            - Change the names that you want to change in the 'Suggested name modification' column;
            - Delete the rows that you don't want to add to the ground truth;
            - Do not worry about changing other columns in the dataframe if they are incorrect, the algorithm will correct them using the updated names.
        """
    
    return (
        gr.update(value=name_identification_message, visible=True), # name_identification_markdown
        gr.update(value=identified_names_df_styled, visible=True), # name_modifications_proposition_df
        identified_names_df_final, # identified_names_df_state
        unidentified_names_df_final, # unidentified_names_df_state
        gr.update(value=temp_file_2.name, visible=True), # name_modifications_proposition_csv
        gr.update(interactive=False), # proceed_with_this_truth_button
        gr.update(interactive=False), # modify_the_truth_button
        gr.update(visible=True), # accept_all_changes_button
        gr.update(visible=True), # make_modifications_to_the_names_button
        gr.update(visible=True) # make_no_modifications_to_names_button
    )

def process_csv(file):

    df = pd.read_csv(file.name)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    
    df.to_csv(temp_file.name, index=False)
    
    return temp_file.name

def truth_data_prep(truth):

    truth.replace('#DIV/0!', np.nan, inplace=True)
    truth.columns = truth.columns.str.strip()

    return truth

def data_prep_for_all_input_dfs(input_df,additional_column:str=None,check=True, raw=True, colors='Colors'):

    input_df.columns = input_df.columns.str.strip()


    if check:
        if raw:
            required_columns = {'Nielsen Item Names', 'Nielsen Item Names ID', 'Brand Families',
                                'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA',
                                '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
                                '%ACV Reach Where Dist NON ALCOHOLIC',
                                '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
                                '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
                                '# Stores Last Period', '# Stores Selling Last Period',
                                '$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY',
                                'ACV Calc'}
            item_id = 'Nielsen Item Names ID'
        
        else:
            required_columns = {'Item Names', 'UPC', 'Brand Families',
                                'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA',
                                '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
                                '%ACV Reach Where Dist NON ALCOHOLIC',
                                '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
                                '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
                                '# Stores Last Period', '# Stores Selling Last Period',
                                '$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY',
                                'ACV Calc'}
            item_id = 'UPC'

        if additional_column:
            required_columns.add(additional_column)

        if not (len(input_df.columns) == len(required_columns) and set(input_df.columns) == required_columns):
            if len(required_columns)==26:
                error_message = f'The columns of your CSV do not match the usual column scheme. Your CSV should have 26 columns. Check that you did not change the column names.'
                print(f"The df's columns are {input_df.columns.tolist()}, there are {input_df.shape[1]} columns")
            elif len(required_columns) == 27:
                error_message = f'The columns of your CSV do not match the usual column scheme. Your CSV should have 27 columns: make sure you did not delete the "{additional_column}" column. Check that you did not change the column names.'
                print(f"The df's columns are {input_df.columns.tolist()}, there are {input_df.shape[1]} columns")
            raise gr.Error(error_message)

        if input_df[item_id].dtype!='int64':
            error_message_upc = 'The Nielsen Item Names ID column is in the wrong format, it must be integers i.e. numbers with no decimals.'
            raise gr.Error(error_message_upc)

        def check_columns(input_df):
            # Check that 'GT' column has no NaN values
            if input_df['GT'].isna().any():
                error_message_gt = 'The GT column must not have empty cells. Please check the column and try again.'
                raise gr.Error(error_message_gt)
            
            # Check that each row has at least one non-NaN value in 'ARP', 'ARP LY', 'ACV Calc'
            if not input_df[['ARP', 'ARP LY', 'ACV Calc']].notna().any(axis=1).all():
                error_message_arp = 'Each row must have at least one non-empty value in ARP, ARP LY, or ACV Calc.'
                raise gr.Error(error_message_arp)
        
        check_columns(input_df)
    
    input_df = input_df.replace('#DIV/0!', np.nan)
    input_df = input_df.replace('', np.nan)
    input_df = input_df.replace(' ', np.nan)

    # Taking care of ARP columns if not floats
    def transform_dolls_into_float(dolls):
        try:
            if str(dolls) != 'nan':
                if '$' in str(dolls):
                    dolls_changed = float(str(dolls).strip().split('$')[1])
                else:
                    dolls_changed = float(str(dolls))
            elif str(dolls)=='nan':
                dolls_changed = np.nan
            return dolls_changed
        except ValueError:
            if check:
                error_message_arp = 'The ARP and ARP LY columns must be numbers (unseparated number format: 100000.00 is okay, 100,000.00 is not), not currencies.'
                raise gr.Error(error_message_arp)
            else:
                pass
    
    def transform_percent_to_float(percent):
        if '%' in str(percent):
            percent_str = str(percent)
            float_res = float(float(percent_str.strip().split('%')[0]) / 100)
            return float_res
        else:
            return str(percent)
    
    if input_df['%ACV Reach Where Dist NON ALCOHOLIC'].dtype == 'object':
        input_df.loc[:,'%ACV Reach Where Dist NON ALCOHOLIC'] = input_df['%ACV Reach Where Dist NON ALCOHOLIC'].apply(transform_percent_to_float)
        input_df['%ACV Reach Where Dist NON ALCOHOLIC'] = input_df['%ACV Reach Where Dist NON ALCOHOLIC'].astype(float)
    
    if input_df['%ACV Reach Where Dist YA NON ALCOHOLIC'].dtype == 'object':
        input_df.loc[:,'%ACV Reach Where Dist YA NON ALCOHOLIC'] = input_df['%ACV Reach Where Dist YA NON ALCOHOLIC'].apply(transform_percent_to_float)
        input_df['%ACV Reach Where Dist YA NON ALCOHOLIC'] = input_df['%ACV Reach Where Dist YA NON ALCOHOLIC'].astype(float)

    #!TODO apply only one function

    if input_df['ARP'].dtype == 'object':
        input_df.loc[:,'ARP'] = input_df['ARP'].apply(transform_dolls_into_float)
        input_df['ARP'] = input_df['ARP'].astype(float)

    if input_df['ARP LY'].dtype == 'object':
        input_df.loc[:,'ARP LY'] = input_df['ARP LY'].apply(transform_dolls_into_float)
        input_df['ARP LY'] = input_df['ARP LY'].astype(float)

    input_df_final = input_df.copy()
    input_df_final[colors] = input_df.apply(change_red_to_white,axis=1, colors=colors)
    

    return input_df_final

def split_df_into_knwon_and_unknown_upcs(input_df, truth):

    input_df = data_prep_for_all_input_dfs(input_df,additional_column=None)

    input_df.rename(columns={'Nielsen Item Names ID':'wine_upc',
                            'Nielsen Item Names':'item_desc_input',
                            'Prestige':'prestige_input',
                            'Brand Families':'brand_family_input',
                            'Colors':'color_input',
                            'Sizes':'size_input'},
                            inplace=True)
    input_df['upc_is_in_truth'] = input_df['wine_upc'].isin(truth['wine_upc'])

    input_df_identified = input_df[input_df['upc_is_in_truth']==True].copy()
    input_df_unknown = input_df[input_df['upc_is_in_truth']==False].copy()

    input_df_identified.drop(columns=['upc_is_in_truth'],inplace=True)
    input_df_unknown.drop(columns=['upc_is_in_truth'],inplace=True)
    
    return input_df_identified, input_df_unknown

def identify_name_of_unknown_upc(df):
    
    num_of_names_changed = 0

    def name_change(row):
        nonlocal num_of_names_changed

        desc_input = str(row['item_desc_input']).strip().replace('  ', ' ')
        color_input = str(row['color_input']).strip().lower()
        size_input = str(row['size_input']).strip().lower()
        arp_input = row['ARP']
        arp_ly_input = row['ARP LY']
        
        ## MOET & CHANDON
        
        if 'MOET & CHANDON' in desc_input and any(substring in desc_input for substring in ['VINTAGE', 'VNTG']) and size_input == '750ml' and color_input == 'pink':
            row['identified_name'] = 'MOET & CHANDON VINTAGE ROSE 750ML'            
            num_of_names_changed +=1

        elif 'MOET & CHANDON' in desc_input and any(substring in desc_input for substring in ['IMPRL', 'IMPERIAL']) and 'ICE' not in desc_input and size_input == '750ml' and color_input=='white':
            row['identified_name'] = 'MOET & CHANDON IMPERIAL 750ML'            
            num_of_names_changed +=1

        elif any(substring in desc_input for substring in ['NCTR', 'NECTAR']) and 'MOET & CHANDON' in desc_input and color_input == 'pink' and size_input == '750ml':
            row['identified_name'] = 'MOET & CHANDON NECTAR ROSE 750ML'
            num_of_names_changed +=1
        
        elif any(substring in desc_input for substring in ['NCTR', 'NECTAR']) and 'MOET & CHANDON' in desc_input and color_input == 'white' and size_input == '750ml':
            row['identified_name'] = 'MOET & CHANDON NECTAR 750ML'
            num_of_names_changed +=1
        
        elif any(substring in desc_input for substring in ['NCTR', 'NECTAR']) and 'MOET & CHANDON' in desc_input and color_input == 'white' and size_input == '1.5l':
            row['identified_name'] = 'MOET & CHANDON NECTAR 1.5L' 
            num_of_names_changed +=1
        
        elif any(substring in desc_input for substring in ['NCTR', 'NECTAR']) and 'MOET & CHANDON' in desc_input and color_input == 'white' and size_input == '187ml':
            row['identified_name'] = 'MOET & CHANDON NECTAR 187ML'
            num_of_names_changed +=1
        
        elif not any(substring in desc_input for substring in ['NCTR', 'NECTAR', 'VINTAGE', 'VNTG', 'ICE']) and 'MOET & CHANDON' in desc_input and color_input == 'pink' and size_input == '750ml':
            row['identified_name'] = 'MOET & CHANDON ROSE 750ML'
            num_of_names_changed +=1
    
        elif 'ICE' in desc_input and 'MOET & CHANDON' in desc_input and color_input == 'white' and size_input == '750ml' and not any(substring in desc_input for substring in ['IMPRL', 'IMPERIAL']) :
            row['identified_name'] = 'MOET & CHANDON ICE 750ML'
            num_of_names_changed +=1
    
        elif 'ICE' in desc_input and 'MOET & CHANDON' in desc_input and color_input == 'pink' and size_input == '750ml':
            row['identified_name'] = 'MOET & CHANDON ICE ROSE 750ML'
            num_of_names_changed +=1
        
        ## TAITTINGER

        elif 'TAITTINGER LA FRANCAISE' in desc_input and size_input == '750ml' and color_input=='white':
            row['identified_name'] = 'TAITTINGER LA FRANCAISE 750ML'    
            num_of_names_changed +=1
        
        ## LOUIS ROEDERER

        elif 'LOUIS ROEDERER' in desc_input and size_input == '750ml' and (arp_input >= 300 or arp_ly_input >= 300) and color_input=='white':
            row['identified_name'] = 'LOUIS ROEDERER CRISTAL 750ML'    
            num_of_names_changed +=1
        
        elif 'LOUIS ROEDERER' in desc_input and size_input == '750ml' and color_input == 'white' and (arp_input<80  or arp_ly_input<80):
            row['identified_name'] = 'LOUIS ROEDERER CHAMPAGNE 750ML'
            num_of_names_changed +=1

        elif 'LOUIS ROEDERER' in desc_input and size_input == '750ml' and color_input == 'white' and (80<=arp_input<=120  or 80<=arp_ly_input<=120):
            row['identified_name'] = 'LOUIS ROEDERER VINTAGE 750ML'
            num_of_names_changed +=1
        
        elif 'CRISTAL' in desc_input and 'LOUIS ROEDERER' in desc_input and 'ROSE' in desc_input:
            row['identified_name'] = 'LOUIS ROEDERER CRISTAL ROSE 750ML'
            num_of_names_changed +=1

        ## VEUVE CLICQUOT

        elif 'VEUVE CLICQUOT' in desc_input and size_input == '750ml' and (arp_input >= 150 or arp_ly_input >= 150) and color_input=='white':
            row['identified_name'] = 'VEUVE CLICQUOT LA GRANDE DAME 750ML'    
            num_of_names_changed +=1

        elif 'VEUVE CLICQUOT' in desc_input and size_input == '750ml' and (arp_input >= 300 or arp_ly_input >= 300) and color_input=='pink':
            row['identified_name'] = 'VEUVE CLICQUOT LA GRANDE DAME ROSE 750ML'            
            num_of_names_changed +=1
        
        elif 'VEUVE CLICQUOT' in desc_input and size_input == '750ml' and (100<=arp_input <= 250 or 100<=arp_ly_input <= 250) and color_input=='pink' and 'RICH' not in desc_input:
            row['identified_name'] = 'VEUVE CLICQUOT VINTAGE ROSE 750ML'
            num_of_names_changed +=1

        elif 'VEUVE CLICQUOT' in desc_input and 'RICH' in desc_input and size_input == '750ml' and color_input == 'white':
            row['identified_name'] = 'VEUVE CLICQUOT RICH 750ML'
            num_of_names_changed +=1

        elif 'VEUVE CLICQUOT' in desc_input and color_input == 'pink' and size_input == '750ml' and ((arp_input if not np.isnan(arp_input) else 0) <= 100 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) <= 100) and 'RICH' not in desc_input:
            row['identified_name'] = 'VEUVE CLICQUOT ROSE 750ML'
            num_of_names_changed +=1

        elif 'VEUVE CLICQUOT' in desc_input and color_input == 'pink' and size_input == '750ml' and ((arp_input if not np.isnan(arp_input) else 0) <= 100 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) <= 100) and 'RICH' in desc_input:
            row['identified_name'] = 'VEUVE CLICQUOT RICH ROSE 750ML'
            num_of_names_changed +=1
        
        elif 'VEUVE CLICQUOT' in desc_input and color_input == 'white' and size_input == '750ml' and 'DEMI SEC' in desc_input and ((arp_input if not np.isnan(arp_input) else 0) <= 200 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) <= 200):
            row['identified_name'] = 'VEUVE CLICQUOT DEMI SEC 750ML'
            num_of_names_changed +=1
        
        elif 'VEUVE CLICQUOT' in desc_input and color_input == 'white' and size_input == '750ml' and 'EX OLD' in desc_input and ((arp_input if not np.isnan(arp_input) else 0) <= 200 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) <= 200):
            row['identified_name'] = 'VEUVE CLICQUOT EXTRA OLD 750ML'
            num_of_names_changed +=1

        elif 'VEUVE CLICQUOT' in desc_input and color_input == 'white' and size_input == '750ml' and not any(substring in desc_input for substring in ['DEMI SEC', 'EX OLD','RICH']) and ((arp_input if not np.isnan(arp_input) else 0) <= 85 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) <= 85):
            row['identified_name'] = 'VEUVE CLICQUOT PONSARDIN 750ML'
            num_of_names_changed +=1
        
        elif 'VEUVE CLICQUOT' in desc_input and color_input == 'white' and size_input == '750ml' and not any(substring in desc_input for substring in ['DEMI SEC', 'EX OLD','RICH']) and ((arp_input if not np.isnan(arp_input) else 0) > 90 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) > 90):
            row['identified_name'] = 'VEUVE CLICQUOT VINTAGE 750ML'
            num_of_names_changed +=1
        
        ## PERRIER JOUET

        elif 'PERRIER JOUET GRAND BRUT' in desc_input and size_input == '750ml' and color_input=='white':
            row['identified_name'] = 'PERRIER JOUET GRAND BRUT 750ML'    
            num_of_names_changed +=1
        
        ## POL ROGER

        elif 'POL ROGER' in desc_input and size_input == '750ml' and color_input=='white' and (arp_input >= 280 or arp_ly_input >= 280):
            row['identified_name'] = 'POL ROGER WINSTON CHURCHILL 750ML'    
            num_of_names_changed +=1

        ## SALON

        elif 'SALON' in desc_input and size_input == '750ml' and color_input=='white' and (arp_input >= 300 or arp_ly_input >= 300):
            row['identified_name'] = 'SALON 750ML'
            num_of_names_changed +=1

        ## LAURENT-PERRIER
        
        elif ('ROSE' or 'PINK') in desc_input and 'LAURENT-PERRIER' in desc_input and size_input == '750ml' and ((arp_input if not np.isnan(arp_input) else 0) <= 140 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) <= 140):
            row['identified_name'] = 'LAURENT-PERRIER ROSE 750ML'
            num_of_names_changed +=1

        elif 'LAURENT-PERRIER' in desc_input and size_input == '750ml' and color_input=='white' and (400>arp_input >= 140 or 400>arp_ly_input >= 140):
            row['identified_name'] = 'LAURENT-PERRIER GRAND SIECLE 750ML'
            num_of_names_changed +=1

        elif 'LAURENT-PERRIER' in desc_input and size_input == '750ml' and color_input=='white' and (arp_input >400 or arp_ly_input >400):
            row['identified_name'] = 'LAURENT-PERRIER GRAND SIECLE REM SIZE'
            num_of_names_changed +=1

        elif 'LAURENT-PERRIER GR SIECLE' in desc_input and '1.5L' in desc_input and (arp_input >= 500  or arp_ly_input >= 500):
            row['identified_name'] = 'LAURENT-PERRIER GRAND SIECLE 1.5L'
            num_of_names_changed +=1
        
        ## KRUG

        elif 'KRUG' in desc_input and size_input == '750ml' and color_input == 'white' and (arp_input < 1000 or arp_ly_input < 1000):
            row['identified_name'] = 'KRUG GRANDE CUVEE 750ML'
            num_of_names_changed +=1

        elif 'KRUG' in desc_input and size_input == '750ml' and color_input == 'white' and (arp_input >= 1000 or arp_ly_input >= 1000):
            row['identified_name'] = 'KRUG CLOS DU MESNIL 750ML'            
            num_of_names_changed +=1

        ## DOM PERIGNON

        elif 'DOM PERIGNON' in desc_input and size_input == '750ml' and color_input == 'white':
            row['identified_name'] = 'DOM PERIGNON CHAMPAGNE 750ML'
            num_of_names_changed +=1

        elif desc_input in ['DOM PERIGNON SPK CHAMPAGNE ROSE PINK BRUT CHAMPAGNE FRANCE 750ML','DOM PERIGNON SPK CHAMPAGNE ROSE PINK CHAMPAGNE FRANCE 750ML']:
            row['identified_name'] = 'DOM PERIGNON ROSE 750ML'
            num_of_names_changed +=1
        
        ## BOLLINGER

        elif 'BOLLINGER' in desc_input and not any(substring in desc_input for substring in ['PN', 'B13']) and (arp_input >= 100 or arp_ly_input >= 100) and size_input == '750ml':
            row['identified_name'] = 'BOLLINGER GRANDE ANNEE 750ML'
            num_of_names_changed +=1

        elif 'BOLLINGER SPECIAL' in desc_input and '750ML' in desc_input and 'ROSE' not in desc_input:
            row['identified_name'] = 'BOLLINGER SPECIAL CHAMPAGNE 750ML'
            num_of_names_changed +=1

        ## PIPER-HEIDSIECK

        elif desc_input.split()[0] =='RARE' and size_input == '750ml':
            row['identified_name'] = 'PIPER-HEIDSIECK RARE 750ML'
            num_of_names_changed +=1
        
        elif 'PIPER-HEIDSIECK' in desc_input and color_input == 'white' and '375ML' in desc_input and 'RARE' not in desc_input:
            row['identified_name'] = 'PIPER-HEIDSIECK CHAMPAGNE 375ML'
            num_of_names_changed +=1
        
        elif desc_input in['PIPER-HEIDSIECK SPK CHAMPAGNE WHITE WHITE BRUT CHAMPAGNE FRANCE 750ML VAP','PIPER-HEIDSIECK SPK CHAMPAGNE WHITE WHITE BRUT CHAMPAGNE FRANCE 750ML','PIPER-HEIDSIECK SPK CHAMPAGNE WHITE WHITE CHAMPAGNE FRANCE 750ML'] and color_input == 'white' and size_input == '750ml':
            row['identified_name'] = 'PIPER-HEIDSIECK CHAMPAGNE 750ML'
            num_of_names_changed +=1
        
        ## POMMERY

        elif 'POMMERY' in desc_input and 'GRAND CRU' not in desc_input and size_input == '750ml' and color_input == 'white' and (arp_input<=60  or arp_ly_input<=60):
            row['identified_name'] = 'POMMERY CHAMPAGNE 750ML'            
            num_of_names_changed +=1
        
        ## BILLECART-SALMON

        elif 'BILLECART-SALMON' in desc_input and 'BLANC DE BLANC' not in desc_input and 'EXTRA BRUT' not in desc_input and 'DEMI SEC' not in desc_input and size_input == '750ml' and color_input == 'white' and (arp_input<=110  or arp_ly_input<=110):
            row['identified_name'] = 'BILLECART-SALMON CHAMPAGNE BRUT 750ML' 
            num_of_names_changed +=1
        
        elif 'BILLECART-SALMON' in desc_input and 'BLANC DE BLANC' in desc_input and size_input == '750ml' and (arp_input >= 190 or arp_ly_input >= 190) and color_input=='white':
            row['identified_name'] = 'BILLECART-SALMON LOUIS SALMON 750ML'
            num_of_names_changed +=1

        elif 'BILLECART-SALMON' in desc_input and 'BLANC DE BLANC' not in desc_input and size_input == '750ml' and (arp_input >= 190 or arp_ly_input >= 190) and color_input=='white':
            row['identified_name'] = 'BILLECART-SALMON CUVEE NICOLAS 750ML'
            num_of_names_changed +=1
    
        elif 'BILLECART-SALMON' in desc_input and 'BLANC DE BLANC' in desc_input and 'DEMI SEC' not in desc_input and size_input == '750ml' and color_input == 'white' and (130>=arp_input>=90  or 130>=arp_ly_input>=90):
            row['identified_name'] = 'BILLECART-SALMON BRUT BLANC DE BLANCS GRAND CRU 750ML'            
            num_of_names_changed +=1
        
        elif 'BILLECART-SALMON' in desc_input and 'EXTRA BRUT' in desc_input and 'DEMI SEC' not in desc_input and size_input == '750ml' and color_input == 'white' and (120>=arp_input>=60  or 120>=arp_ly_input>=60):
            row['identified_name'] = 'BILLECART-SALMON EXTRA BRUT 750ML'            
            num_of_names_changed +=1

        elif desc_input == 'BILLECART-SALMON SPK CHAMPAGNE ROSE PINK BRUT CHAMPAGNE FRANCE 750ML' and ((arp_input if not np.isnan(arp_input) else 0) <= 95 and (arp_ly_input if not np.isnan(arp_ly_input) else 0) <= 95) and size_input=='750ml':
            row['identified_name'] = 'BILLECART-SALMON ROSE 750ML'
            num_of_names_changed +=1

        ## JACQUES BARDELOT

        elif 'JACQUES BARDELOT' in desc_input and color_input == 'white' and '1.5L' in desc_input:
            row['identified_name'] = 'JACQUES BARDELOT CHAMPAGNE 1.5L'
            num_of_names_changed +=1

        ## RUINART

        elif 'RUINART' in desc_input and color_input == 'white' and size_input == '750ml' and (arp_input<=150  or arp_ly_input<=150):
            row['identified_name'] = 'RUINART BLANC DE BLANCS 750ML'
            num_of_names_changed +=1

        elif desc_input == 'RUINART SPK CHAMPAGNE ROSE PINK BRUT CHAMPAGNE FRANCE 750ML':
            row['identified_name'] = 'RUINART ROSE 750ML'
            num_of_names_changed +=1      

        ## GH MUMM

        elif 'GH MUMM GRAND CORDON' in desc_input and color_input == 'white' and size_input == '750ml':
            row['identified_name'] = 'GH MUMM GRAND CORDON 750ML'
            num_of_names_changed +=1

        ## ELSE 
        else:
            row['identified_name'] = ''
        
        return row
    
    df = df.apply(name_change, axis=1)
    df['identified_name'] = df.groupby('wine_upc')['identified_name'].transform(
    lambda x: x.replace('', None).ffill().bfill().fillna(''))
    
    return df[['item_desc_input', 'identified_name', 'wine_upc', 'brand_family_input',
               'color_input', 'size_input', 'Period Descriptions', '$', '$ YA',
               '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
               '%ACV Reach Where Dist NON ALCOHOLIC',
               '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
               '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
               '# Stores Last Period', '# Stores Selling Last Period',
               '$ per Store Selling Last Period', 'prestige_input', 'ARP', 'ARP LY',
               'ACV Calc']]

def return_identified_upcs_df(input_identified_upcs,truth):

    def correct_each_row(row):

        upc = row['wine_upc']
        row_true_attributes = truth[truth['wine_upc'] == upc]

        row['item_desc_input'] = str(row_true_attributes['item_desc_truth'].values[0])
        row['brand_family_input'] = str(row_true_attributes['brand_family_truth'].values[0])
        row['color_input'] = str(row_true_attributes['color_truth'].values[0])
        row['size_input'] = str(row_true_attributes['size_truth'].values[0])
        if str(row_true_attributes['prestige_truth'].values[0]) == 'nan':
            row['prestige_input'] = np.nan
        else:
            row['prestige_input'] = str(row_true_attributes['prestige_truth'].values[0])

        return row
    
    input_identified_upcs = input_identified_upcs.apply(correct_each_row,axis=1)
    input_identified_upcs.rename(columns={'wine_upc':'UPC',
                             'item_desc_input':'Item Names',
                             'prestige_input':'Prestige',
                             'brand_family_input':'Brand Families',
                             'color_input':'Colors',
                             'size_input':'Sizes'},inplace=True)
    
    input_identified_upcs = input_identified_upcs[['Item Names', 'UPC', 'Brand Families',
       'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA',
       '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
       '%ACV Reach Where Dist NON ALCOHOLIC',
       '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
       '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
       '# Stores Last Period', '# Stores Selling Last Period',
       '$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY',
       'ACV Calc']]
    
    input_identified_upcs = input_identified_upcs.fillna('')
    
    return input_identified_upcs

def return_df_in_right_format(df):

    df = df.replace('', np.nan)
    
    def turn_back_arp_to_dolls(row):

        if str(row).lower().strip() !='nan':
            row_dolls = '$' + str(row)
        else:
            row_dolls =''
    
        return row_dolls

    df.columns = df.columns.str.strip()
    df_copy = df.copy()
    df_copy['ARP'] = df['ARP'].astype(str)
    df_copy['ARP LY'] = df['ARP LY'].astype(str)
    df_copy.loc[:,'ARP'] = df['ARP'].apply(turn_back_arp_to_dolls)
    df_copy.loc[:,'ARP LY'] = df['ARP LY'].apply(turn_back_arp_to_dolls)

    def turn_to_int(x):
        if pd.isna(x):
            return ''
        else:
            return int(float(x))
    
    def format_percentage(x):
        if pd.isna(x):
            return ''
        else:
            return '{:.2f}%'.format(x * 100)
    
    df_copy['# Stores Last Period'] = df['# Stores Last Period'].apply(lambda x: turn_to_int(x))
    df_copy['# Stores Selling Last Period'] = df['# Stores Selling Last Period'].apply(lambda x: turn_to_int(x))
    df_copy['# Stores'] = df['# Stores'].apply(lambda x: turn_to_int(x))
    df_copy['# Stores Selling'] = df['# Stores Selling'].apply(lambda x: turn_to_int(x))
    df_copy['$'] = df['$'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['$ YA'] = df['$ YA'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['9 Liter Equivs'] = df['9 Liter Equivs'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['9 Liter Equivs YA'] = df['9 Liter Equivs YA'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['Units'] = df['Units'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['Units LY'] = df['Units LY'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['$ per Store Selling'] = df['$ per Store Selling'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    df_copy['$ per Store Selling Last Period'] = df['$ per Store Selling Last Period'].astype(float).apply(lambda x: '{:.2f}'.format(x)).replace('nan','')
    df_copy['$ Per Point of ACV'] = df['$ Per Point of ACV'].astype(float).apply(lambda x: '{:.2f}'.format(x))
    try:
        df_copy['%ACV Reach Where Dist NON ALCOHOLIC'] = df_copy['%ACV Reach Where Dist NON ALCOHOLIC'].astype(float).apply(lambda x: format_percentage(x))
        df_copy['%ACV Reach Where Dist YA NON ALCOHOLIC'] = df_copy['%ACV Reach Where Dist YA NON ALCOHOLIC'].astype(float).apply(lambda x: format_percentage(x))
    except ValueError:
        pass
    
    df_final = df_copy.fillna('')
    
    return df_final

def highlight_name_changes(df,new_column_name='Suggested name modification'):

    def highlight_proposition(s):
        return ['color: green' if s.name == new_column_name else '' for v in s]

    def highlight_nielsen_name(s):
        return ['color: red' if s.name == 'Nielsen Item Names' else '' for v in s]

    styled_df = df.style.apply(highlight_proposition, subset=[new_column_name])\
                        .apply(highlight_nielsen_name, subset=['Nielsen Item Names'])

    return styled_df

def display_unidentified_names_df(unidentified_names_df):
    return gr.update(value=unidentified_names_df, visible=True)


intro_description = """
**Please ensure the following before uploading your file:**
- The file is in CSV format.
- The UPC column contains only integers i.e. no scientific format (e.g. 8.18E+09 is wrong, 8175305020 is right).
- You did not change any of the column names from Nielsen.
"""

def handle_decision(decision):
    if decision:
        return gr.update(visible=True)
    else:
        return None

def confirm_changes():
    return gr.update(visible=True)

def apply_changes(choice):
    if choice == "Ok":
        return "Changes accepted."
    else:
        return "Changes canceled."

def accept_all_changes_fn(identified_names_df_state,
                          unidentified_names_df_state,
                          truth):
    
    # Clean identified_names_df 
    identified_names_df = identified_names_df_state.copy()
    identified_names_df.drop(columns=['Nielsen Item Names'],inplace=True)
    identified_names_df.rename(columns={'Suggested name modification':'Item Names','Nielsen Item Names ID':'UPC'},inplace=True)
    identified_names_df_int = data_prep_for_all_input_dfs(identified_names_df,additional_column=None,check=False)
    identified_names_df_final = correct_attributes_from_name(identified_names_df_int,truth)
    
    # Create a dataframe of the identified names to add to the truth
    identified_names_df_final_upcs_to_add = identified_names_df_final.drop_duplicates(subset=['UPC'])
    add_to_truth_df = return_df_in_right_format(identified_names_df_final_upcs_to_add)
    
    unidentified_upcs_unidentified_names_message = """# Review the remaining unidentified UPCs
    
    Here are the remaining rows of your CSV file, for which we could not:
    - Identify the UPC
    - Identify the name for an unknown UPC
    
    Again, please review the rows carefully and choose between:
    - **No additional modification** i.e. leaving all the rows as is
    - **Submit additional modifications** i.e. adding new UPCs to the truth after having modified them.
    
    ### If you choose to submit additional modifications, please submit a CSV with only the UPCs that you want to add to the ground truth.
    
    Also, if you want UPCs to be removed from the analysis because they are not champagne, you can add them when you submit additional modifications and put 'REMOVE - NOT FRENCH CHAMPAGNE' in the 'Your Item Name Modifications' column.
    """
    list_of_upcs_of_identified_names = identified_names_df_final['UPC'].unique().tolist()
    
    unidentified_names_df_new_state = unidentified_names_df_state[~unidentified_names_df_state['Nielsen Item Names ID'].isin(list_of_upcs_of_identified_names)]
    unidentified_names_df_csv = unidentified_names_df_new_state.copy()
    unidentified_names_df_csv['Your Item Name Modifications'] = ''
    unidentified_names_df_csv = unidentified_names_df_csv.fillna('')
    unidentified_names_df_csv_final = unidentified_names_df_csv[['Nielsen Item Names', 'Your Item Name Modifications', 'Nielsen Item Names ID', 'Brand Families',
       'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA',
       '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
       '%ACV Reach Where Dist NON ALCOHOLIC',
       '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
       '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
       '# Stores Last Period', '# Stores Selling Last Period',
       '$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY',
       'ACV Calc']]
    temp_file_3 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Unknown_UPCs_unidentified_names_")
    unidentified_names_df_csv_final.to_csv(temp_file_3.name, index=False)
    
    return (
        identified_names_df_final, # identified_names_df_state
        gr.update(interactive=False), # accept_all_changes_button
        gr.update(interactive=False), # make_modifications_to_the_names_button
        gr.update(interactive=False), # make_no_modifications_to_names_button
        gr.update(value= unidentified_upcs_unidentified_names_message, visible=True), # unidentified_names_markdown
        unidentified_names_df_new_state, # unidentified_names_df_state
        gr.update(value=unidentified_names_df_new_state, visible=True), # unidentified_names_df
        gr.update(value=temp_file_3.name,visible=True), # unidentified_names_csv
        gr.update(visible=True,interactive=True), # no_additional_modification_button
        gr.update(visible=True,interactive=True), # yes_additional_modifications_button
        add_to_truth_df) # add_to_truth_df_state


def click_own_name_correction_fn():
    message = """
    # You have decided to provide your own name correction\n
    Please upload your correction as a CSV hereunder.\n
    It will be added to the ground truth at the end of the session."""
    return (gr.update(interactive=False), # accept_all_changes_button
            gr.update(interactive=False), # make_modifications_to_the_names_button
            gr.update(interactive=False), # make_no_modifications_to_names_button
            gr.update(value=message,visible=True), # submit_your_correction_markdown
            gr.update(visible=True), # inp_correction
            gr.update(visible=True) # submit_own_correction_for_unknown_upcs_identified_names_button
            )

def csv_input_cleared_fn():
    return (gr.update(visible = False), # upc_message
            gr.update(visible = False), # name_identification_markdown
            gr.update(value=None, visible = False), # out_df
            gr.update(value=None, visible = False), # name_modifications_proposition_csv
            gr.update(visible = False,interactive=True), # accept_all_changes_button
            gr.update(visible = False, interactive=True), # make_modifications_to_the_names_button
            gr.update(visible=False, interactive=True), # make_no_modifications_to_names_button
            gr.update(visible = False), # submit_your_correction_markdown
            gr.update(visible = False), # inp_correction
            gr.update(visible = False), # accept_all_changes_markdown
            gr.update(visible = False), # unidentified_names_df
            gr.update(visible = False), # unidentified_names_markdown
            gr.update(value=None, visible = False), # unidentified_names_df
            gr.update(value=None, visible = False), # unidentified_names_csv
            gr.update(visible = False, interactive = True), # no_additional_modification_button
            gr.update(visible = False, interactive = True), # yes_additional_modifications_button
            gr.update(interactive = True), # submit_input_csv_button
            gr.update(visible = False), # final_name_identification_markdown
            gr.update(visible=False,value=None), # final_df
            gr.update(visible=False,value=None), # final_csv
            gr.update(visible=False)) # submit_own_correction_for_unknown_upcs_identified_names_button

def confirm_your_choice(confirmation):
    if confirmation == 'CONFIRM':
        return gr.update()
    else:
        return None

def state_to_visible_df_fn(state):
    return gr.update(value=state,visible=True)

def display_unidentified_names_csv_fn(unidentified_names_df):
    unidentified_names_df_pandas_df = pd.DataFrame(unidentified_names_df)
    temp_file_3 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Unknown_UPCs_unidentified_names_")
    unidentified_names_df_pandas_df.to_csv(temp_file_3.name, index=False)
    return (gr.update(value=temp_file_3.name,visible=True), # unidentified_names_csv
            gr.update(visible=True), # no_additional_modification_button
            gr.update(visible=True)) # yes_additional_modifications_button

def no_additional_modification_button_fn(input_csv_df_state,
                                         known_upcs_df_state,
                                         identified_names_df_state,
                                         additional_modifications_df_state,
                                         unidentified_names_df_state,
                                         add_to_truth_df_state,
                                         upcs_to_delete_df_state,
                                         upcs_to_modify_df_state):

    # CHANGE NAMES OF UNIDENTIFIED NAMES
    unidentified_names_df = unidentified_names_df_state.copy()
    unidentified_names_df.rename(columns={'Nielsen Item Names':'Item Names','Nielsen Item Names ID':'UPC'},inplace=True)

    # CHANGES COMPUTATIONS
    # For known UPCs
    number_of_rows_with_identified_upcs = known_upcs_df_state.shape[0]
    known_upcs_df_unique = known_upcs_df_state.drop_duplicates(subset=['UPC'])
    number_of_identified_upcs = known_upcs_df_unique.shape[0]

    # For identified names
    if add_to_truth_df_state is not None:
        number_of_upcs_with_identified_names = add_to_truth_df_state.shape[0]
        list_of_known_upcs = add_to_truth_df_state['UPC'].unique().tolist()
        identified_names_count = input_csv_df_state['Nielsen Item Names ID'].isin(list_of_known_upcs).sum()
    else:
        number_of_upcs_with_identified_names = 0
        identified_names_count = 0


    # Number of rows removed
    known_upcs_df_minus_remove = known_upcs_df_state[~known_upcs_df_state['Item Names'].str.contains('remove', case=False, na=False)].reset_index(drop=True)
    number_of_rows_removed = known_upcs_df_state.shape[0] - known_upcs_df_minus_remove.shape[0]
    
    # For unchanged rows
    unchanged_rows = unidentified_names_df.shape[0]


    final_result_message = f"""
    # Here is the corrected data
    
    Do not forget when copying and pasting the data from this CSV to your Excel, to first convert the UPCs to integers.
    
    Here is a summary of the modifications made:
    - We already knew **{number_of_identified_upcs} UPCs** - this corresponds to **{number_of_rows_with_identified_upcs} rows**.
    - We managed to identify the name of **{number_of_upcs_with_identified_names} unknown UPCs** - this corresponds to **{identified_names_count} rows**.
    - **{number_of_rows_removed} rows** with identified names or UPCs were removed because they were not Champagne.
    - **{unchanged_rows} rows** were unchanged.
    """
    final_df, temp_file4_name = final_df_computation(known_upcs_df_minus_remove,identified_names_df_state,additional_modifications_df_state,unidentified_names_df)

    # INFORMATION ON MODIFICATIONSAND COLORIZE DFs
    if upcs_to_delete_df_state is not None:
        upcs_to_delete_from_truth_message = f"""## UPCs to remove from Truth
        
        The following {upcs_to_delete_df_state.shape[0]} UPCs will be removed from the ground truth."""
        upcs_to_delete_df_state_styled = upcs_to_delete_df_state.style.map(lambda _: 'color: red', subset=['Item Names'])
    else:
        upcs_to_delete_from_truth_message = ''
        upcs_to_delete_df_state_styled = None

    if upcs_to_modify_df_state is not None:
        upcs_to_modify_in_truth_message = f"""## UPCs to modify in Truth

        The following {upcs_to_modify_df_state.shape[0]} UPCs will be modified in the ground truth."""
        upcs_to_modify_df_state_styled = upcs_to_modify_df_state.style.map(lambda _: 'color: orange', subset=['Item Names'])
    else:
        upcs_to_modify_in_truth_message=''
        upcs_to_modify_df_state_styled = None

    if add_to_truth_df_state is not None:
        upcs_to_add_to_truth_message = f"""## UPCs to add to Truth

        The following {add_to_truth_df_state.shape[0]} UPCs will be added to the ground truth."""
        add_to_truth_df_state_styled = add_to_truth_df_state.style.map(lambda _: 'color: green', subset=['Item Names'])
    else:
        upcs_to_add_to_truth_message=''
        add_to_truth_df_state_styled = None

    def conditional_update(df, df_styled, markdown_message):
        if df is not None and df.shape[0] > 0:
            return gr.update(value=markdown_message, visible=True), gr.update(value=df_styled, visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    
    if all(state is None for state in [upcs_to_modify_df_state, add_to_truth_df_state, upcs_to_delete_df_state]):
        upcoming_modifications_message= f"""## No modifications will be made to the truth in this session"""
        
        return (
            unidentified_names_df, # unidentified_names_df_state
            gr.update(value = final_result_message, visible=True), # final_name_identification_markdown
            gr.update(interactive=False), # no_additional_modification_button
            gr.update(interactive=False), # yes_additional_modifications_button
            gr.update(value=final_df,visible=True), # final_df
            gr.update(value=temp_file4_name,visible=True), # final_csv

            gr.update(value=upcoming_modifications_message, visible=True), # upcoming_modifications_markdown
            
            *conditional_update(upcs_to_delete_df_state, upcs_to_delete_df_state_styled, upcs_to_delete_from_truth_message),
            *conditional_update(upcs_to_modify_df_state, upcs_to_modify_df_state_styled, upcs_to_modify_in_truth_message),
            *conditional_update(add_to_truth_df_state, add_to_truth_df_state_styled, upcs_to_add_to_truth_message),
            
            gr.update(visible=False) # confirm_modifications_to_truth_button
            ) 
    else:
        upcoming_modifications_message = """
        # Upcoming Truth modifications
        
        ### Download the CSV above before clicking on **"Make these changes to the ground truth"** as this will reload the app.

        Please, also check these rows before confirming - these modifications will be made on the ground truth and used for the next data cleansing sessions.

        If you do not agree with the listed modifications, start the session over and correct the UPCs.

        If after doing that, you still do not agree with the table, please contact me via email at: augustin.destaff@gmail.com
        """


        return (
            unidentified_names_df, # unidentified_names_df_state
            gr.update(value = final_result_message, visible=True), # final_name_identification_markdown
            gr.update(interactive=False), # no_additional_modification_button
            gr.update(interactive=False), # yes_additional_modifications_button
            gr.update(value=final_df,visible=True), # final_df
            gr.update(value=temp_file4_name,visible=True), # final_csv

            gr.update(value=upcoming_modifications_message, visible=True), # upcoming_modifications_markdown
            
            *conditional_update(upcs_to_delete_df_state, upcs_to_delete_df_state_styled, upcs_to_delete_from_truth_message),
            *conditional_update(upcs_to_modify_df_state, upcs_to_modify_df_state_styled, upcs_to_modify_in_truth_message),
            *conditional_update(add_to_truth_df_state, add_to_truth_df_state_styled, upcs_to_add_to_truth_message),
            
            gr.update(visible=True) # confirm_modifications_to_truth_button
            ) 

def submit_own_correction_for_unknown_upcs_identified_names_fn(inp_correction):
    correction_df = pd.read_csv(process_csv(inp_correction))
    if check_if_column_has_empty_cells(correction_df,'Suggested name modification'):
        raise gr.Error("""The "Suggested name modification" column should not have empty cells. If you do not want to add a UPC to the ground truth, please delete the corresponding line""")
    check_upcs_have_diff_item_names(correction_df,item_names_column='Suggested name modification',upc_column='Nielsen Item Names ID')
    correction_df_prepared = data_prep_for_all_input_dfs(correction_df,additional_column='Suggested name modification')
    correction_final_df = return_df_in_right_format(correction_df_prepared)
    correction_final_df_styled = highlight_name_changes(correction_final_df,'Suggested name modification')
    own_name_correction_message = """
    ### Please go through the correction you submitted before confirming\n
    You can directly modify the dataframe if needed.\n
    If you click on confirm, these UPCs will be added to the ground truth at the end of the session.
    """
    return (
        gr.update(value=correction_final_df_styled,visible=True), # own_name_correction_df
        gr.update(value=own_name_correction_message,visible=True), # own_name_correction_markdown
        gr.update(visible=True), # confirm_own_correction_button
        gr.update(visible=True)) # cancel_own_correction_button

def click_confirm_own_correction_fn(inp_correction,
                                    known_upcs_df_state,
                                    input_csv_df_state,
                                    unidentified_names_df_state,
                                    truth,
                                    add_to_truth_df_state):
    
    identified_names_df = pd.read_csv(inp_correction.name)
    identified_names_df_int = data_prep_for_all_input_dfs(identified_names_df,additional_column='Suggested name modification')
    identified_names_df_int.drop(columns=['Nielsen Item Names'],inplace=True)
    identified_names_df_int.rename(columns={'Suggested name modification':'Item Names','Nielsen Item Names ID':'UPC'},inplace=True)
    identified_names_df_int_2 = correct_attributes_from_name(identified_names_df_int,truth)
    identified_names_final_df = return_df_in_right_format(identified_names_df_int_2)
    
    identified_names_final_df_unique_upcs = identified_names_final_df.drop_duplicates(subset=['UPC'])

    try:
        new_truth_state = pd.concat([add_to_truth_df_state,identified_names_final_df_unique_upcs],ignore_index=True)
    except TypeError:
        new_truth_state = identified_names_final_df_unique_upcs
    

    list_of_known_upc = known_upcs_df_state['UPC'].unique().tolist()
    list_of_named_upcs = identified_names_final_df['UPC'].unique().tolist()
    all_upcs_done_list = list_of_known_upc + list_of_named_upcs
    
    if len(all_upcs_done_list)>0:
        unidentified_names_df = input_csv_df_state[~input_csv_df_state['Nielsen Item Names ID'].isin(all_upcs_done_list)]
    else:
        unidentified_names_df = unidentified_names_df_state
    
    
    ### ADD AN EMPTY COLUMN FOR NAME MODIFICATION FOR THE CSV
    unidentified_names_df_csv = unidentified_names_df.copy()
    unidentified_names_df_csv['Your Item Name Modifications']=np.nan
    unidentified_names_df_csv_final = unidentified_names_df_csv[['Nielsen Item Names', 'Your Item Name Modifications', 'Nielsen Item Names ID', 'Brand Families',
                                                                 'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA',
                                                                 '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
                                                                 '%ACV Reach Where Dist NON ALCOHOLIC',
                                                                 '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
                                                                 '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
                                                                 '# Stores Last Period', '# Stores Selling Last Period',
                                                                 '$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY','ACV Calc']]
    unidentified_names_df_csv_final = unidentified_names_df_csv.fillna('')
    temp_file_3 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="unidentified_names_df_")
    unidentified_names_df_csv_final.to_csv(temp_file_3.name, index=False)
    click_confirm_own_correction_message = """
    ### Your modifications will be added to the ground truth."""
    unidentified_upcs_unidentified_names_message = """# Review the remaining unidentified UPCs
    
    Here are the remaining rows of your CSV file, for which we could not:
    - Identify the UPC
    - Identify the name for an unknown UPC

    Again, please review the rows carefully and choose between:
    - **No additional modification** i.e. leaving all the rows as is
    - **Submit additional modifications** i.e. adding new UPCs to the truth after having modified them.
    
    ### If you choose to submit additional modifications, please submit a CSV with only the UPCs that you want to add to the ground truth.
    For the UPCs for which you can identify the name, plesae fill in the 'Your Item Name Modifications' column only, the algorithm will take care of correcting the other columns from the name.
    """

    return (
        identified_names_final_df, # identified_names_df_state
        unidentified_names_df, # unidentified_names_df_state
        gr.update(interactive=False), # confirm_own_correction_button
        gr.update(interactive=False), # cancel_own_correction_button
        gr.update(value=click_confirm_own_correction_message,visible=True), # click_confirm_own_correction_markdown
        gr.update(value=unidentified_upcs_unidentified_names_message,visible=True), # unidentified_names_markdown
        gr.update(value=unidentified_names_df, visible=True), # unidentified_names_df
        gr.update(visible=True),# no_additional_modification_button
        gr.update(visible=True),#yes_additional_modifications_button
        gr.update(value = temp_file_3.name,visible=True), # unidentified_names_csv
        new_truth_state) # add_to_truth_df_state


def click_cancel_own_correction_button_fn():
    return (gr.update(visible=False), # submit_your_correction_markdown
            gr.update(value=None, visible=False), # inp_correction
            gr.update(interactive=True, visible=False), # submit_own_correction_for_unknown_upcs_identified_names_button
            gr.update(visible=False), # own_name_correction_markdown
            gr.update(visible=False, value=None), # own_name_correction_df
            gr.update(visible=False, interactive=True), # confirm_own_correction_button
            gr.update(visible=False, interactive=True), # cancel_own_correction_button
            gr.update(interactive=True), # accept_all_changes_button
            gr.update(interactive=True), # make_modifications_to_the_names_button
            gr.update(interactive=True)) # make_no_modifications_to_names_button


def click_yes_additional_modifications_button_fn():
    submit_additional_modifications_message = """
    # Submit your additional modifications\n
    As for the initial CSV input, please respect all the rules.
    """
    return (gr.update(value = submit_additional_modifications_message,visible=True), # submit_additional_modifications_markdown
            gr.update(visible=True), # inp_additional_modifications
            gr.update(visible=True), # submit_upload_of_additional_modifications_button                                                  
            gr.update(interactive=False), # yes_additional_modifications_button
            gr.update(interactive=False)) # no_additional_modifications_button

def submit_upload_of_additional_modifications_button_fn(inp,unidentified_names_df_state):
    

    additional_modifications_df = pd.read_csv(process_csv(inp))
    check_upcs_have_diff_item_names(additional_modifications_df,item_names_column='Your Item Name Modifications',upc_column='Nielsen Item Names ID')
    if check_if_column_has_empty_cells(additional_modifications_df,'Your Item Name Modifications'):
        raise gr.Error("""The "Your Item Name Modifications" column should not have empty cells. If you do not want to add a UPC to the ground truth, please delete the corresponding line""")
    
    if additional_modifications_df.shape[0] > unidentified_names_df_state.shape[0]:
        error = "Your additional modifications CSV file cannot have more rows than the remaining rows with unidentified names (i.e. than the CSV you downloaded above)."
        raise gr.Error(error)
    additional_modifications_df_prepared = data_prep_for_all_input_dfs(additional_modifications_df,additional_column='Your Item Name Modifications',check=True)
    additional_modifications_final_df = return_df_in_right_format(additional_modifications_df_prepared)
    additional_modifications_final_df_styled = highlight_name_changes(additional_modifications_final_df,'Your Item Name Modifications')
    additional_modifications_submitted_message = """
    ### Please review your additional modifications carefully before confirming\n
    These UPCs will be added to the ground truth and used for future data cleansing sessions."""
    return(gr.update(value=additional_modifications_submitted_message,visible=True), # additional_modifications_submitted_markdown
           gr.update(value=additional_modifications_final_df_styled,visible=True), # additional_modifications_submitted_df
           gr.update(visible=True), # confirm_additional_modifications_button
           gr.update(visible=True) # cancel_additional_modifications_button
    )

def cancel_additional_modifications_button_fn():
    return(
        None,
        gr.update(visible=False), # submit_additional_modifications_markdown
        gr.update(value=None, visible=False), # inp_additional_modifications
        gr.update(visible=False), # submit_upload_of_additional_modifications_button
        gr.update(visible=False), # additional_modifications_submitted_markdown
        gr.update(visible=False,value=None), # additional_modifications_submitted_df
        gr.update(interactive=True), # no_additional_modification_button
        gr.update(interactive=True), # yes_additional_modifications_button
        gr.update(visible=False), # confirm_additional_modifications_button
        gr.update(visible=False) # cancel_additional_modifications_button
    )

def change_column_names_from_input_to_output(df):
    df_ouput = df[['Item Names', 'UPC', 'Brand Families',
       'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA',
       '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
       '%ACV Reach Where Dist NON ALCOHOLIC',
       '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
       '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
       '# Stores Last Period', '# Stores Selling Last Period',
       '$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY',
       'ACV Calc']]
    return df_ouput

def final_df_computation(known_upcs_df_state,identified_names_df_state,additional_modifications_df_state,unidentified_names_df_state):
    
    dfs_to_concat = []
    
    if known_upcs_df_state is not None:
        dfs_to_concat.append(known_upcs_df_state)
    
    if identified_names_df_state is not None:
        dfs_to_concat.append(identified_names_df_state)

    if additional_modifications_df_state is not None:
        dfs_to_concat.append(additional_modifications_df_state)

    if unidentified_names_df_state is not None:
        dfs_to_concat.append(unidentified_names_df_state)

    final_df = pd.concat(dfs_to_concat,ignore_index=True)
    temp_file_4 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Corrected_data_final_")
    final_df.to_csv(temp_file_4.name, index=False)

    return final_df, temp_file_4.name

def confirm_additional_modifications_button_fn(input_csv_df_state,
                                               inp_additional_modifications,
                                               unidentified_names_df_state,
                                               known_upcs_df_state,
                                               identified_names_df_state,
                                               truth,
                                               add_to_truth_df_state,
                                               upcs_to_delete_df_state,
                                               upcs_to_modify_df_state):

    # Correct the additional modification UPCs from the truth
    additional_modifications_df = pd.read_csv(inp_additional_modifications.name)
    additional_modifications_df_int = data_prep_for_all_input_dfs(additional_modifications_df,additional_column='Your Item Name Modifications',check=False)
    additional_modifications_df_int.drop(columns=['Nielsen Item Names'],inplace=True)
    additional_modifications_df_int.rename(columns={'Your Item Name Modifications':'Item Names','Nielsen Item Names ID':'UPC'},inplace=True)
    additional_modifications_df_int_2 = correct_attributes_from_name(additional_modifications_df_int,truth)
    additional_modifications_df_final = return_df_in_right_format(additional_modifications_df_int_2)
    
    # List the unique UPCs of additional modification 
    additional_modifications_df_final_unique_upcs = additional_modifications_df_final.drop_duplicates(subset=['UPC'])

    # Add the additional modifications to the add_to_truth_df_state
    try:
        new_truth_state = pd.concat([add_to_truth_df_state,additional_modifications_df_final_unique_upcs],ignore_index=True)
    except TypeError:
        new_truth_state = additional_modifications_df_final_unique_upcs
        print("No additional modifications, so add_to_truth_df_state stays the same")

    # Drop the UPCs of additional modifications for the unidentified_names_df
    add_modified_upcs_list = additional_modifications_df_final['UPC'].unique().tolist()
    unidentified_names_df = unidentified_names_df_state[~unidentified_names_df_state['Nielsen Item Names ID'].isin(add_modified_upcs_list)]
    unidentified_names_df_final = unidentified_names_df.rename(columns={'Nielsen Item Names ID':'UPC','Nielsen Item Names':'Item Names'})
    confirm_additional_modifications_message = """### These UPCS will be added to the groumd truth"""

    # CHANGES COMPUTATIONS
    # For known UPCs
    number_of_rows_with_identified_upcs = known_upcs_df_state.shape[0]
    known_upcs_df_unique = known_upcs_df_state.drop_duplicates(subset=['UPC'])
    number_of_identified_upcs = known_upcs_df_unique.shape[0]
    
    # For identified names
    number_of_upcs_with_identified_names = new_truth_state.shape[0]
    list_of_known_upcs = new_truth_state['UPC'].unique().tolist()
    identified_names_count = input_csv_df_state['Nielsen Item Names ID'].isin(list_of_known_upcs).sum()

    # Number of rows removed
    known_upcs_df_minus_remove = known_upcs_df_state[~known_upcs_df_state['Item Names'].str.contains('remove', case=False, na=False)].reset_index(drop=True)
    number_of_rows_removed = known_upcs_df_state.shape[0] - known_upcs_df_minus_remove.shape[0]

    # For unchanged rows    
    unchanged_rows = unidentified_names_df_final.shape[0]


    final_result_message = f"""
    # Here is the corrected data

    Do not forget when copying and pasting the data from this CSV to your Excel, to first convert the UPCs to integers.

    Here is a summary of the modifications made:
    - We already knew **{number_of_identified_upcs} UPCs** - this corresponds to **{number_of_rows_with_identified_upcs} rows**.
    - We managed to identify the name of **{number_of_upcs_with_identified_names} unknown UPCs** - this corresponds to **{identified_names_count} rows**.
    - **{number_of_rows_removed} rows** with identified names or UPCs were removed because they were not Champagne.
    - **{unchanged_rows} rows** were unchanged.
    """

    final_df, temp_file4_name = final_df_computation(known_upcs_df_minus_remove,
                                                     identified_names_df_state,
                                                     additional_modifications_df_final,
                                                     unidentified_names_df_final)

    # INFORMATION ON MODIFICATIONSAND COLORIZE DFs
    if upcs_to_delete_df_state is not None:
        upcs_to_delete_from_truth_message = f"""## UPCs to remove from Truth
    
        The following {upcs_to_delete_df_state.shape[0]} UPCs will be removed from the ground truth."""
        upcs_to_delete_df_state_styled = upcs_to_delete_df_state.style.map(lambda _: 'color: red', subset=['Item Names'])
    else:
        upcs_to_delete_from_truth_message = ''
        upcs_to_delete_df_state_styled = None

    if upcs_to_modify_df_state is not None:
        upcs_to_modify_in_truth_message = f"""## UPCs to modify in Truth

        The following {upcs_to_modify_df_state.shape[0]} UPCs will be modified in the ground truth."""
        upcs_to_modify_df_state_styled = upcs_to_modify_df_state.style.map(lambda _: 'color: orange', subset=['Item Names'])
    else:
        upcs_to_modify_in_truth_message= ''
        upcs_to_modify_df_state_styled = None

    if new_truth_state is not None:
        upcs_to_add_to_truth_message = f"""## UPCs to add to Truth

        The following {new_truth_state.shape[0]} UPCs will be added to the ground truth."""
        add_to_truth_df_state_styled = new_truth_state.style.map(lambda _: 'color: green', subset=['Item Names'])
    else:
        upcs_to_add_to_truth_message = ''
        add_to_truth_df_state_styled = None


    def conditional_update(df, df_styled, markdown_message):
        if df is not None and df.shape[0] > 0:
            return gr.update(value=markdown_message, visible=True), gr.update(value=df_styled, visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    return (
        known_upcs_df_minus_remove, # known_upcs_df_state
        additional_modifications_df_final, # additional_modifications_df_state
        unidentified_names_df_final, # unidentified_names_df_state
        
        gr.update(interactive=False), # confirm_additional_modifications_button
        gr.update(interactive=False), # cancel_additional_modifications_button
        
        gr.update(value=confirm_additional_modifications_message,visible=True), # confirm_additional_modifications_markdown
        gr.update(value = final_result_message, visible=True), # final_name_identification_markdown
        gr.update(value=final_df,visible=True), # final_df
        gr.update(value=temp_file4_name,visible=True), # final_csv
        new_truth_state,  # add_to_truth_df_state

        gr.update(visible=True), # upcoming_modifications_markdown

        *conditional_update(upcs_to_delete_df_state, upcs_to_delete_df_state_styled, upcs_to_delete_from_truth_message),
        *conditional_update(upcs_to_modify_df_state, upcs_to_modify_df_state_styled, upcs_to_modify_in_truth_message),
        *conditional_update(new_truth_state, add_to_truth_df_state_styled, upcs_to_add_to_truth_message),

        gr.update(visible=True)) # confirm_modifications_to_truth_button

    

def check_if_column_has_empty_cells(df,column_name):
    
    number_of_empty_cells = df[df[column_name].isin(['',' ',np.nan])].shape[0]

    if number_of_empty_cells==0:
        return False
    else:
        return True

def correct_attributes_from_name(df, truth):

    # df_closer_look = pd.DataFrame(columns=df.columns)

    def change_attributes(row):

        name = str(row['Item Names']).strip().replace('  ', ' ')


        try:
            true_attributes = truth.loc[str(truth['item_desc_truth']).strip().replace('  ', ' ')==name].iloc[0]
            row['Brand Families'] = true_attributes['brand_family_truth']
            row['Sizes'] = true_attributes['size_truth']
            row['Colors'] = true_attributes['color_truth']
            return row
        except:

            def guess_size(name):
                guessed_size = np.nan
                try:
                    if 'REMOVE' in name:
                        guessed_size = 'REMOVE - NOT FRENCH CHAMPAGNE'
                    elif 'REM SIZE' in name:
                        guessed_size = 'REM SIZE'
                    elif name.split()[-1][-1] != 'L':
                        for i in range(15):
                            if name.split()[-(2+i)][-1] == 'L':
                                guessed_size = name.split()[-(2+i)]  
                                break   
                    elif name.split()[-1][-1] == 'L':
                        guessed_size = name.split()[-1]
                except:
                    pass
                
                return guessed_size
            
            def guess_brand(name):
                recognized_brand = np.nan

                brands_df = pd.read_csv('all_brand_families.csv')
                brands = brands_df['brands'].to_list()

                for brand in brands:
                    if brand.lower() in name.lower():
                        recognized_brand = brand
                    elif 'PL' in name.split():
                        recognized_brand = 'PRIVATE LABEL'
                    elif 'BRNS DE RTHSCHLD' in name:
                        recognized_brand = 'BARONS DE ROTHSCHILD'
                    elif 'CHAMPAGNE MOUZON LRX & FLS' in name:
                        recognized_brand = 'CHAMPAGNE MOUZON LEROUX & FILS'
                    elif 'CHAMPAGNE CHATEAU DE BLGNY' in name:
                        recognized_brand = 'CHAMPAGNE CHATEAU DE BLIGNY'
                    elif 'CRX DS VNQU' in name:
                        recognized_brand = 'DOMAINE DE LA CROIX DES VAINQUEURS'
                    elif 'CHAMPAGNE DAPSN' in name:
                        recognized_brand = 'PASCAL DOQUET'
                return recognized_brand

            def guess_color(name):
                guessed_color = np.nan
                if ('ROSE' in name or 'PINK' in name) and 'ROSES' not in name:
                    guessed_color = 'PINK'
                elif 'REMOVE' in name:
                    guessed_color = 'REMOVE - NOT FRENCH CHAMPAGNE'
                else:
                    guessed_color = 'WHITE'
                
                return guessed_color
            

            brand_guess = guess_brand(name)
            size_guess = guess_size(name)
            color_guess = guess_color(name)
            
            # if pd.isna(brand_guess) or pd.isna(size_guess) or pd.isna(color_guess):
            #     df_closer_look.loc[len(df_closer_look)] = row
            # else:
            row['Brand Families'] = brand_guess
            row['Sizes'] = size_guess
            row['Colors'] = color_guess
        
            return row
    df = df.replace('', np.nan)
    df = df.apply(lambda row: change_attributes(row), axis=1)

    def is_prestige(row):
        brand = str(row['Brand Families']).strip().replace('  ', ' ')
        color = str(row['Colors']).strip().lower()
        size = str(row['Sizes']).strip().lower()
        desc = str(row['Item Names']).strip().replace('  ', ' ')
        arp = float(row['ARP'])
        arp_ly = float(row['ARP LY'])
        
        if brand == 'KRUG' and color == 'white' and size == '375ml':
            return 'Prestige'
        if desc == 'LAURENT-PERRIER GRAND SIECLE 1.5L':
            return 'Prestige'
        if color == 'white' and size == '750ml':
            if brand == 'SALON':
                return 'Prestige'
            if brand == 'BILLECART-SALMON' and (arp >= 200 or arp_ly >= 200):
                return 'Prestige'
            if desc == 'BILLECART-SALMON CUVEE NICOLAS 750ML':
                return 'Prestige'
            if desc == 'VEUVE CLICQUOT LA GRANDE DAME 750ML' and (arp >= 150 or arp_ly >= 150):
                return 'Prestige'
            if brand == 'POL ROGER' and (arp >= 200 or arp_ly >= 200):
                return 'Prestige'
            if desc == 'BOLLINGER R.D. SPK CHAMPAGNE WHITE WHITE ORGANIC CHAMPAGNE FRANCE 750ML':
                return 'Prestige'
            if desc == 'BOLLINGER GRANDE ANNEE 750ML':
                return 'Prestige'
            if 'CLOS LANSON' in desc:
                return 'Prestige'
            if brand == 'RUINART' and (arp >= 200 or arp_ly >= 200):
                return 'Prestige'
            if brand == 'PIPER-HEIDSIECK' and (arp >= 200 or arp_ly >= 200):
                return 'Prestige'
            if 'PERRIER JOUET BELLE EPOQUE' in desc and (arp >= 150 or arp_ly >= 150):
                return 'Prestige'
            if brand == 'NICOLAS FEUILLATTE' and (arp >= 140 or arp_ly >= 140):
                return 'Prestige'
            if brand == 'HENRIOT' and (arp >= 180 or arp_ly >= 180):
                return 'Prestige'
            if desc == 'HENRIOT HEMERA 750ML':
                return 'Prestige'
            if desc == 'TAITTINGER COMTES DE CHAMPAGNE 750ML':
                return 'Prestige'
            if desc == 'ARMAND DE BRIGNAC ACE OF SPADES 750ML':
                return 'Prestige'
            if brand == 'LOUIS ROEDERER' and (arp >= 200 or arp_ly >= 200):
                return 'Prestige'
            if desc == 'POMMERY CUVEE LOUISE 750ML' and (arp >= 160 or arp_ly >= 160):
                return 'Prestige'
            if 'LAURENT-PERRIER GRAND SIECLE' in desc:
                return 'Prestige'
            if desc == 'KRUG GRANDE CUVEE 750ML':
                return 'Prestige'
            if desc == 'KRUG CLOS DU MESNIL 750ML':
                return 'Prestige'
            if brand == 'DOM PERIGNON':
                return 'Prestige'
        return np.nan
    
    df['Prestige'] = df.apply(is_prestige, axis=1)

    return df

def check_upcs_have_diff_item_names(df,item_names_column,upc_column):

    df_duplicates_full = df.drop_duplicates(subset=[item_names_column,upc_column])
    list_of_dup_upcs = (df_duplicates_full[upc_column][df_duplicates_full[upc_column].duplicated()]).tolist()

    if len(list_of_dup_upcs) > 0:
        raise gr.Error(f"Each UPC must be associated with a unique Item Name. Please check the following UPCs in your CSV: {list_of_dup_upcs}")
    

def click_UPCs_to_add_to_truth_button_fn(truth, add_to_truth_df_state,upcs_to_modify_df_state,upcs_to_delete_df_state):
    
    update_the_truth(truth, 
                    add_to_truth_df_state, 
                    upcs_to_modify_df_state, 
                    upcs_to_delete_df_state)


def modify_the_truth_button_fn():
    modify_the_truth_button_message = f"""Please submit your correction below."""
    return (
        gr.update(value = modify_the_truth_button_message,visible=True), # modify_the_truth_button_markdown
        gr.update(visible=True), # truth_correction_csv
        gr.update(visible=True) # submit_modifications_to_the_truth_button
    )


def submit_modifications_to_the_truth_button_fn(input_csv_df_state,
                                                truth_correction_csv, 
                                                known_upcs_df_state, 
                                                unidentified_names_df_state):

    correction_to_truth = pd.read_csv(truth_correction_csv)

    def check_column(df):
        valid_values = ['deleted', 'modified', '', np.nan]
        invalid_entries = df[~df['Modification Status'].isin(valid_values)]
        
        if not invalid_entries.empty:
            raise gr.Error(f"Please ensure the 'Modification Status' column is filled with either 'deleted', 'modified', or left blank. Invalid values found: {invalid_entries['Modification Status'].tolist()}")

    # Check the Modification Status column was filled using the right words
    check_column(correction_to_truth)

    # Clean the correction and check for anomalies
    correction_to_truth_cleaned = data_prep_for_all_input_dfs(correction_to_truth,additional_column='Modification Status', check=True, raw=False)
    
    # Create upcs_to_modify_df and upcs_to_delete_df to store the modifications to the truth
    columns_for_upc_dfs = correction_to_truth_cleaned.columns.tolist()
    columns_for_upc_dfs.remove('Modification Status')
    upcs_to_modify_df = pd.DataFrame(columns=columns_for_upc_dfs)
    upcs_to_delete_df = pd.DataFrame(columns=columns_for_upc_dfs)

    def check_truth_modifications(row):

        if row['Modification Status'] == "modified":
            new_row = row.drop(labels='Modification Status')
            upcs_to_modify_df.loc[len(upcs_to_modify_df)] = new_row
        
        if row['Modification Status'] == "deleted":
            new_row = row.drop(labels='Modification Status')
            upcs_to_delete_df.loc[len(upcs_to_delete_df)] = new_row
    
    # Store the modifications to the truth in the 2 dataframes and drop the rows to delete from the truth
    correction_to_truth_cleaned.apply(check_truth_modifications, axis=1)
    list_of_upcs_to_drop = upcs_to_delete_df['UPC'].unique().tolist()
    correction_to_truth_inter = correction_to_truth_cleaned[~correction_to_truth_cleaned['UPC'].isin(list_of_upcs_to_drop)]
    correction_to_truth_final = correction_to_truth_inter.drop(columns='Modification Status')
    
    list_of_known_upcs = correction_to_truth_final['UPC'].unique().tolist()
    identified_upcs_raw_df = input_csv_df_state[input_csv_df_state['Nielsen Item Names ID'].isin(list_of_known_upcs)]
    identified_upcs_df_int = identified_upcs_raw_df.rename(columns={'Nielsen Item Names':'Item Names',
                                           'Nielsen Item Names ID': 'UPC'})

    
    def update_features_from_truth(identified_upcs_df_int, correction_to_truth_final, id_col, features):
        
        identified_upcs_df_final = identified_upcs_df_int.drop(columns=features).merge(correction_to_truth_final[[id_col] + features], on=id_col, how='left')
        
        return identified_upcs_df_final[['Item Names', 'UPC', 'Brand Families', 'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA', '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY', '%ACV Reach Where Dist NON ALCOHOLIC', '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV','# Stores', '# Stores Selling', '$ per Store Selling', 'GT','# Stores Last Period', '# Stores Selling Last Period','$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY','ACV Calc']    ]
    
    identified_upcs_df_final = update_features_from_truth(identified_upcs_df_int, correction_to_truth_final, 'UPC', ['Item Names', 'Colors', 'Sizes', 'Brand Families','Prestige'])
    
    corrected_identified_upcs_final = return_df_in_right_format(identified_upcs_df_final)

    # Add the deleted rows from the truth to unidentified_names_df_state
    deleted_upcs_from_truth = known_upcs_df_state[known_upcs_df_state['UPC'].isin(upcs_to_delete_df['UPC'].unique().tolist())]

    deleted_upcs_from_truth_int = deleted_upcs_from_truth.rename(columns={'Item Names':'item_desc_input',
                                                                        'UPC':'wine_upc',
                                                                        'Brand Families':'brand_family_input',
                                                                        'Colors':'color_input',
                                                                        'Sizes':'size_input',
                                                                        'Prestige':'prestige_input'})

    if deleted_upcs_from_truth.shape[0] >0:
        unidentified_names_df_int = pd.concat([unidentified_names_df_state, deleted_upcs_from_truth_int],ignore_index=True)
        unidentified_names_df_final = return_df_in_right_format(unidentified_names_df_int)
    else:
        unidentified_names_df_final = return_df_in_right_format(unidentified_names_df_state)

    modified_truth_message = f"""In the submitted correction, you decided to **modify {upcs_to_modify_df.shape[0]} UPCs** and **remove {upcs_to_delete_df.shape[0]} UPCs** from the truth.
                                   
                                   After taking into account these modifications, a total number of **{len(identified_upcs_df_final['UPC'].unique().tolist())} UPCs were identified** in your dataset. In other words, **{identified_upcs_df_final.shape[0]} rows were automatically corrected**."""

    upcs_to_modify_df_final = return_df_in_right_format(upcs_to_modify_df)
    upcs_to_delete_df_final = return_df_in_right_format(upcs_to_delete_df)

    upcs_to_delete_df.rename(columns={'Item Names':'Nielsen Item Names', 'UPC':'Nielsen Item Names ID'}, inplace=True)

    # NOW IDENTIFY NAMES AMONG UNKNOWN UPCs
    # Try to match unknown UPC names with known names
    unidentified_names_df_int_int = data_prep_for_all_input_dfs(unidentified_names_df_final, additional_column=None, check=False, raw=False,colors='color_input')
    corrected_unknown_upcs = identify_name_of_unknown_upc(unidentified_names_df_int_int)
    corrected_unknown_upcs.rename(columns={'wine_upc':'Nielsen Item Names ID',
                                        'identified_name':'Suggested name modification',
                                        'item_desc_input':'Nielsen Item Names',
                                        'prestige_input':'Prestige',
                                        'brand_family_input':'Brand Families',
                                        'color_input':'Colors',
                                        'size_input':'Sizes'},inplace=True
                            )

    identified_names_df = corrected_unknown_upcs[corrected_unknown_upcs['Suggested name modification']!='']
    unidentified_names_df = corrected_unknown_upcs[corrected_unknown_upcs['Suggested name modification']=='']


    if identified_names_df.shape[0] > 0:
        identified_names_df_final = return_df_in_right_format(identified_names_df)
        identified_names_df_styled = highlight_name_changes(identified_names_df_final)
        
        unidentified_names_df_int = unidentified_names_df.copy()
        unidentified_names_df_int.drop(columns=['Suggested name modification'],inplace=True)
        unidentified_names_df_final = return_df_in_right_format(unidentified_names_df_int)
        
        temp_file_22 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Unidentified_UPCS_name_changes_proposition_")
        identified_names_df_final.to_csv(temp_file_22.name, index=False)
        
        name_identification_message = f"""
        # Name modifications for unknown UPCs

        Among the rows with unknown UPCs, we propose the following **{identified_names_df_final.shape[0]} name modifications** in the column 'Suggested name modification'
        - If all modifications look correct, click on 'Accept all suggested modifications' - these rows will be added to the truth and used in future tasks as the ground truth for these UPCS.
        - If not, download the output and do the following:
            - Change the names that you want to change in the 'Suggested name modification' column;
            - Delete the rows that you don't want to add to the ground truth;
            - Do not worry about changing other columns in the dataframe if they are incorrect, the algorithm will correct them using the updated names.
        """



    return (
        corrected_identified_upcs_final, # known_upcs_df_state
        gr.update(value=modified_truth_message, visible=True), # modified_truth_markdown
        upcs_to_modify_df_final, # upcs_to_modify_df_state
        upcs_to_delete_df_final, # upcs_to_delete_df_state
        gr.update(value=name_identification_message, visible=True), # name_identification_markdown
        gr.update(value=identified_names_df_styled, visible=True), # name_modifications_proposition_df
        gr.update(value=temp_file_22.name,visible=True), # name_modifications_proposition_csv
        gr.update(interactive=False), # proceed_with_this_truth_button
        gr.update(interactive=False), # modify_the_truth_button
        gr.update(visible=True), # accept_all_changes_button
        gr.update(visible=True), # make_modifications_to_the_names_button
        gr.update(visible=True), # make_no_modifications_to_names_button
        gr.update(interactive=False), # submit_modifications_to_the_truth_button
        unidentified_names_df_final, # unidentified_names_df_state
        identified_names_df_final # identified_names_df_state
    )

def make_no_modifications_to_names_button_fn(unidentified_names_df_state):

    make_no_modifications_to_names_message = f"""### You decided to skip the name identification step, no UPC will be added to the truth from this step."""
        
    unidentified_upcs_unidentified_names_message = """# Review the remaining unidentified UPCs
    
    Here are the remaining rows of your CSV file, for which we could not:
    - Identify the UPC
    - Identify the name for an unknown UPC
    
    Again, please review the rows carefully and choose between:
    - **No additional modification** i.e. leaving all the rows as is
    - **Submit additional modifications** i.e. adding new UPCs to the truth after having modified them.
    
    ### If you choose to submit additional modifications, please submit a CSV with only the UPCs that you want to add to the ground truth.
    
    Also, if you want UPCs to be removed from the analysis because they are not champagne, you can add them when you submit additional modifications and put 'REMOVE - NOT FRENCH CHAMPAGNE' in the 'Your Item Name Modifications' column.
    """
    
    unidentified_names_df_csv = unidentified_names_df_state.copy()
    unidentified_names_df_csv['Your Item Name Modifications'] = ''
    unidentified_names_df_csv = unidentified_names_df_csv.fillna('')
    unidentified_names_df_csv_final = unidentified_names_df_csv[['Nielsen Item Names', 'Your Item Name Modifications', 'Nielsen Item Names ID', 'Brand Families',
       'Colors', 'Sizes', 'Period Descriptions', '$', '$ YA',
       '9 Liter Equivs', '9 Liter Equivs YA', 'Units', 'Units LY',
       '%ACV Reach Where Dist NON ALCOHOLIC',
       '%ACV Reach Where Dist YA NON ALCOHOLIC', '$ Per Point of ACV',
       '# Stores', '# Stores Selling', '$ per Store Selling', 'GT',
       '# Stores Last Period', '# Stores Selling Last Period',
       '$ per Store Selling Last Period', 'Prestige', 'ARP', 'ARP LY',
       'ACV Calc']]
    temp_file_33 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="Unknown_UPCs_unidentified_names_")
    unidentified_names_df_csv_final.to_csv(temp_file_33.name, index=False)
    
    return (
        gr.update(interactive=False), # accept_all_changes_button
        gr.update(interactive=False), # make_modifications_to_the_names_button
        gr.update(interactive=False), # make_no_modifications_to_names_button
        gr.update(value=make_no_modifications_to_names_message, visible=True), # make_no_modifications_to_names_markdown
        gr.update(value= unidentified_upcs_unidentified_names_message, visible=True), # unidentified_names_markdown
        gr.update(value=unidentified_names_df_state, visible=True), # unidentified_names_df
        gr.update(value=temp_file_33.name,visible=True), # unidentified_names_csv
        gr.update(visible=True,interactive=True), # no_additional_modification_button
        gr.update(visible=True,interactive=True) # yes_additional_modifications_button

    )