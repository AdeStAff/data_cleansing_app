import gradio as gr
import pandas as pd
import numpy as np
import tempfile 
from io import StringIO
from functions import *
from huggingface_hub import HfApi

api = HfApi()


with gr.Blocks() as interface:
    
    ## Define initial states
    input_csv_df_state = gr.State()
    identified_upcs_df_state = gr.State()
    identified_names_df_state = gr.State()
    unidentified_names_df_state = gr.State()
    additional_modifications_df_state = gr.State()
    add_to_truth_df_state = gr.State()
    
    # Intro
    gr.Markdown("# Welcome to a new data cleansing session")
    gr.Markdown(intro_description)
    truth = gr.State(value=pd.read_csv('current_truth.csv'))
    input_csv = gr.File(label="Upload the Nielsen data as a CSV file", file_types=['csv'])
    submit_input_csv_button = gr.Button("Submit")
    
    upc_message = gr.Markdown(visible=False)
    identified_upcs_df = gr.DataFrame(label="Identified UPCs from the truth",visible=False)
    identified_upcs_csv = gr.File(label="Download the identified UPCs file",visible=False)

    ## PROCEED WITH IDENTIFIED UPCS OR MAKE MODIFICATIONS TO THE TRUTH
    with gr.Row():
        proceed_with_this_truth_button = gr.Button(value='Proceed with the current truth', icon='right_icon.svg', visible=False)
        modiify_the_truth_button = gr.Button(value='Modify the current truth', icon='wrong_icon.svg', visible=False)
    
    # Unidentified UPCs
    result_of_corrected_known_upcs_markdown = gr.Markdown(visible=False)
    name_modifications_proposition_df = gr.Dataframe(label="Name modifications",visible=False)
    unidentified_names_df = gr.Dataframe(label="Unidentified UPCs and names",visible=False)
    name_modifications_proposition_csv = gr.File(label="Download the name modification propositions file",visible=False)
    
    ## ACCEPT OR MAKE YOUR OWN MODIFICATIONS
    with gr.Row():
        accept_all_changes_button = gr.Button(value='Accept all suggested modifications', icon='right_icon.svg', visible=False)
        make_modifications_to_the_names_button = gr.Button(value='Submit your own correction', icon='wrong_icon.svg', visible=False)

    ## FIRST SUBMIT CLICK
    submit_input_csv_button.click(fn=identify_known_upcs, inputs=input_csv, outputs=[input_csv_df_state,
                                                                                     upc_message,
                                                                                     result_of_corrected_known_upcs_markdown,
                                                                                     name_modifications_proposition_df,
                                                                                     name_modifications_proposition_csv,
                                                                                     identified_upcs_df_state,
                                                                                     identified_upcs_df,
                                                                                     proceed_with_this_truth_button,
                                                                                     modiify_the_truth_button,
                                                                                     identified_names_df_state,
                                                                                     unidentified_names_df_state,
                                                                                     submit_input_csv_button,
                                                                                     identified_upcs_csv
                                                                                     ])

            

    
    proceed_with_this_truth_button.click(fn=proceed_with_identified_upcs_fn, outputs= [
        result_of_corrected_known_upcs_markdown,
        name_modifications_proposition_df,
        name_modifications_proposition_csv,
        proceed_with_this_truth_button,
        modiify_the_truth_button,
        result_of_corrected_known_upcs_markdown,
        name_modifications_proposition_df,
        name_modifications_proposition_csv,
        accept_all_changes_button,
        make_modifications_to_the_names_button
        ])
    
    accept_all_changes_markdown = gr.Markdown(visible=False)
    
    ## OWN CORRECTION
    submit_your_correction_markdown = gr.Markdown(visible=False)
    inp_correction = gr.File(label="Upload your own correction as a CSV file", file_types=['csv'],visible=False)
    submit_own_correction_for_unknown_upcs_identified_names_button = gr.Button("Submit",visible=False)

    own_name_correction_markdown = gr.Markdown(visible=False)
    own_name_correction_df = gr.DataFrame(visible=False)
    with gr.Row():
        confirm_own_correction_button = gr.Button('Confirm my modifications',icon='right_icon.svg',visible=False)
        cancel_own_correction_button = gr.Button('Cancel my modifications',icon='wrong_icon.svg',visible=False)


    submit_own_correction_for_unknown_upcs_identified_names_button.click(submit_own_correction_for_unknown_upcs_identified_names_fn,
                                                                         inp_correction,
                                                                         [own_name_correction_df,
                                                                          own_name_correction_markdown,
                                                                          confirm_own_correction_button,
                                                                          cancel_own_correction_button])
    
    click_confirm_own_correction_markdown = gr.Markdown(visible=False)
     
    unidentified_names_markdown = gr.Markdown(visible=False)
    unidentified_names_df = gr.Dataframe(label='Unknown UPCs with unidentified names',visible=False,interactive=False)
    
    unidentified_names_csv = gr.File(label="Download the unknown UPCs and unidentified names file",visible=False)
    
    with gr.Row():
        no_additional_modification_button = gr.Button(value='No additional modifications', icon='thumb_up.svg', visible=False)
        yes_additional_modifications_button = gr.Button(value='Submit additional modifications', icon='upload.svg', visible=False)

    
    confirm_own_correction_button.click(click_confirm_own_correction_fn,
                                        [inp_correction,
                                         identified_upcs_df_state,
                                         input_csv_df_state,
                                         unidentified_names_df_state,
                                         truth,
                                         add_to_truth_df_state],
                                        [identified_names_df_state,
                                         unidentified_names_df_state,
                                         confirm_own_correction_button,
                                         cancel_own_correction_button,
                                         click_confirm_own_correction_markdown,
                                         unidentified_names_markdown,
                                         unidentified_names_df,
                                         no_additional_modification_button,
                                         yes_additional_modifications_button,
                                         unidentified_names_csv,
                                         add_to_truth_df_state])

    cancel_own_correction_button.click(click_cancel_own_correction_button_fn,
                                       None,
                                       [submit_your_correction_markdown,
                                        inp_correction,
                                        submit_own_correction_for_unknown_upcs_identified_names_button,
                                        own_name_correction_markdown,
                                        own_name_correction_df,
                                        confirm_own_correction_button,
                                        cancel_own_correction_button,
                                        accept_all_changes_button,
                                        make_modifications_to_the_names_button
                                        ])



    accept_all_changes_button.click(accept_all_changes_fn, 
                                    [identified_names_df_state,
                                     unidentified_names_df_state,
                                     truth,
                                     add_to_truth_df_state], 
                                    [identified_names_df_state,
                                     accept_all_changes_button,
                                     make_modifications_to_the_names_button,
                                     accept_all_changes_markdown,
                                     unidentified_names_markdown,
                                     unidentified_names_df_state,
                                     unidentified_names_df,
                                     unidentified_names_csv,
                                     no_additional_modification_button,
                                     yes_additional_modifications_button,
                                     add_to_truth_df_state])
    
    make_modifications_to_the_names_button.click(click_own_name_correction_fn,
                                                 None,
                                                 [accept_all_changes_button,
                                                  make_modifications_to_the_names_button,
                                                  submit_your_correction_markdown,
                                                  inp_correction,
                                                  submit_own_correction_for_unknown_upcs_identified_names_button])

    ## SUBMIT ADDITIONAL MODIFICATIONS
    submit_additional_modifications_markdown = gr.Markdown(visible=False)
    inp_additional_modifications = gr.File(label="Upload your additional modifications as a CSV file", file_types=['csv'],visible=False)
    submit_upload_of_additional_modifications_button = gr.Button(value = 'Submit',visible=False)
    
    additional_modifications_submitted_markdown = gr.Markdown(visible=False)
    additional_modifications_submitted_df = gr.DataFrame(visible=False)

    with gr.Row():
        confirm_additional_modifications_button = gr.Button(value="Confirm these modifications",visible=False, icon='right_icon.svg')
        cancel_additional_modifications_button = gr.Button(value="Cancel these modifications",visible=False, icon='wrong_icon.svg')



    yes_additional_modifications_button.click(click_yes_additional_modifications_button_fn,
                                                 None,
                                                 [submit_additional_modifications_markdown,
                                                  inp_additional_modifications,
                                                  submit_upload_of_additional_modifications_button,
                                                  yes_additional_modifications_button,
                                                  no_additional_modification_button
                                                  ])

    submit_upload_of_additional_modifications_button.click(submit_upload_of_additional_modifications_button_fn,
                                                           [inp_additional_modifications,
                                                            unidentified_names_df_state],
                                                           [additional_modifications_submitted_markdown,
                                                            additional_modifications_submitted_df,
                                                            confirm_additional_modifications_button,
                                                            cancel_additional_modifications_button])
    cancel_additional_modifications_button.click(cancel_additional_modifications_button_fn,
                                                None,
                                                [additional_modifications_df_state,
                                                 submit_additional_modifications_markdown,
                                                 inp_additional_modifications,
                                                 submit_upload_of_additional_modifications_button,
                                                 additional_modifications_submitted_markdown,
                                                 additional_modifications_submitted_df,
                                                 no_additional_modification_button,
                                                 yes_additional_modifications_button,
                                                 confirm_additional_modifications_button,
                                                 cancel_additional_modifications_button])
    

    confirm_additional_modifications_markdown = gr.Markdown(visible=False)

    final_result_of_corrected_known_upcs_markdown = gr.Markdown(visible=False)
    final_df = gr.DataFrame(visible=False)
    final_csv = gr.File(label="Final CSV - please download",visible=False)    
    UPCs_to_add_to_truth_message = """
    # UPCs to be added to truth
    
    ### Download the CSV above before clicking on **"Add these UPCs to the ground truth"** as this will reload the app.

    Please, also check these rows before confirming - these UPCs will then be added to the ground truth and used for the next data cleansing sessions.

    If you do not agree with this table, start the session over and correct the UPCs.

    If after doing that, you still do not agree with the table, please contact me via email at: augustin.destaff@gmail.com
    """
    UPCs_to_add_to_truth_mardown = gr.Markdown(value=UPCs_to_add_to_truth_message,visible=False)
    UPCs_to_add_to_truth_df = gr.DataFrame(visible=False)
    UPCs_to_add_to_truth_button = gr.Button('Add these UPCs to the ground truth',visible=False)


    no_additional_modification_button.click(no_additional_modification_button_fn,
                                            [identified_upcs_df_state,
                                             identified_names_df_state,
                                             additional_modifications_df_state,
                                             unidentified_names_df_state,
                                             add_to_truth_df_state],
                                             [unidentified_names_df_state,
                                              final_result_of_corrected_known_upcs_markdown,
                                              no_additional_modification_button,
                                              yes_additional_modifications_button,
                                              final_df,
                                              final_csv,
                                              UPCs_to_add_to_truth_mardown,
                                              UPCs_to_add_to_truth_df,
                                              UPCs_to_add_to_truth_button])

    confirm_additional_modifications_button.click(confirm_additional_modifications_button_fn,
                                                  [inp_additional_modifications,
                                                  unidentified_names_df_state,
                                                  identified_upcs_df_state,
                                                  identified_names_df_state,
                                                  truth,
                                                  add_to_truth_df_state],
                                                  [identified_upcs_df_state,
                                                   additional_modifications_df_state,
                                                   unidentified_names_df_state,
                                                   confirm_additional_modifications_button,
                                                   cancel_additional_modifications_button,
                                                   confirm_additional_modifications_markdown,
                                                   final_result_of_corrected_known_upcs_markdown,
                                                   final_df,
                                                   final_csv,
                                                   add_to_truth_df_state,
                                                   UPCs_to_add_to_truth_mardown,
                                                   UPCs_to_add_to_truth_df,
                                                   UPCs_to_add_to_truth_button]
                                                   )


    UPCs_to_add_to_truth_button.click(click_UPCs_to_add_to_truth_button_fn,
                                      [truth,add_to_truth_df_state],
                                      None)

    input_csv.change(csv_input_cleared_fn,None,[upc_message,
                                                result_of_corrected_known_upcs_markdown,
                                                name_modifications_proposition_df,
                                                name_modifications_proposition_csv,
                                                accept_all_changes_button,
                                                make_modifications_to_the_names_button,
                                                submit_your_correction_markdown,
                                                inp_correction,
                                                accept_all_changes_markdown,
                                                unidentified_names_markdown,
                                                unidentified_names_markdown,
                                                unidentified_names_df,
                                                unidentified_names_csv,
                                                no_additional_modification_button,
                                                yes_additional_modifications_button,
                                                submit_input_csv_button,
                                                final_result_of_corrected_known_upcs_markdown,
                                                final_df,
                                                final_csv,
                                                submit_own_correction_for_unknown_upcs_identified_names_button])
    
    
if __name__ == "__main__":
    interface.launch()