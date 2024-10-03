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
    known_upcs_df_state = gr.State()
    identified_names_df_state = gr.State()
    unidentified_names_df_state = gr.State()
    additional_modifications_df_state = gr.State()
    add_to_truth_df_state = gr.State()
    upcs_to_modify_df_state = gr.State()
    upcs_to_delete_df_state = gr.State()
    
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
        modify_the_truth_button = gr.Button(value='Modify the current truth', icon='wrong_icon.svg', visible=False)
    
    # IF MODIFICATIONS TO THE TRUTH
    modify_the_truth_button_markdown = gr.Markdown(visible=False)
    truth_correction_csv = gr.File(label="Upload your correction to the truth", file_types=['csv'], visible=False)
    submit_modifications_to_the_truth_button = gr.Button("Submit", visible=False)

    # AFTER SUBMITTING THE MODIFICATIONS TO THE TRUTH
    modified_truth_markdown = gr.Markdown(visible=False)

    # Unidentified UPCs
    name_identification_markdown = gr.Markdown(visible=False)
    name_modifications_proposition_df = gr.Dataframe(label="Name modifications",visible=False)
    unidentified_names_df = gr.Dataframe(label="Unidentified UPCs and names",visible=False)
    name_modifications_proposition_csv = gr.File(label="Download the name modification propositions file",visible=False)
    
    ## ACCEPT OR MAKE YOUR OWN MODIFICATIONS
    with gr.Row():
        accept_all_changes_button = gr.Button(value='Accept all suggested modifications', icon='right_icon.svg', visible=False)
        make_modifications_to_the_names_button = gr.Button(value='Submit your own correction', icon='wrong_icon.svg', visible=False)
        make_no_modifications_to_names_button = gr.Button(value='Reject suggested modifications and skip step',icon='fast-forward.png', visible=False)

    ## FIRST SUBMIT CLICK
    submit_input_csv_button.click(fn=identify_known_upcs, 
                                  inputs=input_csv, 
                                  outputs= [
                                      input_csv_df_state,
                                      upc_message,
                                      known_upcs_df_state,
                                      identified_upcs_df,
                                      unidentified_names_df_state,
                                      proceed_with_this_truth_button,
                                      modify_the_truth_button,
                                      submit_input_csv_button,
                                      identified_upcs_csv                                      
                                  ])
    
    proceed_with_this_truth_button.click(fn=proceed_with_identified_upcs_fn, 
                                         inputs = unidentified_names_df_state,
                                         outputs= [
                                             name_identification_markdown,
                                             name_modifications_proposition_df,
                                             identified_names_df_state,
                                             unidentified_names_df_state,
                                             name_modifications_proposition_csv,
                                             proceed_with_this_truth_button,
                                             modify_the_truth_button,
                                             accept_all_changes_button,
                                             make_modifications_to_the_names_button,
                                             make_no_modifications_to_names_button
                                             ])
    
    modify_the_truth_button.click(fn=modify_the_truth_button_fn,
                                   outputs=[
                                       modify_the_truth_button_markdown,
                                       truth_correction_csv,
                                       submit_modifications_to_the_truth_button
                                   ])
    
    submit_modifications_to_the_truth_button.click(fn=submit_modifications_to_the_truth_button_fn,
                                                   inputs= [
                                                       input_csv_df_state,
                                                       truth_correction_csv,
                                                       known_upcs_df_state,
                                                       unidentified_names_df_state 
                                                       ],
                                                   outputs=[
                                                       known_upcs_df_state,
                                                       modified_truth_markdown,
                                                       upcs_to_modify_df_state,
                                                       upcs_to_delete_df_state,
                                                       name_identification_markdown,
                                                       name_modifications_proposition_df,
                                                       name_modifications_proposition_csv,
                                                       proceed_with_this_truth_button,
                                                       modify_the_truth_button,
                                                       accept_all_changes_button,
                                                       make_modifications_to_the_names_button,
                                                       make_no_modifications_to_names_button,
                                                       submit_modifications_to_the_truth_button,
                                                       unidentified_names_df_state,
                                                       identified_names_df_state
                                                       ])

    
    accept_all_changes_markdown = gr.Markdown(visible=False)
    make_no_modifications_to_names_markdown = gr.Markdown(visible=False)
    
    ## OWN CORRECTION
    submit_your_correction_markdown = gr.Markdown(visible=False)
    inp_correction = gr.File(label="Upload your own correction as a CSV file", file_types=['csv'],visible=False)
    submit_own_correction_for_unknown_upcs_identified_names_button = gr.Button("Submit modifications",visible=False)

    own_name_correction_markdown = gr.Markdown(visible=False)
    own_name_correction_df = gr.DataFrame(visible=False)
    with gr.Row():
        confirm_own_correction_button = gr.Button('Confirm my modifications',icon='right_icon.svg',visible=False)
        cancel_own_correction_button = gr.Button('Cancel my modifications',icon='wrong_icon.svg',visible=False)


    submit_own_correction_for_unknown_upcs_identified_names_button.click(fn = submit_own_correction_for_unknown_upcs_identified_names_fn,
                                                                         inputs= inp_correction,
                                                                         outputs= [
                                                                             own_name_correction_df,
                                                                             own_name_correction_markdown,
                                                                             confirm_own_correction_button,
                                                                             cancel_own_correction_button
                                                                         ])
    
    click_confirm_own_correction_markdown = gr.Markdown(visible=False)
     
    unidentified_names_markdown = gr.Markdown(visible=False)
    unidentified_names_df = gr.Dataframe(label='Unknown UPCs with unidentified names',visible=False,interactive=False)
    
    unidentified_names_csv = gr.File(label="Download the unknown UPCs and unidentified names file",visible=False)
    
    with gr.Row():
        no_additional_modification_button = gr.Button(value='No additional modifications', icon='thumb_up.svg', visible=False)
        yes_additional_modifications_button = gr.Button(value='Submit additional modifications', icon='upload.svg', visible=False)

    
    confirm_own_correction_button.click(click_confirm_own_correction_fn,
                                        [inp_correction,
                                         known_upcs_df_state,
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
                                        make_modifications_to_the_names_button,
                                        make_no_modifications_to_names_button
                                        ])



    accept_all_changes_button.click(accept_all_changes_fn, 
                                    [identified_names_df_state,
                                     unidentified_names_df_state,
                                     truth], 
                                    [identified_names_df_state,
                                     accept_all_changes_button,
                                     make_modifications_to_the_names_button,
                                     make_no_modifications_to_names_button,
                                     unidentified_names_markdown,
                                     unidentified_names_df_state,
                                     unidentified_names_df,
                                     unidentified_names_csv,
                                     no_additional_modification_button,
                                     yes_additional_modifications_button,
                                     add_to_truth_df_state])
    
    make_modifications_to_the_names_button.click(fn = click_own_name_correction_fn,
                                                 inputs = None,
                                                 outputs= [
                                                     accept_all_changes_button,
                                                     make_modifications_to_the_names_button,
                                                     make_no_modifications_to_names_button,
                                                     submit_your_correction_markdown,
                                                     inp_correction,
                                                     submit_own_correction_for_unknown_upcs_identified_names_button  
                                                 ])
    
    make_no_modifications_to_names_button.click(
        fn = make_no_modifications_to_names_button_fn,
        inputs = unidentified_names_df_state,
        outputs= [
            accept_all_changes_button,
            make_modifications_to_the_names_button,
            make_no_modifications_to_names_button,
            make_no_modifications_to_names_markdown,
            unidentified_names_markdown,
            unidentified_names_df,
            unidentified_names_csv,
            no_additional_modification_button,
            yes_additional_modifications_button
        ]
    )

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

    final_name_identification_markdown = gr.Markdown(visible=False)
    final_df = gr.DataFrame(visible=False)
    final_csv = gr.File(label="Final CSV - please download",visible=False)    
    
    # SUMMARIZE UPCOMING MODIFICATIONS
    upcoming_modifications_markdown = gr.Markdown(visible=False)

    # DELETING UPCs FROM THE TRUTH
    upcs_to_delete_from_truth_markdown = gr.Markdown(visible=False)
    upcs_to_delete_from_truth_df = gr.DataFrame(visible=False)


    # MODIFYING UPC INFORMATION IN THE TRUTH
    upcs_to_modify_in_truth_markdown = gr.Markdown(visible=False)
    upcs_to_modify_in_truth_df = gr.DataFrame(visible=False)

    
    # ADDING UPCs TO THE TRUTH
    upcs_to_add_to_truth_markdown = gr.Markdown(visible=False)
    upcs_to_add_to_truth_df = gr.DataFrame(visible=False)
    
    confirm_modifications_to_truth_button = gr.Button('Make these modifications to the ground truth',visible=False)


    no_additional_modification_button.click(fn=no_additional_modification_button_fn,
                                            inputs= [
                                                input_csv_df_state,
                                                known_upcs_df_state,
                                                identified_names_df_state,
                                                additional_modifications_df_state,
                                                unidentified_names_df_state,
                                                add_to_truth_df_state,
                                                upcs_to_delete_df_state,
                                                upcs_to_modify_df_state
                                            ],
                                             outputs=[
                                                 unidentified_names_df_state,
                                                 final_name_identification_markdown,
                                                 no_additional_modification_button,
                                                 yes_additional_modifications_button,
                                                 final_df,
                                                 final_csv,
                                                 upcoming_modifications_markdown,
                                                 upcs_to_delete_from_truth_markdown,
                                                 upcs_to_delete_from_truth_df,
                                                 upcs_to_modify_in_truth_markdown,
                                                 upcs_to_modify_in_truth_df,
                                                 upcs_to_add_to_truth_markdown,
                                                 upcs_to_add_to_truth_df,
                                                 confirm_modifications_to_truth_button   
                                             ])

    confirm_additional_modifications_button.click(fn= confirm_additional_modifications_button_fn,
                                                  inputs=[
                                                      input_csv_df_state,
                                                      inp_additional_modifications,
                                                      unidentified_names_df_state,
                                                      known_upcs_df_state,
                                                      identified_names_df_state,
                                                      truth,
                                                      add_to_truth_df_state,
                                                      upcs_to_delete_df_state,
                                                      upcs_to_modify_df_state
                                                  ],
                                                  outputs= [
                                                      known_upcs_df_state,
                                                      additional_modifications_df_state,
                                                      unidentified_names_df_state,

                                                      confirm_additional_modifications_button,
                                                      cancel_additional_modifications_button,
                                                      
                                                      confirm_additional_modifications_markdown,
                                                      final_name_identification_markdown,
                                                      final_df,
                                                      final_csv,
                                                      add_to_truth_df_state,

                                                      upcoming_modifications_markdown,
                                                      
                                                      upcs_to_delete_from_truth_markdown,
                                                      upcs_to_delete_from_truth_df,

                                                      upcs_to_modify_in_truth_markdown,
                                                      upcs_to_modify_in_truth_df,

                                                      upcs_to_add_to_truth_markdown,
                                                      upcs_to_add_to_truth_df,

                                                      confirm_modifications_to_truth_button 
                                                  ])


    confirm_modifications_to_truth_button.click(
        fn=click_UPCs_to_add_to_truth_button_fn,
        inputs=[truth, add_to_truth_df_state,upcs_to_modify_df_state,upcs_to_delete_df_state],
        outputs=None)

    input_csv.change(csv_input_cleared_fn,None,[upc_message,
                                                name_identification_markdown,
                                                name_modifications_proposition_df,
                                                name_modifications_proposition_csv,
                                                accept_all_changes_button,
                                                make_modifications_to_the_names_button,
                                                make_no_modifications_to_names_button,
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
                                                final_name_identification_markdown,
                                                final_df,
                                                final_csv,
                                                submit_own_correction_for_unknown_upcs_identified_names_button])
    
    
if __name__ == "__main__":
    interface.launch()