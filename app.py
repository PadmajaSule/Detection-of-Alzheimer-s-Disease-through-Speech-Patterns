import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential, Model, load_model
df = pd.read_csv('/content/drive/MyDrive/Mini_project_entities/new_prepro_dataset_alzimer.csv')
columns_to_copy = [
    'File',
    'Age(Month)',
    'Group',
    'mor_Utts',
    'mor_words',
    'mor_syllables',
    'words_min',
    'syllables_min',
    '%_WWR',
    '%_mono-WWR',
    '%_WWR-RU',
    '%_mono-WWR-RU',
    'Mean_RU',
    '%_Phonological_fragment',
    '%_Phrase_repetitions',
    '%_Word_revisions',
    '%_Phrase_revisions',
    '%_Pauses',
    '%_Filled_pauses',
    '%_TD',
    '%_SLD',
    '%_Total_(SLD+TD)',
    'SLD_Ratio',
    'Content_all_words_ratio',
    'Function_all_words_ratio',
    'Weighted_SLD',
    'MLU_Words',
    'MLU_Morphemes',
    'FREQ_types',
    'FREQ_tokens',
    'Verbs_Utt',
    'TD_Words',
    'TD_Words_Time',
    'Word_Errors',
    'Utt_Errors',
    'retracing',
    'repetition',
    'DSS_Utts',
    'mor_Words',
    '*-PRESP',
    'in',
    'irr-PAST',
    'u-cop',
    'det:art',
    'irr-3S',
    'u-aux',
]
df = df[columns_to_copy].copy()
df['Group'] = df['Group'].replace('Control',0)
df['Group'] = df['Group'].replace('MCI',1)
df['Group'] = df['Group'].replace('Memory',2)
df['Group'] = df['Group'].replace('Vascular',3)
df['Group'] = df['Group'].replace('PossibleAD',4)
df['Group'] = df['Group'].replace('ProbableAD',5)

model = load_model('/content/drive/MyDrive/Mini_project_entities/model3.h5')

def get_attributes_for_file(df, file_name):
    # Select the row corresponding to the given file name
    row = df.loc[df['File'] == file_name]

    # Check if the file name exists in the DataFrame
    if len(row) == 0:
        print(f"File '{file_name}' not found.")
        return None

    # Extract values of all attributes present in that row and convert to NumPy array
    attributes_array = row.iloc[0].values
    attributes_array = np.delete(attributes_array, [0])  # Remove unwanted elements

    return attributes_array

# Define a function to perform prediction
def predict(input_text):
    try:
        attributes = get_attributes_for_file(df, input_text)
        attribute = attributes[1]
        attributes = np.delete(attributes, [1])

        test_input = np.array(attributes, dtype=np.float32)
        test_input = test_input.reshape(1, -1)
        class_names = ['Control','MCI','Memory','Vascular','PossibleAD','ProbableAD']
        result1 = model.predict(test_input)
        predicted_class_index = result1
        predicted_class = class_names[predicted_class_index]

        return predicted_class
    except ValueError:
        return "Error: Invalid input."

def main():
    st.title("Detection of Alzheimer's Disease through Speech Patterns")

    # Get input string from user
    input_string = st.text_input("Enter the file name:")

    # Process input string using your deep learning model
    if st.button("Process"):
        if input_string:
            processed_output = predict(input_string)
            st.write("Processed Output:")
            st.write(processed_output)
            if(processed_output=="ProbableAD"):
                st.write("The model predicts ProbableAD for you, it suggests that your cognitive health may be indicative of Alzheimer's disease based on the input data and the model's analysis. This prediction should prompt you to seek further evaluation and consultation with healthcare professionals for appropriate diagnosis, treatment, and management. It's important to understand that this prediction is based on statistical analysis and may not be definitive, but early detection and intervention can be crucial for managing Alzheimer's disease effectively.")
            elif(processed_output=="MCI"):
                st.write("The model predicts MCI (Mild Cognitive Impairment) for you, it indicates that there may be a slight decline in your cognitive abilities beyond what is expected for your age. While this prediction is not a diagnosis, it suggests the need for further evaluation by healthcare professionals to assess your cognitive health and determine appropriate management strategies. Early detection of MCI can be important for addressing underlying causes and implementing interventions to potentially slow cognitive decline or prevent progression to more severe forms of dementia.")
            elif(processed_output=="Memory"):
                st.write("The model predicts Memory for you, it suggests that there may be specific issues related to memory function based on the input data and the model's analysis. While this prediction does not provide a diagnosis, it indicates the importance of discussing any memory concerns with healthcare professionals. Further evaluation and assessment of memory function may be necessary to understand the underlying causes and develop appropriate management strategies. Taking proactive steps to address memory issues, such as lifestyle modifications and cognitive interventions, can be beneficial for maintaining cognitive health.")
            elif(processed_output=="Vascular"):
                st.write("The model predicts Vascular for you, it indicates that there may be significant damage to the brain's blood vessels affecting your cognitive health. This prediction suggests the need for further evaluation by healthcare professionals to assess vascular health and its impact on cognitive function. Vascular dementia is often associated with conditions affecting blood flow to the brain, such as strokes or cardiovascular disease. Early detection and management of vascular risk factors, such as hypertension and diabetes, are crucial for preventing further cognitive decline and optimizing overall health outcomes.")
            elif(processed_output=="PossibleAD"):
                st.write("The model predicts PossibleAD for you, it suggests that there are symptoms or indicators present that could potentially be associated with Alzheimer's disease (AD). While this prediction is not a definitive diagnosis, it underscores the importance of seeking further evaluation and consultation with healthcare professionals. Additional assessments, such as cognitive testing and medical imaging, may be necessary to confirm or rule out AD and determine appropriate management strategies. Early detection and intervention are critical for addressing cognitive changes and implementing strategies to support cognitive health and overall well-being.")
            elif(processed_output=="Control"):
                st.write("The model predicts Control for a patient, it means their cognitive health is likely within the normal range, indicating no signs of cognitive impairment or dementia. Patients can feel reassured but should continue monitoring their cognitive health and maintain a healthy lifestyle. Consulting with healthcare professionals for personalized guidance is recommended.")
        
        else:
            st.warning("Please enter the file name.")

if __name__ == "__main__":
    main()
