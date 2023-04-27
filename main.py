import pandas as pd
import streamlit as st
import mindsdb_sdk

LOGIN = 'andrey@mindsdb.com'
PASSWORD = st.secrets["password"]


st.write("## English to French translation")
st.write("Mindsdb python SDK example application: translate from English to French using mindsdb with t5-base hugginface model")

st.write("[Example source code](https://github.com/mindsdb/python-example)")
st.write("[Mindsdb python SDK documentation](https://mindsdb.github.io/mindsdb_python_sdk/)")


@st.cache_resource
def get_model():
    loading = st.empty()
    loading.markdown('## connecting to mindsdb cloud server...')

    server = mindsdb_sdk.connect(login=LOGIN, password=PASSWORD)
    project = server.get_project()
    _model = project.get_model('hf_t5_en_fr')  # hf_t5_en_fr is the name of model in mindsdb

    loading.markdown("")
    return _model


model = get_model()

source = st.text_area(label="Input text:")

button = st.button(label='Translate')

translated = st.code("", language="markdown")

if button:
    translated.markdown('*translating...*')

    # convert text to dataframe
    df_in = pd.DataFrame([{'text': source}])
    df_out = model.predict(df_in)

    # get translated text from output dataframe
    text = df_out.PRED[0]
    translated.markdown(text)
