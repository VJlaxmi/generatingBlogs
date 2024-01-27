import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from fpdf import FPDF
import os
import base64

def getLLamaresponse(input_text,no_words,blog_style):

    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    return response

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'),index=0)
    
submit=st.button("Generate")


def save_response_as_pdf(response, filename="response.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    pdf.multi_cell(0, 10, response)
    pdf.output(filename)
    return filename

def create_download_link(filename):
    with open(filename, "rb") as file:
        base64_file = base64.b64encode(file.read()).decode()
    href = f'<a href="data:file/pdf;base64,{base64_file}" download="{filename}">Download PDF</a>'
    return href

## Final response
if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    st.write(response)
    
    # Save the response to a PDF
    pdf_filename = save_response_as_pdf(response)

    
    # Create a download link and display it
    download_link = create_download_link(pdf_filename)
    st.markdown(download_link, unsafe_allow_html=True)

    # Optionally, delete the PDF file if you don't want to keep it on the server
    os.remove(pdf_filename)
