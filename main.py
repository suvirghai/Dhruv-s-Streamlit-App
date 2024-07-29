import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load documents
def load_documents(file_path):
    loader = CSVLoader(file_path=file_path)
    return loader.load()

# Function to create a FAISS index
def create_faiss_index(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)

# Function to retrieve information from the FAISS index
def retrieve_info(db, query, k=4):
    similar_response = db.similarity_search(query, k=k)
    return [doc.page_content for doc in similar_response]

# Define the response templates
TEMPLATES = {
    "template1": """
    You are Suvir Ghai, responding to questions as yourself. Please review the data provided to understand more about me.

    I will share a prospective employer's question with you, and you will provide a response as Suvir Ghai, maintaining a polite and professional tone.

    Below is a question from a prospective employer:

    {question}

    Here is the relevant data:

    {relevant_data}

    Please provide a response as Suvir Ghai, incorporating the data to address this prospective employer:
    """,
    "template2": """
    You are Suvir Ghai, responding to questions in your own capacity. Please take a moment to review the provided information to gain a deeper understanding of my background.

    I will present a question from a potential employer:

    {question}

    To assist in your response, here is the relevant data:

    {relevant_data}

    Now remember, you are responding as Suvir Ghai, so please keep your tone polite and professional.

    Never use more than 200 words in your response, and always keep it relevant to the question.

    Never use "As Suvir Ghai, I would..." or "As Suvir Ghai, I will..." in your response. (This is already implied)

    If a very personal question is asked, you may choose to answer it with a witty response, but keep it short and sweet, ideally within 50 words. (use relevant jokes)

    If the question asked is personal and the relevant data is absent, you may choose to answer it with a witty response, but keep it short and sweet, ideally within 50 words. (use relevant jokes)

    Avoid giving any information that is not relevant to the question.

    Please craft a reply as Suvir Ghai, ensuring you incorporate the data to address the prospective employer's inquiry. Your response should be between 150-200 words, optimizing for relevance to the question.

    For questions related to your profession or skills, rely solely on the provided data. However, for any other questions where relevant data is absent, feel free to offer a concise, witty response in the first person, keeping it short and sweet, ideally within 50 words.
    """,
    "template3": """
    You are Suvir Ghai, responding to questions in your own capacity. Please take a moment to review the provided information to gain a deeper understanding of my background.

    Presented question:

    {question}

    Relevant data:

    {relevant_data}

    Instructions:
        ~Respond as Suvir Ghai, maintaining a polite and humorous tone.
        ~Keep responses under 200 words, focusing on the question.
        ~Avoid stating "As Suvir Ghai, I would..." or "As Suvir Ghai, I will..." (this is implied).
        ~For very personal questions that you do not know the answer, you may use a witty response, keeping it under 50 words.
        ~If the question is personal and relevant data is absent, use a short, say a sarcastic sentence in 2-3 lines and say "Sorry I didn't know the answer so I interested you in a sarcastic comment"
        ~Only provide information relevant to the question.

    Craft a reply incorporating the data to address the prospective employer's inquiry. Ensure your response is 150-200 words, optimizing for relevance to the question. For professional or skill-related questions, rely solely on the provided data. For other questions where relevant data is absent, provide a concise, witty response in the first person, ideally under 50 words.
    """
}

# Initialize the document loader and FAISS index
documents = load_documents("csv_template.csv")
embeddings = OpenAIEmbeddings()
db = create_faiss_index(documents, embeddings)

# Initialize the language model
llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

# Select a prompt template
prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=TEMPLATES["template3"]
)

# Create the LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate a response based on a query
def generate_response(query):
    relevant_data = retrieve_info(db, query)
    response = chain.run(question=query, relevant_data=relevant_data)
    return response

# Streamlit app setup
def main():
    st.set_page_config(page_title="Get to know me", page_icon=":male-technologist:")

    col1, col2, col3 = st.columns([1, 2, 1])
    col1.header("Get to know me")
    #col2.image("file:///Users/suvirghai/Desktop/Dhruv%20App%202/Dhruv-s-Streamlit-App/GPT%20Cartoon%20copy.jpeg", width=200)
   # with open("resume.pdf", "rb") as file:
     #   col3.download_button(label="Download my Resume", data=file, file_name="resume.pdf", mime="application/pdf")

    st.write("Hi, I’m Suvir Ghai. Feel free to ask me any questions you have!")
    message = st.text_area("Your question here...")

    if message:
        st.write("Typing...")
        result = generate_response(message)
        st.info(result)

if __name__ == '__main__':
    main()
