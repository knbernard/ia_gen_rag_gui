import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from pydantic import BaseModel, Field
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion"
]


def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://fr.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAG_project/0.0.1 (contact@datascientist.fr)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


def get_documents():

    # Récupère les documents
    docs = []
    for page_title in PAGES:
        content = get_wikipedia_page(page_title)
        docs.append(Document(
            page_content=content,
            metadata={
                "title": page_title,
                "url": f"https://fr.wikipedia.org/wiki/{page_title}",
                "source": "Wikipedia"
            }
        ))

    # Découpage en chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250
    )
    return text_splitter.split_documents(docs)


def get_retriever():

    # Construction du vector store
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=OpenAIEmbeddings()
    )

    # Check if the vector store is empty
    if len(vectorstore.get(limit=1)['ids']) == 0:
        with st.spinner('Loading documents...'):
            docs = get_documents()
            vectorstore.add_documents(docs)

    # Document retriever
    return vectorstore.as_retriever(top_k=10)

def get_variant_queries(original_query):
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    class QueriesStructure(BaseModel):
        queries: List[str] = Field(
            ...,
            description="List of queries to generate different perspectives on the original query."
        )

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = model.with_structured_output(QueriesStructure)

    generate_queries = (
        prompt_perspectives
        | structured_llm
    )

    return generate_queries.invoke(original_query).queries

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """

    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    sorted_docs = sorted(fused_scores.items(),
                         key=lambda x: x[1], reverse=True)
    return [loads(doc) for doc, _ in sorted_docs]


def format_documents(docs):
    result = ""
    for doc in docs:
        result += "---\n"
        result += f"Lien source : {doc.metadata['url']}\n"
        result += f"Contenu : {doc.page_content}\n"
        result += "---\n"
    return result

def rag_stream(query, retriever):
    # Prompt
    template = """Réponds à la question en utilisant uniquement le contexte donné ci-dessous.
    Indique à la fin de ta réponse, les liens vers les documents utilisés.
    \"\"\"
    {context}
    \"\"\"
    
    Exemple de réponse :
    \"\"\"
    Gemini est un LLM (grand modèle de langage) car il utilise le réseau de neurones du modèle PaLM 2 et l'architecture « Google Transformer », qui sont des caractéristiques typiques des grands modèles de langage. Ces modèles possèdent un grand nombre de paramètres et sont capables de traiter et de générer du texte, ainsi que d'autres types de données, ce qui est le cas de Gemini. De plus, Gemini est capable de générer et de combiner des objets sonores, visuels et textuels, ce qui le rapproche d'une intelligence artificielle générale. 

    Sources : 
    - https://fr.wikipedia.org/wiki/Google_Gemini
    \"\"\"
    
    Question: {query}
    """

    def retrieve_documents(query):

        retrieval_chain_rag_fusion = (
            get_variant_queries |
            retriever.map() |
            reciprocal_rank_fusion
        )

        docs = retrieval_chain_rag_fusion.invoke(query)
        return docs[:5]

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    rag_chain = (
        {
            "context": RunnablePassthrough() | retrieve_documents | format_documents,
            "query": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    for chunk in rag_chain.stream(query):
        yield chunk

def streamlit_gui() :
#    st.title('RAG : IA Générative avec Wikipédia')
    st.title('Assistant IA Générative')

    # Setup message with first message
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Bonjour, je suis un assistant virtuel. Posez-moi des questions sur l'IA générative."
        }]

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("C'est quoi l'IA générative ?"):
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        retriever = get_retriever()

        with st.chat_message("assistant"):
            stream = rag_stream(prompt, retriever)
            response = st.write_stream(stream)

        ai_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(ai_message)


if __name__ == "__main__":
    streamlit_gui()
#    print('Building retriever...')
#    retriever = get_retriever()
#
#    try:
#        while True:
#            print('-' * 50)
#            print('Posez une question :')
#            question = input('> ')
#            print()
#            stream = rag_stream(question, retriever)
#            for chunk in stream:
#                print(chunk, end="")
#            print('\n')
#
#    except KeyboardInterrupt:
#        print("\nExiting...")
