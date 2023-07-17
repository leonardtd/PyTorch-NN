# Load tools

from langchain.agents import Tool
from langchain.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch


def get_tools(llm) -> list:
    embeddings = OpenAIEmbeddings()

    # Company
    company_file = './docs/minesafe.txt'
    company_docs = TextLoader(company_file).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(company_docs)

    company_db = DocArrayInMemorySearch.from_documents(
        texts,
        embeddings,
        collection_name="company_description"
    )

    company_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=company_db.as_retriever()
    )

    # Products
    products_file = './docs/products.csv'
    products_docs = CSVLoader(file_path=products_file).load()

    products_db = DocArrayInMemorySearch.from_documents(
        products_docs,
        embeddings,
        collection_name="products_descriptions",
    )

    products_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=products_db.as_retriever()
    )

    tools = [
        Tool(
            name="Company_QA",
            func=company_qa.run,
            description="useful for when you need to answer questions about the company itself. information includes a summary about the business and a mission, vision and core values for the company.",
        ),
        Tool(
            name="Products_QA",
            func=products_qa.run,
            description="useful for when you need to answer questions about the products that the company offers. information provided in this document is item name, item description and unit price.",
        ),
    ]

    return tools
