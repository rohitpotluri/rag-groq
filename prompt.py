"""
Prompt template for interacting with the GROQ model
"""

from langchain_core.prompts import ChatPromptTemplate

def get_prompt_template():
    """
    Returns a structured prompt template for generating accurate responses
    """
    return ChatPromptTemplate.from_template(
        """
        Answer the questions based only on the provided context, donot add your interpretations
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )
