# Week-4_Build_NLP_Chatbot_Project_using_LangChain_Gradio_FastAPI


**Knowledge-Based Chatbot using LangChain and Groq API:** Developed an intelligent, context-aware chatbot system that leverages LangChain, Gradio, and FastAPI. 

Step-by-step workflow of your chat system:

  1. User opens browser → goes to /chat.
  
  2. User types a question → hits Send.
  
  3. **Gradio** calls chat_gradio(user_message).
  
  4. chat_gradio sends the question to LangChain’s ContextualChatSystem.
  
  5. **LangChain:**
  
       -Selector model finds the most relevant context.
  
       -Generator model produces a concise answer using that context.
  
  6. Response is returned to chat_gradio → displayed in chat history with timestamp and response time.
  
  7. **FastAPI** serves this interaction over the web in real-time.
