import random
import string
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load conversational model from transformers
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Example conversational data for similarity-based responses (optional if you want to mix rule-based and generative)
corpus = [
    # Greetings and Introduction
    "Hello! How can I assist you today?",
    "Hi, what can I do for you?",
    "Good day! How can I help you?",
    "Hey there, what’s on your mind?",
    # Basic Information
    "I’m ChatBot, here to help with any questions or concerns.",
    "Can you tell me about yourself?",
    "I’m a chatbot created to assist with general queries and tasks. Ask me anything!",
    # Well-being
    "How are you doing today?",
    "I’m just a bot, but thanks for asking! How about you?",
    "I’m functioning well! How can I assist you today?",
    # Time and Date
    "What time is it right now?",
    "I can’t tell time exactly, but you can always check your device’s clock.",
    "What day is it today?",
    "You can check your calendar for today’s date.",
    # Weather
    "What’s the weather like today?",
    "I don’t have weather information, but you can check online weather services."
    # Personal Topics
    "How can I be more productive?",
    "Set clear goals, prioritize tasks, and take breaks when needed for the best productivity!",
    "What’s a good way to stay healthy?",
    "Eat balanced meals, exercise regularly, and get enough sleep!",
    "Can you motivate me?",
    "You got this! Focus on one step at a time and keep moving forward.",
    # Travel
    "Where should I go for vacation?",
    "How about a nature retreat, a beach holiday, or exploring a new city? It depends on your mood!",
    "What’s the best way to travel?",
    "It depends on the distance. For short trips, a car might be best; for longer ones, maybe a plane or train.",
    # Technology and Help
    "How can I reset my password?",
    "Go to the password reset page on the service, enter your email, and follow the instructions.",
    "My computer is running slow. What can I do?",
    "Try restarting your computer, closing unnecessary programs, or checking for software updates.",
    # Random Facts
    "Tell me a fun fact.",
    "Did you know? Honey never spoils, and it’s edible even after thousands of years!",
    "What’s the capital of France?",
    "The capital of France is Paris.",
    # Casual Conversations
    "I don’t have preferences, but I’d imagine blue is quite calming!",
    "Do you have any hobbies?",
    "I enjoy helping people with their questions, that's my main job!",
    # Learning and Education
    "How do I learn a new language?",
    "Start with the basics, practice daily, and try using apps like Duolingo or watching shows in that language.",
    "Sure! What math problem do you have in mind?",
    # Farewell
    "Goodbye!",
    "Goodbye! Feel free to return if you need more help.",
    "See you later!",
    "Take care! I’ll be here if you have more questions."
]


# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

# Preprocessing sentences using spaCy
def preprocess_sentence(text):
    doc = nlp(text.lower())  # Tokenization and lowercasing
    tokens = [token.lemma_ for token in doc if token.is_alpha]  # Lemmatization, removing non-alphabetic tokens
    return tokens

# Update get_response_similarity to use preprocess_sentence
def get_response_similarity(user_input):
    # Preprocess both corpus and user input using the preprocess_sentence function
    processed_corpus = [" ".join(preprocess_sentence(sentence)) for sentence in corpus]
    processed_input = " ".join(preprocess_sentence(user_input))
    
    processed_corpus.append(processed_input)  # Append preprocessed user input to the corpus
    
    count_vec = CountVectorizer().fit_transform(processed_corpus)  # Vectorize the corpus
    similarity_scores = cosine_similarity(count_vec[-1], count_vec)  # Compute similarity scores
    idx = similarity_scores.argsort()[0][-2]  # Get the most similar sentence index
    
    flat_scores = similarity_scores.flatten()
    flat_scores.sort()

    if flat_scores[-2] == 0:
        return "Sorry, I don't understand."
    else:
        return corpus[idx]

    
# Using transformers-based model for generating responses
def get_response_transformers(user_input):
    # Tokenize input and append end of string token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Create attention mask based on non-padding tokens (1 for actual tokens, 0 for padding tokens)
    attention_mask = torch.ones_like(new_user_input_ids)

    # Generate response from the model
    chat_history_ids = model.generate(
        new_user_input_ids, 
        attention_mask=attention_mask,  # Pass attention mask
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response and return it
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


# Main chatbot function
def chatbot():
    print("ChatBot: Hello! Type 'bye' to exit.")
    while True:
        user_input = input("You: ").lower()
        if user_input == 'bye':
            print("ChatBot: Goodbye!")
            break
        else:
            # First, try using similarity-based response
            similarity_response = get_response_similarity(user_input)
            
            # If the similarity-based response is "Sorry, I don't understand.", fallback to transformers
            if similarity_response == "Sorry, I don't understand.":
                response = get_response_transformers(user_input)
            else:
                response = similarity_response

            print(f"ChatBot: {response}")


# Entry point for the chatbot
if __name__ == "__main__":
    chatbot()
