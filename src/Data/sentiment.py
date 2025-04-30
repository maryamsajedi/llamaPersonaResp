import re
import json
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

topics = [
    "Arts & Culture",
    "Business & Entrepreneurs",
    "Celebrity & Pop Culture",
    "Diaries & Daily Life",
    "Family",
    "Fashion & Style",
    "Film, TV & Video",
    "Fitness & Health",
    "Food & Dining",
    "Gaming",
    "Learning & Educational",
    "Music",
    "News & Social Concern",
    "Other Hobbies",
    "Relationships",
    "Science & Technology",
    "Sports",
    "Travel & Adventure",
    "Youth & Student Life"
]


base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "user_messages.pkl")

with open(data_path, 'rb') as file:
    user_messages = pickle.load(file)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

BATCH_SIZE = 8  

def call_llama(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def clean_sentiment(text):
    text = text.split("The Sentiment is:")[-1].strip().lower()
    sentiment = re.sub(r'[^a-z]', '', text)
    return sentiment

def clean_topic(text):
    text = text.split("The Topic is:")[-1].strip().lower()
    text = re.sub(r'[^a-z ]', '', text) 
    
    for topic in topics:
        if topic.lower() in text:
            return topic
    return "Other"  

user_sentiments = {user: {'positive': 0, 'negative': 0, 'neutral': 0} for user in user_messages}
user_topics = {user: [] for user in user_messages}
user_message_pairs = [(user, msg) for user, messages in user_messages.items() for msg in messages]

with tqdm(total=len(user_message_pairs), desc="Processing", ncols=100) as pbar:
    for i in range(0, len(user_message_pairs), BATCH_SIZE):
        batch = user_message_pairs[i:i + BATCH_SIZE]
        prompts_sentiment = [
            f'''Classify the sentiment as one of the following: Positive, Negative, or Neutral.
                User Message: "{msg}"
                The Sentiment is:''' for _, msg in batch
        ]
        
        prompts_topic = [
            f'''Classify the topic of the following user message into {", ".join(topics)}.
                User Message: "{msg}"
                The Topic is:''' for _, msg in batch
        ]

        inputs_sentiment = tokenizer(prompts_sentiment, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        inputs_topic = tokenizer(prompts_topic, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs_sentiment = model.generate(
                input_ids=inputs_sentiment["input_ids"],
                attention_mask=inputs_sentiment["attention_mask"],
                max_length=500,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

            outputs_topic = model.generate(
                input_ids=inputs_topic["input_ids"],
                attention_mask=inputs_topic["attention_mask"],
                max_length=500,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_sentiment = tokenizer.batch_decode(outputs_sentiment, skip_special_tokens=True)
        decoded_topic = tokenizer.batch_decode(outputs_topic, skip_special_tokens=True)

        for (user, _), sentiment, topic in zip(batch, decoded_sentiment, decoded_topic):
            clean_sentiment_value = clean_sentiment(sentiment)
            clean_topic_value = clean_topic(topic)

            if clean_sentiment_value in user_sentiments[user]:
                user_sentiments[user][clean_sentiment_value] += 1

            user_topics[user].append(clean_topic_value)

        pbar.update(1)

final_result = []
for user, sentiment_counts in user_sentiments.items():
    final_result.append({
        'user': user,
        'sentiment': sentiment_counts,
        'topics': list(set(user_topics[user]))  
    })

with open('results_with_topics.json', 'w') as f:
    json.dump(final_result, f, indent=4)

print("Done! Results saved to 'results_with_topics.json'")
