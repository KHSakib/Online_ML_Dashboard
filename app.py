import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure necessary NLTK data is downloaded
nltk.download('vader_lexicon', quiet=True)

# Define the function to scrape data from Reddit and perform sentiment analysis
def scrape_reddit_data(subreddit_name, limit):
    user_agent = "Scraper 1.0 by /u/KH_Sakib6229"
    reddit = praw.Reddit(client_id="hcJtjwx5jWq3Qmb8jH-GPg",
                         client_secret="JNjCCvzvN2ze7kaljz5lPOe7V3StFA",
                         user_agent=user_agent)

    headlines = set()
    for submission in reddit.subreddit(subreddit_name).hot(limit=limit):
        headlines.add(submission.title)

    sia = SIA()
    results = []
    for line in headlines:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)

    df = pd.DataFrame.from_records(results)
    df['label'] = 'Neutral'
    df.loc[df['compound'] > 0.2, 'label'] = 'Positive'
    df.loc[df['compound'] < -0.2, 'label'] = 'Negative'
    
    return df[['headline', 'label']]

# Define the function to train and evaluate the selected model
def train_model(model_name, X_train, X_test, y_train, y_test):
    if model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'SVM':
        model = SVC(random_state=42)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return classification_report(y_test, predictions)

# Set up sidebar
st.sidebar.title('Reddit Sentiment Analysis')
subreddit_name = st.sidebar.selectbox('Select a subreddit:', ['SuicideWatch', 'stress', 'depression'])
limit = st.sidebar.slider('Select the number of posts to fetch:', 1, 1000, 100)
fetch_button = st.sidebar.button('Fetch Reddit Data')
model_name = st.sidebar.selectbox('Select a model:', ['Random Forest', 'SVM', 'Decision Tree'])
train_button = st.sidebar.button('Train Model')

# Fetch and store the dataset in session state
if fetch_button:
    st.session_state.df = scrape_reddit_data(subreddit_name, limit)

# Display dataset and model performance
if 'df' in st.session_state and not st.session_state.df.empty:
    
    st.write('### Displaying dataset:')
    st.dataframe(st.session_state.df, width=800)  # Adjust width as needed

    col1, col2 = st.columns(2)
    
    with col1:
        # Generate word cloud
        all_text = ' '.join(st.session_state.df['headline'])
        wordcloud = WordCloud(width=400, height=400, background_color='white').generate(all_text)
        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
            
    with col2:
        if train_button:
            # Prepare the data
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(st.session_state.df['headline'])
            y = st.session_state.df['label']
                
            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
            # Train and evaluate the model
            st.write(f'### Training {model_name} model...')
            report = train_model(model_name, X_train, X_test, y_train, y_test)
            st.write('### Model performance:')
            st.text(report)
