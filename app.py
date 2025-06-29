import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import praw
from textblob import TextBlob
from datetime import datetime
import networkx as nx
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Streamlit UI
st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")
st.title("📊 Reddit Sentiment Dashboard")

# User inputs
col_input1, col_input2, col_input3 = st.columns(3)
topic = col_input1.text_input("Enter topic to analyze:", "Pixel")
subreddit_name = col_input2.text_input("Enter subreddit (default: all):", "all")
post_type = col_input3.selectbox("Post type", ["hot", "new", "top"])
num_posts = st.slider("Number of posts to analyze", 10, 500, 100, 10)

if topic:
    # Reddit API setup
    reddit = praw.Reddit(
        client_id='o-kEGySSNCVXVjTHRiiStw',
        client_secret='FHuig0v-dNdhRS7hk3WWSohfbO-BcA',
        user_agent='Analysis u/Kaishi_Light'
    )

    # Fetch posts based on user settings
    subreddit = reddit.subreddit(subreddit_name)
    if post_type == "hot":
        posts = subreddit.search(topic, sort='hot', limit=num_posts)
    elif post_type == "new":
        posts = subreddit.search(topic, sort='new', limit=num_posts)
    else:
        posts = subreddit.search(topic, sort='top', limit=num_posts)

    # Extract data from posts
    data = []
    for post in posts:
        title = post.title
        comments_text = []
        post.comments.replace_more(limit=0)
        for comment in post.comments.list():
            comments_text.append(comment.body)
        all_text = title + " ".join(comments_text[:10])
        data.append({
            'Post': title,
            'Sentiment': TextBlob(all_text).sentiment.polarity,
            'Upvotes': post.score,
            'Timestamp': datetime.utcfromtimestamp(post.created_utc),
            'Comments': " ".join(comments_text[:10])
        })

    df = pd.DataFrame(data)

    if df.empty:
        st.warning(f"No posts found for '{topic}' in r/{subreddit_name}.")
    else:
        st.success(f"Fetched {len(df)} posts containing '{topic}' from r/{subreddit_name}")

        # --- Layout Section ---
        st.markdown("### 📌 Overview Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts", len(df))
        col2.metric("Average Sentiment", f"{df['Sentiment'].mean():.2f}")
        col3.metric("Top Upvotes", df.loc[df['Upvotes'].idxmax(), 'Upvotes'])

        st.markdown("---")

        # --- Sentiment Over Time ---
        st.subheader("📈 Sentiment Over Time")
        fig1, ax1 = plt.subplots()
        df_sorted = df.sort_values("Timestamp")
        sns.lineplot(data=df_sorted, x="Timestamp", y="Sentiment", marker="o", ax=ax1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

        # --- Sentiment Distribution ---
        st.subheader("📊 Sentiment Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df['Sentiment'], bins=10, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Sentiment Polarity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Sentiment Scores')
        st.pyplot(fig2)

        # --- Word Cloud ---
        st.subheader("📝 Word Cloud of Post Titles & Comments")
        full_text = " ".join(df['Post'].tolist() + df['Comments'].tolist())
        wordcloud = WordCloud(background_color='white', stopwords=stop_words, width=800, height=400).generate(full_text)
        fig3, ax3 = plt.subplots()
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)

        # --- Network Analysis ---
        st.subheader("🌐 Co-occurrence Network of Common Words")
        words = [w for w in full_text.lower().split() if w.isalpha() and w not in stop_words]
        word_pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        G = nx.Graph()
        G.add_edges_from(word_pairs)
        freq = Counter(word_pairs)
        for u, v in G.edges():
            G[u][v]['weight'] = freq[(u, v)]
        top_edges = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:50]
        G_top = nx.Graph()
        G_top.add_edges_from([pair for pair, _ in top_edges])
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G_top, k=0.5)
        nx.draw(G_top, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800, font_size=10, ax=ax4)
        st.pyplot(fig4)

        # --- Top Posts ---
        st.subheader("🔥 Top Posts by Upvotes")
        top_n = st.slider("Select number of top posts", 1, 20, 5)
        top_df = df.nlargest(top_n, 'Upvotes')
        st.bar_chart(top_df.set_index('Post')['Upvotes'])

        # --- Raw Data Table ---
        st.subheader("📋 Raw Data Table")
        st.dataframe(df.sort_values("Timestamp", ascending=False))