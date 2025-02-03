import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import time

class ResearchPaperSearchAssistant:
    def __init__(self):
        self.platforms = {
            "Semantic Scholar": self._search_semantic_scholar,
            "arXiv": self._search_arxiv,
            "CrossRef": self._search_crossref
        }

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.papers_df = None
        self.tfidf_matrix = None

    def _search_semantic_scholar(self, query, limit=50):
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {'Content-Type': 'application/json'}

        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,authors,year,url,citationCount'
        }

        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()

            papers_data = response.json().get('data', [])

            processed_papers = []
            for paper in papers_data:
                processed_papers.append({
                    'title': paper.get('title', 'No Title'),
                    'abstract': paper.get('abstract', 'No Abstract Available'),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'year': paper.get('year', 'Unknown'),
                    'url': paper.get('url', ''),
                    'platform': 'Semantic Scholar'
                })

            return processed_papers

        except Exception as e:
            st.error(f"Semantic Scholar API Error: {e}")
            return []

    def _search_arxiv(self, query, limit=50):
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': limit
        }

        try:
            response = requests.get(base_url, params=params)
            # Parse XML response (would require additional XML parsing logic)
            # Placeholder for actual implementation
            return []

        except Exception as e:
            st.error(f"arXiv API Error: {e}")
            return []

    def _search_crossref(self, query, limit=50):
        base_url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': limit
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            papers_data = response.json().get('message', {}).get('items', [])

            processed_papers = []
            for paper in papers_data:
                processed_papers.append({
                    'title': paper.get('title', ['No Title'])[0],
                    'abstract': paper.get('abstract', 'No Abstract Available'),
                    'authors': [author.get('given', '') + ' ' + author.get('family', '') for author in paper.get('author', [])],
                    'year': paper.get('published', {}).get('date-parts', [['']])[0][0],
                    'url': paper.get('URL', ''),
                    'platform': 'CrossRef'
                })

            return processed_papers

        except Exception as e:
            st.error(f"CrossRef API Error: {e}")
            return []

    def search_papers(self, query, platforms, limit=50):
        all_papers = []
        for platform in platforms:
            search_method = self.platforms.get(platform)
            if search_method:
                papers = search_method(query, limit)
                all_papers.extend(papers)

        return all_papers

    def preprocess_text(self, text):
        if not text or text == 'No Abstract Available':
            return ""

        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())

        return text

    def prepare_recommendation_system(self, papers_data):
        filtered_papers = [
            paper for paper in papers_data
            if self.preprocess_text(paper.get('abstract', ''))
        ]

        self.papers_df = pd.DataFrame(filtered_papers)

        self.papers_df['processed_abstract'] = self.papers_df['abstract'].apply(self.preprocess_text)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.papers_df['processed_abstract'])

        st.info(f"Recommendation system prepared with {len(self.papers_df)} papers")

    def recommend_papers(self, topic, top_k=10):
        if self.tfidf_matrix is None or len(self.tfidf_matrix.toarray()) == 0:
            st.error("Recommendation system not prepared.")
            return pd.DataFrame()

        processed_topic = self.preprocess_text(topic)
        topic_vector = self.vectorizer.transform([processed_topic])

        cosine_similarities = cosine_similarity(topic_vector, self.tfidf_matrix)[0]

        recommendations_df = self.papers_df.copy()
        recommendations_df['similarity_score'] = cosine_similarities

        top_recommendations = recommendations_df.sort_values(
            'similarity_score',
            ascending=False
        ).head(top_k)

        return top_recommendations[['title', 'abstract', 'authors', 'year', 'url', 'platform', 'similarity_score']]

def main():
    st.set_page_config(page_title="Academic reseach assistance- Research Paper Recommender", page_icon="📚")

    st.title("🔬 Academic Research Assistant/Research Reference papers")

    research_assistant = ResearchPaperSearchAssistant()

    # Sidebar for search parameters
    st.sidebar.header("Search Parameters")
    research_topic = st.sidebar.text_input("Enter Research Topic", "machine learning in healthcare")

    # Platform selection
    available_platforms = list(research_assistant.platforms.keys())
    selected_platforms = st.sidebar.multiselect(
        "Select Research Platforms",
        available_platforms,
        default=available_platforms
    )

    # Number of recommendations
    top_k = st.sidebar.slider("Number of Similar Papers", min_value=2, max_value=10, value=5)

    # Search button
    if st.sidebar.button("Find Paper Recommendation"):
        with st.spinner("Searching for papers..."):
            papers = research_assistant.search_papers(research_topic, selected_platforms, limit=50)

        with st.spinner("Preparing recommendation system..."):
            research_assistant.prepare_recommendation_system(papers)

        with st.spinner("Getting Papers..."):
            recommendations = research_assistant.recommend_papers(
                research_topic,
                top_k=top_k
            )

        # Display recommendations
        if not recommendations.empty:
            st.header("Top Similar Papers")

            for idx, paper in recommendations.iterrows():
                st.markdown(f"### [{paper['title']}]({paper['url']})")
                st.markdown(f"**Platform:** {paper['platform']}")
                st.markdown(f"**Authors:** {', '.join(eval(paper['authors']) if isinstance(paper['authors'], str) else paper['authors'])}")
                st.markdown(f"**Year:** {paper['year']}")

                with st.expander("Abstract"):
                    st.write(paper['abstract'])

                st.progress(paper['similarity_score'])
                st.divider()
        else:
            st.warning("No recommendations found. Try adjusting search parameters.")

if __name__ == "__main__":
    main()