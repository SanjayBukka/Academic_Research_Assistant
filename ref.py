import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import xml.etree.ElementTree as ET

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
        headers = {'User-Agent': 'ResearchAssistant (mailto:your.email@example.com)'}
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,authors,year,url,citationCount'
        }
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            papers_data = response.json().get('data', [])
            return [{
                'title': paper.get('title', 'No Title'),
                'abstract': paper.get('abstract', 'No Abstract Available'),
                'authors': [author.get('name', '') for author in paper.get('authors', [])],
                'year': str(paper.get('year', 'Unknown')),
                'url': paper.get('url', ''),
                'platform': 'Semantic Scholar'
            } for paper in papers_data]
        except Exception as e:
            st.error(f"Semantic Scholar Error: {e}")
            return []

    def _search_arxiv(self, query, limit=50):
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': limit
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                papers.append({
                    'title': entry.find('{http://www.w3.org/2005/Atom}title').text or 'No Title',
                    'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text or 'No Abstract Available',
                    'authors': [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')] or ['Unknown'],
                    'year': entry.find('{http://www.w3.org/2005/Atom}published').text[:4] or 'Unknown',
                    'url': entry.find('{http://www.w3.org/2005/Atom}id').text or '',
                    'platform': 'arXiv'
                })
            return papers
        except Exception as e:
            st.error(f"arXiv Error: {e}")
            return []

    def _search_crossref(self, query, limit=50):
        base_url = "https://api.crossref.org/works"
        headers = {'User-Agent': 'ResearchAssistant (mailto:your.email@example.com)'}
        params = {
            'query': query,
            'rows': limit
        }
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            papers_data = response.json().get('message', {}).get('items', [])
            return [{
                'title': paper.get('title', ['No Title'])[0],
                'abstract': paper.get('abstract', 'No Abstract Available'),
                'authors': [f"{author.get('given', '')} {author.get('family', '')}".strip() for author in paper.get('author', [])] or ['Unknown'],
                'year': str(paper.get('published', {}).get('date-parts', [['']])[0][0]) or 'Unknown',
                'url': paper.get('URL', ''),
                'platform': 'CrossRef'
            } for paper in papers_data]
        except Exception as e:
            st.error(f"CrossRef Error: {e}")
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
        return ' '.join(text.split())

    def prepare_recommendation_system(self, papers_data):
        filtered_papers = [paper for paper in papers_data if self.preprocess_text(paper.get('abstract', ''))]
        if not filtered_papers:
            st.error("No valid abstracts found for recommendation.")
            return
        self.papers_df = pd.DataFrame(filtered_papers)
        self.papers_df['processed_abstract'] = self.papers_df['abstract'].apply(self.preprocess_text)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.papers_df['processed_abstract'])
        st.info(f"Recommendation system ready with {len(self.papers_df)} papers.")

    def recommend_papers(self, topic, top_k=10):
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            st.error("Recommendation system not prepared.")
            return pd.DataFrame()
        processed_topic = self.preprocess_text(topic)
        topic_vector = self.vectorizer.transform([processed_topic])
        cosine_similarities = cosine_similarity(topic_vector, self.tfidf_matrix)[0]
        recommendations_df = self.papers_df.copy()
        recommendations_df['similarity_score'] = cosine_similarities
        return recommendations_df.sort_values('similarity_score', ascending=False).head(top_k)

def main():
    st.set_page_config(page_title="Research Paper Recommender", page_icon="📚")
    st.title("🔬 Academic Research Assistant: Paper Finder")

    research_assistant = ResearchPaperSearchAssistant()

    # Sidebar
    st.sidebar.header("Search Parameters")
    research_topic = st.sidebar.text_input("Enter Research Topic", "machine learning in healthcare")
    available_platforms = list(research_assistant.platforms.keys())
    selected_platforms = st.sidebar.multiselect("Select Platforms", available_platforms, default=available_platforms)
    top_k = st.sidebar.slider("Number of Recommendations", 2, 10, 5)

    if st.sidebar.button("Find Papers"):
        with st.spinner("Searching papers..."):
            papers = research_assistant.search_papers(research_topic, selected_platforms, limit=50)
        if papers:
            with st.spinner("Preparing recommendations..."):
                research_assistant.prepare_recommendation_system(papers)
            with st.spinner("Generating recommendations..."):
                recommendations = research_assistant.recommend_papers(research_topic, top_k=top_k)
            if not recommendations.empty:
                st.header("Top Recommended Papers")
                for idx, paper in recommendations.iterrows():
                    st.markdown(f"### [{paper['title']}]({paper['url']})")
                    st.markdown(f"**Platform:** {paper['platform']}")
                    st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                    st.markdown(f"**Year:** {paper['year']}")
                    with st.expander("Abstract"):
                        st.write(paper['abstract'])
                    st.progress(float(paper['similarity_score']))
                    st.divider()
            else:
                st.warning("No recommendations found.")
        else:
            st.warning("No papers retrieved. Check connection or topic.")

if __name__ == "__main__":
    main()