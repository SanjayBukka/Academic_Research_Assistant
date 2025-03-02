import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import time
from scholarly import scholarly
import arxiv
from crossref.restful import Works
from concurrent.futures import ThreadPoolExecutor

class MultiSourceResearchAssistant:
    def __init__(self):
        self.papers_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.works = Works()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def search_semantic_scholar(self, query, limit=20):
        """Search papers from Semantic Scholar"""
        base_url = "https://api.semanticscholar.org/graph/v1"
        search_url = f"{base_url}/paper/search"
        
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,authors,year,url,citationCount'
        }
        
        try:
            response = requests.get(search_url, params=params)
            papers = response.json().get('data', [])
            return [{
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': [author.get('name', '') for author in paper.get('authors', [])],
                'year': paper.get('year', ''),
                'url': paper.get('url', ''),
                'citation_count': paper.get('citationCount', 0),
                'source': 'Semantic Scholar'
            } for paper in papers]
        except:
            return []
    
    def search_arxiv(self, query, limit=20):
        """Search papers from arXiv"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            return [{
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'year': result.published.year,
                'url': result.pdf_url,
                'citation_count': 0,  # arXiv doesn't provide citation counts
                'source': 'arXiv'
            } for result in search.results()]
        except:
            return []
    
    def search_crossref(self, query, limit=20):
        """Search papers from Crossref"""
        try:
            works = self.works.query(query).filter(has_abstract='true').limit(limit)
            return [{
                'title': work.get('title', [''])[0],
                'abstract': work.get('abstract', ''),
                'authors': [author.get('given', '') + ' ' + author.get('family', '') 
                          for author in work.get('author', [])],
                'year': work.get('published-print', {}).get('date-parts', [['']])[0][0],
                'url': work.get('URL', ''),
                'citation_count': work.get('is-referenced-by-count', 0),
                'source': 'Crossref'
            } for work in works]
        except:
            return []
    
    def search_all_sources(self, query, limit_per_source=20):
        """Search papers from all sources concurrently"""
        with ThreadPoolExecutor() as executor:
            semantic_scholar_future = executor.submit(self.search_semantic_scholar, query, limit_per_source)
            arxiv_future = executor.submit(self.search_arxiv, query, limit_per_source)
            crossref_future = executor.submit(self.search_crossref, query, limit_per_source)
            
            all_papers = (
                semantic_scholar_future.result() +
                arxiv_future.result() +
                crossref_future.result()
            )
            
        return all_papers
    
    def preprocess_text(self, text):
        """Preprocess text for TF-IDF vectorization"""
        if not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        
        return text
    
    def prepare_recommendation_system(self, papers_data):
        """Prepare the recommendation system with collected papers"""
        filtered_papers = [
            paper for paper in papers_data 
            if self.preprocess_text(paper.get('abstract', ''))
        ]
        
        self.papers_df = pd.DataFrame(filtered_papers)
        
        self.papers_df['processed_abstract'] = self.papers_df['abstract'].apply(self.preprocess_text)
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.papers_df['processed_abstract'])
        
        st.info(f"Recommendation system prepared with {len(self.papers_df)} papers from multiple sources")
    
    def recommend_papers(self, topic, top_k=10, min_citation_threshold=0):
        """Generate paper recommendations based on topic similarity"""
        if self.tfidf_matrix is None or len(self.tfidf_matrix.toarray()) == 0:
            st.error("Recommendation system not prepared.")
            return pd.DataFrame()
        
        processed_topic = self.preprocess_text(topic)
        topic_vector = self.vectorizer.transform([processed_topic])
        
        cosine_similarities = cosine_similarity(topic_vector, self.tfidf_matrix)[0]
        
        recommendations_df = self.papers_df.copy()
        recommendations_df['similarity_score'] = cosine_similarities
        
        filtered_recommendations = recommendations_df[
            recommendations_df['citation_count'] >= min_citation_threshold
        ]
        
        top_recommendations = filtered_recommendations.sort_values(
            ['similarity_score', 'citation_count'], 
            ascending=[False, False]
        ).head(top_k)
        
        return top_recommendations

def main():
    st.set_page_config(page_title="Multi-Source Research Paper Recommender", 
                      page_icon="📚",
                      layout="wide")
    
    st.title("📚 Multi-Source Research Paper Recommendation System")
    
    st.markdown("""
    This system searches and recommends papers from multiple sources:
    - Semantic Scholar
    - arXiv
    - Crossref
    """)
    
    research_assistant = MultiSourceResearchAssistant()
    
    st.sidebar.header("Search Parameters")
    research_topic = st.sidebar.text_input("Enter Research Topic", "machine learning in healthcare")
    top_k = st.sidebar.slider("Number of Recommendations", min_value=2, max_value=10, value=5)
    min_citations = st.sidebar.number_input("Minimum Citation Count", min_value=0, value=0)
    
    if st.sidebar.button("Find Recommendations"):
        with st.spinner("Searching papers from multiple sources..."):
            papers = research_assistant.search_all_sources(research_topic)
        
        with st.spinner("Preparing recommendation system..."):
            research_assistant.prepare_recommendation_system(papers)
        
        with st.spinner("Generating recommendations..."):
            recommendations = research_assistant.recommend_papers(
                research_topic, 
                top_k=top_k, 
                min_citation_threshold=min_citations
            )
        
        if not recommendations.empty:
            st.header("Top Recommended Papers")
            
            for idx, paper in recommendations.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### [{paper['title']}]({paper['url']})")
                    st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                    st.markdown(f"**Year:** {paper['year']} | **Citations:** {paper['citation_count']}")
                    st.markdown(f"**Source:** {paper['source']}")
                
                with col2:
                    st.markdown("### Similarity Score")
                    st.progress(paper['similarity_score'])
                
                with st.expander("Abstract"):
                    st.write(paper['abstract'])
                
                st.divider()
        else:
            st.warning("No recommendations found. Try adjusting search parameters.")

if __name__ == "__main__":
    main()