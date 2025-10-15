import requests
import json
import logging
from typing import List, Dict, Any, Optional
from config.settings import Config
from core.models import SearchResult

logger = logging.getLogger(__name__)


class SerperClient:
    """Client for interacting with Serper.dev search API"""
    
    def __init__(self):
        self.api_key = Config.SERPER_API_KEY
        self.base_url = "https://google.serper.dev"
        self.timeout = Config.REQUEST_TIMEOUT
        self.max_results = Config.MAX_SEARCH_RESULTS
        
        if not self.api_key:
            raise ValueError("Serper.dev API key not configured")
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Serper.dev"""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        logger.info(f"Making Serper API request to {url} with query: {data.get('q', 'N/A')}")
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Serper API request successful. Found {len(result.get('organic', []))} results")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Serper.dev API request failed: {e}")
            logger.error(f"Request URL: {url}")
            logger.error(f"Request data: {data}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    def search(self, 
               query: str, 
               search_type: str = "search",
               num_results: Optional[int] = None) -> List[SearchResult]:
        """Perform a web search"""
        
        if num_results is None:
            num_results = self.max_results
        
        # Sanitize query but do not truncate to preserve research quality
        sanitized_query = " ".join(query.split()) if query else ""
        
        # Log query length for monitoring but don't truncate
        if len(sanitized_query) > 1000:
            logger.info(f"Large research query detected: {len(sanitized_query)} characters")

        request_data = {
            "q": sanitized_query,
            "num": num_results
        }
        
        try:
            response_data = self._make_request(search_type, request_data)
            return self._parse_search_results(response_data, query)
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:100]}...': {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full error details: {str(e)}")
            return []
    
    def _parse_search_results(self, response_data: Dict[str, Any], query: str) -> List[SearchResult]:
        """Parse search results into SearchResult objects"""
        results = []
        
        # Extract organic results
        organic_results = response_data.get("organic", [])
        
        for result in organic_results[:self.max_results]:
            search_result = SearchResult(
                title=result.get("title", ""),
                link=result.get("link", ""),
                snippet=result.get("snippet", ""),
                source="serper.dev",
                relevance_score=self._calculate_relevance_score(result, query),
                confidence_score=0.8  # Default confidence
            )
            results.append(search_result)
        
        logger.info(f"Parsed {len(results)} search results for query: {query}")
        return results
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for a search result"""
        score = 0.0
        
        # Check title relevance
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        query_terms = query.lower().split()
        
        # Score based on title matches
        title_matches = sum(1 for term in query_terms if term in title)
        score += (title_matches / len(query_terms)) * 0.6
        
        # Score based on snippet matches
        snippet_matches = sum(1 for term in query_terms if term in snippet)
        score += (snippet_matches / len(query_terms)) * 0.4
        
        return min(1.0, score)
    
    def search_technology_comparison(self, technologies: List[str], use_case: str) -> List[SearchResult]:
        """Search for technology comparisons"""
        query = f"comparison {', '.join(technologies)} for {use_case} best practices benchmarks"
        return self.search(query)
    
    def search_implementation_guides(self, technology: str, task: str) -> List[SearchResult]:
        """Search for implementation guides and tutorials"""
        query = f"{technology} {task} tutorial implementation guide best practices"
        return self.search(query)
    
    def search_trends_analysis(self, domain: str) -> List[SearchResult]:
        """Search for current trends and innovations in a domain"""
        query = f"{domain} trends 2024 innovations best practices emerging technologies"
        return self.search(query)
    
    def search_best_practices(self, technology: str, use_case: str) -> List[SearchResult]:
        """Search for best practices and patterns"""
        query = f"{technology} {use_case} best practices patterns architecture design"
        return self.search(query)
    
    def search_risk_assessment(self, technology: str, use_case: str) -> List[SearchResult]:
        """Search for risks and challenges"""
        query = f"{technology} {use_case} risks challenges limitations pitfalls"
        return self.search(query)
    
    def perform_comprehensive_research(self, user_prompt: str) -> List[SearchResult]:
        """Perform comprehensive initial research for a user prompt"""
        logger.info(f"Starting comprehensive research for: {user_prompt}")
        all_results = []
        
        # Search for general domain information
        domain_query = f"{user_prompt} technologies frameworks architecture"
        logger.info(f"Executing domain search: {domain_query}")
        domain_results = self.search(domain_query)
        logger.info(f"Domain search returned {len(domain_results)} results")
        all_results.extend(domain_results)
        
        # Search for current trends
        trends_query = f"{user_prompt} trends 2024 innovations"
        logger.info(f"Executing trends search: {trends_query}")
        trends_results = self.search(trends_query)
        logger.info(f"Trends search returned {len(trends_results)} results")
        all_results.extend(trends_results)
        
        # Search for best practices
        practices_query = f"{user_prompt} best practices patterns"
        logger.info(f"Executing practices search: {practices_query}")
        practices_results = self.search(practices_query)
        logger.info(f"Practices search returned {len(practices_results)} results")
        all_results.extend(practices_results)
        
        # Remove duplicates based on URL
        unique_results = {}
        for result in all_results:
            if result.link not in unique_results:
                unique_results[result.link] = result
        
        final_results = list(unique_results.values())[:self.max_results * 2]
        logger.info(f"Comprehensive research completed: {len(final_results)} unique results")
        return final_results
    
    def perform_targeted_research(self, research_gaps: List[str]) -> List[SearchResult]:
        """Perform targeted research to fill specific knowledge gaps"""
        all_results = []
        
        for gap in research_gaps:
            query = f"{gap} implementation guide tutorial examples"
            results = self.search(query, num_results=3)  # Fewer results per gap
            all_results.extend(results)
        
        # Remove duplicates and return
        unique_results = {}
        for result in all_results:
            if result.link not in unique_results:
                unique_results[result.link] = result
        
        return list(unique_results.values())
    
    def extract_key_insights(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract key insights from search results"""
        insights = []
        
        for result in search_results:
            insight = {
                'title': result.title,
                'source': result.link,
                'content': result.snippet,
                'relevance_score': result.relevance_score,
                'categories': self._categorize_insight(result.snippet),
                'key_points': self._extract_key_points(result.snippet)
            }
            insights.append(insight)
        
        # Sort by relevance score
        insights.sort(key=lambda x: x['relevance_score'], reverse=True)
        return insights
    
    def _categorize_insight(self, snippet: str) -> List[str]:
        """Categorize an insight based on content"""
        categories = []
        snippet_lower = snippet.lower()
        
        if any(word in snippet_lower for word in ['performance', 'speed', 'fast', 'slow']):
            categories.append('performance')
        if any(word in snippet_lower for word in ['security', 'safe', 'vulnerability', 'attack']):
            categories.append('security')
        if any(word in snippet_lower for word in ['cost', 'price', 'expensive', 'cheap']):
            categories.append('cost')
        if any(word in snippet_lower for word in ['easy', 'simple', 'complex', 'difficult']):
            categories.append('complexity')
        if any(word in snippet_lower for word in ['scale', 'scalability', 'large', 'grow']):
            categories.append('scalability')
        if any(word in snippet_lower for word in ['maintain', 'maintenance', 'update']):
            categories.append('maintainability')
        
        return categories
    
    def _extract_key_points(self, snippet: str) -> List[str]:
        """Extract key points from a snippet"""
        # Simple extraction - split by sentences and take first few
        sentences = snippet.split('.')
        key_points = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
        return key_points