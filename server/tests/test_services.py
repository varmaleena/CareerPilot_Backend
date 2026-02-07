import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestLLMGateway:
    """Tests for LLM Gateway service."""
    
    def test_model_router_task_routing(self):
        """Test model router routes tasks correctly."""
        from app.services.llm.model_router import ModelRouter
        
        router = ModelRouter()
        
        # Lite tasks - cheap and fast
        assert router.route("validate").model == "gemini-2.0-flash-lite"
        assert router.route("extract").model == "gemini-2.0-flash-lite"
        
        # Flash tasks - balanced
        assert router.route("generate").model == "gemini-2.5-flash"
        
        # Pro tasks - expensive, high quality
        assert router.route("reason").model == "gemini-2.5-pro"
        assert router.route("resolve").model == "gemini-2.5-pro"
    
    def test_key_manager_rotation(self):
        """Test API key rotation."""
        from app.services.llm.key_manager import KeyManager
        
        with patch("app.services.llm.key_manager.settings") as mock_settings:
            mock_settings.gemini_keys_list = ["key1", "key2", "key3"]
            
            manager = KeyManager()
            
            # Should rotate through keys
            keys = [manager.get_next_key() for _ in range(6)]
            assert keys == ["key1", "key2", "key3", "key1", "key2", "key3"]
    
    def test_key_manager_handles_failures(self):
        """Test key manager skips failed keys."""
        from app.services.llm.key_manager import KeyManager
        
        with patch("app.services.llm.key_manager.settings") as mock_settings:
            mock_settings.gemini_keys_list = ["key1", "key2", "key3"]
            
            manager = KeyManager()
            
            # Mark key1 as failed
            manager.mark_failed("key1")
            
            # Should skip key1
            key = manager.get_next_key()
            assert key != "key1"
    
    def test_token_counter(self):
        """Test token counting estimation."""
        from app.services.llm.token_counter import TokenCounter
        
        counter = TokenCounter()
        
        prompt = "Hello world"  # 11 chars
        response = "Hi there"   # 8 chars
        
        tokens = counter.count(prompt, response)
        # Rough estimation: 19 chars / 4 ≈ 4-5 tokens
        assert tokens >= 4
        assert tokens <= 10


class TestSemanticCache:
    """Tests for Semantic Cache."""
    
    @pytest.mark.asyncio
    async def test_cache_ttl_config(self):
        """Test different operations have different TTLs."""
        from app.services.cache.semantic_cache import SemanticCache
        
        cache = SemanticCache()
        
        # Validation should cache longer (24h)
        assert cache.TTL_CONFIG["validate"] == 86400
        
        # Interview should cache shorter (2h)
        assert cache.TTL_CONFIG["interview"] == 7200
    
    def test_prompt_hashing(self):
        """Test prompt hashing is consistent."""
        from app.services.cache.semantic_cache import SemanticCache
        
        cache = SemanticCache()
        
        prompt = "Test prompt content"
        hash1 = cache._hash_prompt(prompt)
        hash2 = cache._hash_prompt(prompt)
        
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars


class TestPDFExporter:
    """Tests for PDF/LaTeX export service."""
    
    def test_latex_generator_escapes_special_chars(self, sample_resume_data):
        """Test LaTeX special characters are escaped."""
        from app.services.export.pdf_generator import LaTeXGenerator
        
        generator = LaTeXGenerator()
        
        # Test escaping
        escaped = generator._escape("Test & Co. with $100")
        assert r"\&" in escaped
        assert r"\$" in escaped
    
    def test_latex_generation(self, sample_resume_data):
        """Test LaTeX document generation."""
        from app.services.export.pdf_generator import LaTeXGenerator
        
        generator = LaTeXGenerator()
        latex = generator.generate_latex(sample_resume_data)
        
        assert r"\documentclass" in latex
        assert sample_resume_data["name"] in latex
        assert r"\section{Experience}" in latex
    
    @pytest.mark.asyncio
    async def test_pdf_export_fallback(self, sample_resume_data):
        """Test PDF export falls back to LaTeX when pdflatex not available."""
        from app.services.export.pdf_generator import PDFExporter
        
        with patch("subprocess.run", side_effect=FileNotFoundError):
            exporter = PDFExporter()
            result = await exporter.export_resume_to_pdf(sample_resume_data)
            
            # Should return LaTeX bytes as fallback
            assert isinstance(result, bytes)
            assert b"\\documentclass" in result


class TestJobFetcher:
    """Tests for Job Fetcher service."""
    
    @pytest.mark.asyncio
    async def test_search_returns_jobs(self):
        """Test job search returns results."""
        from app.services.jobs.fetcher import JobFetcher
        
        fetcher = JobFetcher()
        jobs = await fetcher.search(query="Python Developer", location="Remote")
        
        assert len(jobs) > 0
        assert jobs[0].source == "mock"  # Using mock data
    
    @pytest.mark.asyncio
    async def test_job_matching(self, sample_resume_data):
        """Test job matching against resume."""
        from app.services.jobs.fetcher import JobFetcher, JobListing
        
        fetcher = JobFetcher()
        
        jobs = [
            JobListing(
                title="Python Developer",
                company="Test Co",
                location="Remote",
                description="Looking for Python and AWS skills",
                url="http://example.com",
                source="test",
            ),
        ]
        
        matches = await fetcher.match_jobs_to_resume(sample_resume_data, jobs)
        
        assert len(matches) == 1
        assert matches[0]["match_score"] > 0


class TestMarketService:
    """Tests for Market Service."""
    
    @pytest.mark.asyncio
    async def test_get_role_insights(self):
        """Test getting market insights for a role."""
        from app.services.market.market_service import MarketService
        
        service = MarketService()
        insights = await service.get_role_insights("Software Engineer")
        
        assert insights.demand_score > 0
        assert len(insights.top_skills) > 0
        assert insights.salary_range["median"] > 0
    
    @pytest.mark.asyncio
    async def test_compare_skills(self):
        """Test skill comparison to market."""
        from app.services.market.market_service import MarketService
        
        service = MarketService()
        comparison = await service.compare_skills_to_market(
            user_skills=["Python", "AWS"],
            target_role="Software Engineer",
        )
        
        assert "readiness_score" in comparison
        assert "matching_skills" in comparison
        assert "missing_core_skills" in comparison
    
    @pytest.mark.asyncio
    async def test_salary_benchmark(self):
        """Test salary benchmarking."""
        from app.services.market.market_service import MarketService
        
        service = MarketService()
        benchmark = await service.salary_benchmark(
            role="Software Engineer",
            experience_years=5,
        )
        
        assert benchmark["salary_range"]["min"] > 0
        assert benchmark["percentiles"]["50th"] > 0
