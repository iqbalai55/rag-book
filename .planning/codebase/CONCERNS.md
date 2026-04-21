# Codebase Concerns

**Analysis Date:** 2026-04-21

## Tech Debt

**[Event Loop Policy]:**
- Issue: Commented out assertion for Windows event loop policy in main_fastapi.py
- Files: `main_fastapi.py:57`
- Impact: Potential asyncio event loop issues on Windows systems
- Fix approach: Uncomment and fix the event loop policy assertion, or remove if not needed

**[Database Setup]:**
- Issue: Commented out checkpointer setup line in main_fastapi.py
- Files: `main_fastapi.py:61`
- Impact: Database checkpointer may not be properly initialized
- Fix approach: Uncomment and ensure proper async setup of checkpointer

**[Filename Typo]:**
- Issue: Misspelled filename "chace_manager.py" instead of "cache_manager.py"
- Files: `utils/chace_manager.py`
- Impact: Confusion and potential import issues
- Fix approach: Rename file to `cache_manager.py` and update all imports

**[Hardcoded Paths]:**
- Issue: Multiple hardcoded storage paths throughout codebase
- Files: 
  - `main_fastapi.py:36, 38` (STORAGE_PATH, MLRUNS_PATH)
  - `main_modal.py:34-36` (QDRANT_PATH, MLFLOW_PATH, HF_CACHE_PATH)
  - `ingest_book.py:81, 89` (qdrant_client path, pdf_path)
- Impact: Reduces portability and makes environment-specific configuration difficult
- Fix approach: Move all paths to environment variables or config file

**[Debug Mode in Production]:**
- Issue: debug=True flag in uvicorn server configuration
- Files: `main_fastapi.py:143, 148`
- Impact: Exposes sensitive information and reduces performance in production
- Fix approach: Make debug mode configurable via environment variable

## Known Bugs

**[API Key Verification Inconsistency]:**
- Symptoms: Different API key verification approaches between main_fastapi.py and main_modal.py
- Files: 
  - `main_fastapi.py:77-81` (uses global API_KEY variable)
  - `main_modal.py:52-55` (uses os.environ.get("API_KEY") directly)
- Impact: Inconsistent authentication behavior between deployment modes
- Trigger: Deploying the application in different modes (local vs Modal)
- Workaround: Ensure both files use the same API key verification method

**[Rate Limit Inconsistency Risk]:**
- Symptoms: Rate limits defined in multiple places with potential for drift
- Files: 
  - `main_fastapi.py:85, 103` (streaming: 10/min, ingest: 3/min)
  - `main_modal.py:123, 136` (streaming: 10/min, ingest: 3/min)
- Impact: Rate limits may become inconsistent between deployment modes
- Trigger: Updates to rate limits in one file but not the other
- Workaround: Centralize rate limit configuration

## Security Considerations

**[API Key Exposure]:**
- Risk: API keys loaded from environment but potentially exposed in logs or error messages
- Files: Multiple files using API_KEY from environment
- Current mitigation: API key header verification
- Recommendations: 
  - Ensure API keys are never logged
  - Consider using API key hashing for storage comparison
  - Implement rate limiting on authentication failures

**[CORS Configuration Missing]:**
- Risk: No CORS configuration visible in FastAPI applications
- Files: `main_fastapi.py`, `main_modal.py`
- Current mitigation: None visible
- Recommendations: Add appropriate CORS middleware based on deployment requirements

**[Database Connection String in Logs]:**
- Risk: Database connection strings might appear in error logs or exceptions
- Files: Files using SUPABASE_DB_URL environment variable
- Current mitigation: None visible
- Recommendations: 
  - Catch and sanitize database connection errors
  - Avoid logging full connection strings
  - Use connection pooling with proper error handling

## Performance Bottlenecks

**[Single Collection Bottleneck]:**
- Problem: All courses stored in single Qdrant collection ("lms_content")
- Files: 
  - `utils/chace_manager.py:26` (_collection_name = "lms_content")
  - `services/rag/qdrant/qdrant_db.py:29` (collection_name parameter)
  - `main_fastapi.py:117` (hardcoded "lms_content")
- Cause: Architectural decision to use single collection with course_id filtering
- Improvement path: 
  - Evaluate performance with large number of courses
  - Consider sharding or separate collections for high-volume courses
  - Add course-specific indexing strategies

**[Embedding Model Loading]:**
- Problem: Embedding model loaded multiple times in different contexts
- Files: 
  - `main_fastapi.py:42-46`
  - `main_modal.py:40-45` 
  - `ingest_book.py:76-79`
- Cause: No shared embedding model singleton
- Improvement path: Create shared embedding model service or singleton

## Fragile Areas

**[Configuration Management]:**
- Files: Multiple files with load_dotenv() calls
- Why fragile: Environment loading scattered throughout codebase
- Safe modification: Centralize environment loading in one location
- Test coverage: Gaps in configuration validation testing

**[Rate Limiting Configuration]:**
- Files: Rate limits defined in main_fastapi.py and main_modal.py
- Why fragile: Duplicate configuration that can diverge
- Safe modification: Create centralized rate limiting configuration
- Test coverage: Limited testing of rate limit boundaries

**[Path Handling]:**
- Files: Various file path operations (tmp files, storage paths)
- Why fragile: Hardcoded and relative paths may behave differently across environments
- Safe modification: Use pathlib or os.path.join for cross-platform compatibility
- Test coverage: Missing tests for path resolution in different environments

## Scaling Limits

**[Qdrant Single Instance]:**
- Current capacity: Single Qdrant instance on local filesystem or volume
- Limit: Disk space, memory, and Qdrant performance limits
- Scaling path: 
  - Move to managed Qdrant service or cluster
  - Implement backup and disaster recovery procedures
  - Add monitoring for storage and performance metrics

**[Modal Scaling Constraints]:**
- Current capacity: Limited by Modal volume sizes and GPU allocation
- Limit: Modal-specific constraints on storage volumes and compute
- Scaling path: 
  - Evaluate Modal plan limits
  - Consider hybrid approach with external services
  - Implement autoscaling policies where available

## Dependencies at Risk

**[LangGraph Postgres Checkpointer]:**
- Risk: Dependency on langgraph-checkpoint-postgres which may have breaking changes
- Impact: Database checkpointing functionality could break
- Migration plan: 
  - Pin to specific version in requirements
  - Monitor changelog for breaking changes
  - Abstract checkpointing interface for easier migration

**[Document Processing Libraries]:**
- Risk: Dependencies on docling, transformers for PDF processing
- Impact: PDF ingestion pipeline could break with library updates
- Migration plan: 
  - Version pinning and regular dependency audits
  - Consider abstraction layer for document processing
  - Maintain fallback processing options

## Missing Critical Features

**[Health Check Endpoints]:**
- Problem: No health check or readiness endpoints for monitoring
- Blocks: Proper deployment orchestration and monitoring
- 
**[Comprehensive Error Handling]:**
- Problem: Limited error handling visible in endpoint functions
- Blocks: Production reliability and debugging capability
- 
**[Authentication Audit Logging]:**
- Problem: No logging of authentication attempts or failures
- Blocks: Security monitoring and incident response
- 
**[Request/Response Logging]:**
- Problem: Limited visibility into API requests and responses
- Blocks: Debugging and performance analysis

## Test Coverage Gaps

**[Error Scenario Testing]:**
- What's not tested: Error handling in API endpoints, database failures, invalid inputs
- Files: test/ directory shows some tests but limited error case coverage
- Risk: Production failures not caught by testing
- Priority: High

**[Authentication Boundary Testing]:**
- What's not tested: Invalid API keys, missing headers, rate limit exceeded scenarios
- Files: test/ directory - need to verify auth test coverage
- Risk: Security vulnerabilities in authentication
- Priority: High

**[Multitenancy Edge Cases]:**
- What's not tested: Cross-course data leakage, invalid course IDs, permission boundaries
- Files: Course-specific functionality in cache manager and agents
- Risk: Data isolation failures between tenants
- Priority: High

**[External Service Failure]:**
- What's not tested: Qdrant unavailable, Supabase down, embedding service failures
- Files: Integration points with external services
- Risk: Cascading failures in production
- Priority: Medium