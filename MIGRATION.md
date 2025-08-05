# Migration Guide: From Gemini-Only to Multi-Provider LLM Support

This guide helps you migrate from the previous Gemini-only version to the new multi-provider LLM architecture.

## Overview of Changes

The application has been completely refactored to support multiple LLM providers while maintaining backward compatibility with existing Gemini configurations. The key changes include:

- **Multi-Provider Architecture**: Support for Google Gemini, Azure OpenAI, AWS Bedrock, and OpenAI-compatible providers
- **Unified LLM Interface**: All providers use the same standardized interface
- **Dynamic Provider Selection**: Switch between providers through UI or configuration
- **Enhanced Configuration**: More flexible environment variable configuration
- **Frontend Updates**: New provider and model selection components

## Breaking Changes

### 1. Environment Variables

**Old Configuration (Gemini-only):**
```env
GEMINI_API_KEY=your_api_key_here
```

**New Configuration (Multi-provider):**
```env
# Choose your provider
LLM_PROVIDER=GOOGLE_GEMINI

# Google Gemini configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODELS=gemini-1.5-pro,gemini-1.5-flash,gemini-1.0-pro
GEMINI_DEFAULT_MODEL=gemini-1.5-pro
```

### 2. API Changes

**Old API Request:**
```javascript
// Frontend API calls
const response = await fetch('/runs/stream', {
  method: 'POST',
  body: JSON.stringify({
    input: { query: userQuery }
  })
});
```

**New API Request:**
```javascript
// Frontend API calls with provider selection
const response = await fetch('/runs/stream', {
  method: 'POST',
  body: JSON.stringify({
    input: { 
      query: userQuery,
      llmProvider: selectedProvider // Optional, uses default if not specified
    }
  })
});
```

### 3. Code Structure Changes

**Old Structure:**
```
backend/src/agent/
├── graph.py          # Direct Gemini API calls
├── configuration.py  # Simple configuration
└── app.py           # Basic FastAPI app
```

**New Structure:**
```
backend/src/agent/
├── graph.py              # Uses LLM service layer
├── configuration.py      # Multi-provider configuration
├── llm_types.py         # Core data structures
├── llm_providers.py     # Base provider interface
├── llm_factory.py       # Provider factory
├── llm_service.py       # Unified service layer
├── app.py              # Enhanced API with provider endpoints
└── providers/
    ├── gemini_llm_provider.py
    ├── azure_openai_provider.py
    ├── bedrock_llm_provider.py
    └── openai_compatible_provider.py
```

## Migration Steps

### Step 1: Update Dependencies

Update your Python dependencies:

```bash
cd backend
pip install .
```

The new `pyproject.toml` includes additional dependencies for multi-provider support:
- `langchain-openai` for OpenAI and Azure OpenAI
- `boto3` for AWS Bedrock
- Enhanced error handling libraries

### Step 2: Update Environment Configuration

1. **Backup your current `.env` file:**
   ```bash
   cp backend/.env backend/.env.backup
   ```

2. **Copy the new environment template:**
   ```bash
   cp backend/.env.example backend/.env
   ```

3. **Migrate your Gemini configuration:**
   
   **If you had:**
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   
   **Update to:**
   ```env
   # Default provider (maintains backward compatibility)
   LLM_PROVIDER=GOOGLE_GEMINI
   
   # Google Gemini configuration
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODELS=gemini-1.5-pro,gemini-1.5-flash,gemini-1.0-pro
   GEMINI_DEFAULT_MODEL=gemini-1.5-pro
   
   # Task-specific model assignments (optional)
   GEMINI_QUERY_GENERATION_MODEL=gemini-1.5-flash
   GEMINI_REFLECTION_MODEL=gemini-1.5-pro
   GEMINI_ANSWER_GENERATION_MODEL=gemini-1.5-pro
   ```

### Step 3: Update Frontend Dependencies (if customized)

If you've customized the frontend, update your dependencies:

```bash
cd frontend
npm install
```

The frontend now includes new components for provider and model selection.

### Step 4: Test Your Migration

1. **Start the development servers:**
   ```bash
   make dev
   ```

2. **Verify Gemini still works:**
   - Open the application in your browser
   - The default provider should be Google Gemini
   - Test a research query to ensure functionality

3. **Test provider selection:**
   - Use the new provider dropdown in the UI
   - Verify that your Gemini configuration appears in the model dropdown

### Step 5: Optional - Add Additional Providers

Once your Gemini configuration is working, you can add additional providers:

#### Add Azure OpenAI:
```env
# Add to your .env file
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_MODELS=gpt-4,gpt-4-turbo,gpt-35-turbo
AZURE_OPENAI_DEFAULT_MODEL=gpt-4
```

#### Add AWS Bedrock:
```env
# Add to your .env file
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
BEDROCK_MODELS=anthropic.claude-3-sonnet-20240229-v1:0,anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_DEFAULT_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

#### Add OpenAI Compatible:
```env
# Add to your .env file
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODELS=gpt-4,gpt-4-turbo,gpt-3.5-turbo
OPENAI_DEFAULT_MODEL=gpt-4
```

## Backward Compatibility

### What Still Works

- **Existing Gemini API Keys**: Your current `GEMINI_API_KEY` will continue to work
- **Default Behavior**: If no `LLM_PROVIDER` is specified, the system defaults to Google Gemini
- **CLI Scripts**: Existing CLI scripts will continue to work without modification
- **API Endpoints**: The core `/runs/stream` endpoint maintains backward compatibility

### What's Enhanced

- **Provider Selection**: You can now choose different providers through the UI
- **Model Selection**: Dynamic model selection based on the chosen provider
- **Configuration Flexibility**: More granular control over model selection for different tasks
- **Error Handling**: Improved error handling and fallback mechanisms
- **Performance**: Better rate limiting and retry logic

## Troubleshooting Migration Issues

### Issue: "Provider not found" errors

**Solution:** Ensure your `LLM_PROVIDER` environment variable is set correctly:
```env
LLM_PROVIDER=GOOGLE_GEMINI  # Must be uppercase
```

### Issue: Models not appearing in dropdown

**Solution:** Check your model configuration:
```env
GEMINI_MODELS=gemini-1.5-pro,gemini-1.5-flash  # Comma-separated, no spaces
```

### Issue: API authentication errors

**Solution:** Verify your API keys are correctly set:
```bash
# Check your environment variables
echo $GEMINI_API_KEY
```

### Issue: Frontend not showing provider selection

**Solution:** Clear your browser cache and restart the development server:
```bash
# Stop servers
# Clear browser cache
make dev  # Restart
```

### Issue: Docker deployment issues

**Solution:** Update your docker-compose.yml environment variables:
```yaml
environment:
  - LLM_PROVIDER=GOOGLE_GEMINI
  - GEMINI_API_KEY=${GEMINI_API_KEY}
  - GEMINI_DEFAULT_MODEL=gemini-1.5-pro
```

## Rollback Plan

If you encounter issues and need to rollback:

1. **Restore your backup:**
   ```bash
   cp backend/.env.backup backend/.env
   ```

2. **Use git to revert changes:**
   ```bash
   git checkout HEAD~1  # Or your previous working commit
   ```

3. **Reinstall old dependencies:**
   ```bash
   cd backend
   pip install .
   ```

## Testing Your Migration

### Basic Functionality Test

1. Start the application
2. Submit a research query
3. Verify the response includes citations
4. Check that the provider shows as "Google Gemini" in the UI

### Multi-Provider Test (if configured)

1. Select different providers from the dropdown
2. Choose different models for each provider
3. Submit queries and verify responses
4. Check that switching works without errors

### CLI Test

```bash
cd backend
python examples/cli_research.py "Test query for migration"
```

## Getting Help

If you encounter issues during migration:

1. **Check the logs:** Look for error messages in the console output
2. **Verify configuration:** Use the debug endpoints to check your configuration
3. **Test incrementally:** Add one provider at a time
4. **Check the documentation:** Refer to the updated README.md for detailed configuration options

## Post-Migration Benefits

After successful migration, you'll have access to:

- **Multiple LLM Providers**: Switch between different AI models based on your needs
- **Cost Optimization**: Use different providers for different tasks based on cost and performance
- **Redundancy**: Fallback to different providers if one is unavailable
- **Enterprise Features**: Access to Azure OpenAI and AWS Bedrock for enterprise deployments
- **Future-Proofing**: Easy addition of new providers as they become available

## Next Steps

Once migration is complete:

1. **Explore Different Providers**: Test different models to find the best fit for your use cases
2. **Optimize Configuration**: Fine-tune model assignments for different tasks
3. **Monitor Performance**: Compare response quality and speed across providers
4. **Scale Deployment**: Consider enterprise providers for production deployments

The migration maintains full backward compatibility while opening up new possibilities for LLM provider flexibility and optimization.