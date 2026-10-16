import os

# Set before app.py is imported. load_dotenv() never overrides existing variables,
# so tests use these dummies instead of the real secrets in .env.
os.environ["APP_TOKEN"] = "test-token"
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["ALLOW_FORCE_BAD"] = "false"  # tests turn it on explicitly where needed
