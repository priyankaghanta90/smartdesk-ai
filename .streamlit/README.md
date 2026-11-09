# .streamlit/

## Why this folder exists
Streamlit reads its configuration and secrets from this folder by convention. Settings that are safe to share
(port) are committed; credentials are kept in a git-ignored file, with a committed template showing what's needed.

## Files

| File | What it is | Why it exists |
|---|---|---|
| `config.toml` | Port 8502, usage stats off | 8501 is reserved by Windows on the dev machine, so the port is fixed here instead of on every command line |
| `secrets.toml.example` | Template for Google sign-in (`[auth]`: client id/secret, redirect URI, cookie secret) | Shows exactly which values are needed without exposing real ones |
| `secrets.toml` | Your real values. **Git-ignored; never commit it** | Holds the OAuth credentials. Without it, the app runs in demo mode |
