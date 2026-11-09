# app/

## Why this folder exists
Entry points for people. The UIs are kept thin and separate from `smartdesk/` so the same agent can be driven from
a browser, a terminal, tests or evals, and a new interface (Slack, Teams, an API) can be added without touching the
agent. Neither file contains business logic; both call `SmartDeskAgent.send()`.

## Files

| File | What it does | Why it exists |
|---|---|---|
| `streamlit_app.py` | Web chat on port 8502: Google sign-in when configured, otherwise a labelled demo mode; Yes/No buttons; sidebar with sources and confidence | The main user experience. Identity is decided here (sign-in vs demo) and passed to the agent, so the agent never trusts an email typed in production mode. Buttons make confirmations unambiguous |
| `cli.py` | Terminal chat; `--debug` prints sources, confidence and state; `--email` sets the identity | Fast local testing and debugging without a browser; `--debug` shows why the agent did what it did |
| `__init__.py` | Makes `app` a package | Needed for `python -m app.cli` |
