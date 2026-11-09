"""Terminal chat with SmartDesk.

    python -m app.cli            # add --debug to print sources / confidence each turn
    python -m app.cli --email jane.doe@novatech.com   # act as this employee (trusted local identity)
"""

import argparse
import sys
import uuid

from smartdesk.agent.flow import SmartDeskAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--email", help="signed-in employee for this terminal session (local dev stand-in for SSO)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    agent = SmartDeskAgent()
    thread_id = str(uuid.uuid4())
    print("SmartDesk AI — NovaTech IT & HR help desk. Type 'exit' to quit, 'new' for a new session.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text.lower() == "new":
            thread_id = str(uuid.uuid4())
            print("(new session)\n")
            continue
        result = agent.send(thread_id, text, user_email=args.email)
        print(f"\nSmartDesk: {result.reply}\n")
        if args.debug:
            src = ", ".join(f"{s['chunk_id']}({s['score']})" for s in result.sources)
            print(f"  [pending={result.pending} cache={result.cache_hit} conf={result.confidence} "
                  f"email={result.email} reason={result.escalation_reason}]\n  [sources: {src}]\n")


if __name__ == "__main__":
    main()
