"""Server tools: ``web_fetch`` and ``code_execution``.

Unlike the client tools in this package, these are *server* tools: Anthropic's own
infrastructure runs them, and this code never executes anything itself. There is no Python
function to call here; a server tool's definition is the whole story on our side. See
:mod:`scoutbot.agent.server_results` for how a server tool's result is read back out of the
model's response.

``STATUS_PAGE_DOMAIN`` is a real, public status page (Stripe's) standing in for "the
vendor's live status page". Using a real domain makes ``allowed_domains`` a genuine,
testable restriction rather than a placeholder that would break against the real API.
"""
STATUS_PAGE_DOMAIN = "status.stripe.com"

WEB_FETCH_TOOL = {
    "type": "web_fetch_20250910",
    "name": "web_fetch",
    "max_uses": 3,
    "allowed_domains": [STATUS_PAGE_DOMAIN],
}

CODE_EXECUTION_TOOL = {
    "type": "code_execution_20250825",
    "name": "code_execution",
}

SERVER_TOOLS = [WEB_FETCH_TOOL, CODE_EXECUTION_TOOL]
