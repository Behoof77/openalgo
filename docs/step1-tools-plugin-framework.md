# Step 1: Tools/ Plugin Framework — Detailed Implementation Plan

## 1. Objective

Transform OpenAlgo's 50+ standalone tool blueprints into a self-discovering plugin system where each tool:
- Lives in its own `tools/<name>/` directory
- Declares metadata in `plugin.json` (name, routes, auth type, MCP tools, frontend route, nav category)
- Auto-registers with Flask at startup (no manual `app.register_blueprint()` needed)
- Shares common auth/response middleware
- Exposes MCP tool names in metadata (MCP-aware from day one)

**What this step produces:**
- `tools/` directory with framework boilerplate
- `utils/tool_loader.py` — auto-discovery engine
- `tools/_base.py` — shared auth decorator + standard response helpers
- `tools/_registry.py` — runtime registry (tools → routes, tools → MCP names)
- `app.py` changes: replace 50+ manual `register_blueprint()` calls with single `load_tools(app)` call
- Migration proof-of-concept: OI Tracker moved to `tools/oitracker/`
- Existing blueprints in `blueprints/` remain untouched during this step (old blueprints coexist)

## 2. Architecture Overview

```
openalgo/
├── tools/                          # NEW — plugin directory
│   ├── __init__.py                 # Package init
│   ├── _base.py                    # Shared auth, response helpers
│   ├── _registry.py                # Runtime tool registry
│   ├── _loader.py                  # Auto-discovery from plugin.json
│   ├── oitracker/                  # First migrated tool (PoC)
│   │   ├── plugin.json             # Tool metadata
│   │   ├── __init__.py             # Blueprint definition + routes
│   │   └── service.py              # Business logic (optional, if tool has services)
│   ├── gex/                        # Example future migration
│   │   ├── plugin.json
│   │   ├── __init__.py
│   │   └── service.py
│   └── ...                         # One directory per tool
├── blueprints/                     # EXISTING — untouched during Step 1
│   ├── oitracker.py                # Still works (old path)
│   ├── gex.py                      # Still works (old path)
│   └── ...
├── utils/
│   ├── plugin_loader.py            # Existing — broker plugins
│   └── tool_loader.py              # NEW — tool plugin auto-discovery
├── app.py                          # MODIFIED — add load_tools(app) call
└── ...
```

## 3. Design Decisions

### 3.1 Auto-Discovery vs Explicit Registration

**Decision: Auto-discover from `tools/*/plugin.json`** — same pattern as broker plugins.

Why: The existing `utils/plugin_loader.py` already does this for brokers. We follow the same proven pattern. A scan of `tools/*/plugin.json` at startup finds all tools, loads their blueprints, and registers them with Flask.

### 3.2 Auth Middleware — Where Does It Live?

**Decision: Shared decorator in `tools/_base.py`**, NOT a middleware wrapper.

Why: The existing `check_session_validity` decorator in `utils/session.py` already does session validation. Tool blueprints ALSO need `get_api_key_for_tradingview()`. The shared decorator wraps BOTH into a single `@tool_auth` decorator:

```python
# tools/_base.py
from functools import wraps
from flask import session, jsonify, g
from database.auth_db import get_api_key_for_tradingview
from utils.session import check_session_validity

def tool_auth(f):
    """Combined session + API key auth for tool endpoints."""
    @wraps(f)
    @check_session_validity
    def decorated(*args, **kwargs):
        login_username = session.get("user")
        if not login_username:
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        api_key = get_api_key_for_tradingview(login_username)
        if not api_key:
            return jsonify({"status": "error", "message": "API key not configured. Please generate an API key in /apikey"}), 401
        g.api_key = api_key
        g.login_username = login_username
        return f(*args, **kwargs)
    return decorated
```

Tools that need ONLY session auth (no API key) use `@check_session_validity` directly.

### 3.3 Standard Response Format

**Decision: Helper functions in `tools/_base.py`** that wrap `jsonify`:

```python
def tool_success(data=None, message="success"):
    return jsonify({"status": "success", "data": data, "message": message})

def tool_error(message="error", code=400):
    return jsonify({"status": "error", "message": message}), code
```

### 3.4 plugin.json Schema

```json
{
    "name": "oitracker",
    "title": "OI Tracker",
    "description": "Open Interest tracking and analysis",
    "category": "options",
    "url_prefix": "/oitracker",
    "blueprint_module": "tools.oitracker",
    "auth_type": "session_apikey",
    "csrf_exempt": false,
    "db_init": null,
    "mcp_tools": [],
    "frontend": {
        "route": "/oitracker",
        "component": "OITracker",
        "import_path": "@/pages/OITracker",
        "nav_category": "tools"
    }
}
```

**Field Definitions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique tool identifier (lowercase, underscore) |
| `title` | string | yes | Display name for Tools page |
| `description` | string | yes | Short description |
| `category` | string | yes | `options`, `dashboard`, `scanner`, `strategy`, `live_intelligence` |
| `url_prefix` | string | yes | Route prefix (e.g., `/oitracker`) |
| `blueprint_module` | string | yes | Python module path for the Blueprint |
| `auth_type` | string | yes | `session_apikey` (default), `session_only`, `api_key_only`, `none` |
| `csrf_exempt` | boolean | no | Default false. Set true for webhook endpoints |
| `db_init` | string\|null | no | Module path for `init_db()` call, or null |
| `mcp_tools` | array | no | MCP tool names this plugin exposes |
| `frontend.route` | string | no | React route path |
| `frontend.component` | string | no | React component name |
| `frontend.import_path` | string | no | Dynamic import path for lazy loading |
| `frontend.nav_category` | string | no | Which nav menu item it appears under |

### 3.5 Coexistence Strategy (Critical)

**Decision: Old blueprints and new tool plugins coexist during migration.**

- During Step 1, OI Tracker exists in BOTH `blueprints/oitracker.py` AND `tools/oitracker/`
- `app.py` loads new tools via `load_tools(app)` AND still registers old blueprints
- A `DEPRECATED` flag in `plugin.json` marks old blueprints: `"deprecated": true`
- When a tool is migrated, its old blueprint gets the deprecated flag and is removed from manual registration
- The frontend route points to the new tool's endpoint
- **No tool is removed until its new version is verified working**

### 3.6 URL Routing

**Decision: Keep existing URL structure.** 

Current: `@blueprint.route("/oitracker/api/oi")` → full URL `/oitracker/api/oi`
New: Blueprint uses `url_prefix="/oitracker"` + route `"/api/oi"` → same full URL `/oitracker/api/oi`

No URL changes means no frontend breakage.

## 4. File-by-File Implementation Plan

### 4.1 `tools/__init__.py` (NEW — ~5 lines)
```python
"""OpenAlgo Tools Plugin System."""
```
Just a package marker.

### 4.2 `tools/_base.py` (NEW — ~80 lines)
Contains:
- `tool_auth` decorator (session + API key combined)
- `tool_success(data, message)` — standard success response
- `tool_error(message, code)` — standard error response  
- `get_tool_api_key()` — helper to get API key from `g.api_key`
- `get_tool_user()` — helper to get username from `g.login_username`

### 4.3 `tools/_registry.py` (NEW — ~100 lines)
Runtime registry that tracks all loaded tools:
- `register_tool(name, metadata, blueprint)` — called by loader
- `get_tool(name)` — returns metadata + blueprint
- `list_tools()` — returns all registered tools (for Tools page API)
- `get_mcp_tool_map()` — returns `{mcp_tool_name: tool_name}` mapping
- `get_tools_by_category(category)` — filtered list

The registry also provides an API endpoint `GET /api/tools` that the frontend can call to dynamically populate the Tools page (future enhancement — not in Step 1).

### 4.4 `tools/_loader.py` (NEW — ~120 lines)
Auto-discovery engine:
- `load_tools(app)`: scan `tools/*/plugin.json`, import blueprint module, register with app
- Handles `db_init` calls
- Handles `csrf_exempt` flag
- Handles `auth_type` (applies appropriate decorator to all routes in blueprint)
- Logs discovered tools at startup
- Graceful error handling: if one tool fails to load, log warning and continue
- Returns count of loaded tools for startup logging

Key logic:
```python
def load_tools(app):
    tools_dir = Path(__file__).parent
    loaded = 0
    for tool_dir in tools_dir.iterdir():
        if not tool_dir.is_dir() or tool_dir.name.startswith("_"):
            continue
        plugin_json = tool_dir / "plugin.json"
        if not plugin_json.exists():
            continue
        metadata = json.loads(plugin_json.read_text())
        # Import blueprint module
        module = importlib.import_module(f"tools.{tool_dir.name}")
        bp = getattr(module, "blueprint")
        # Register with Flask
        app.register_blueprint(bp, url_prefix=metadata["url_prefix"])
        # Handle db_init if specified
        if metadata.get("db_init"):
            init_fn = importlib.import_module(metadata["db_init"])
            init_fn.init_db()
        # Register in registry
        register_tool(metadata["name"], metadata, bp)
        loaded += 1
    app.logger.info(f"Loaded {loaded} tool plugins")
    return loaded
```

### 4.5 `app.py` Changes (~5 lines)

Add near existing imports:
```python
from tools._loader import load_tools
```

Add after existing blueprint registrations:
```python
# Load tool plugins (auto-discovery)
load_tools(app)
```

**During migration phase**, old manual `register_blueprint()` calls remain. They are removed one-by-one as each tool migrates.

### 4.6 `tools/oitracker/plugin.json` (NEW — first migrated tool)
```json
{
    "name": "oitracker",
    "title": "OI Tracker",
    "description": "Track open interest changes across indices and stocks",
    "category": "options",
    "url_prefix": "/oitracker",
    "blueprint_module": "tools.oitracker",
    "auth_type": "session_apikey",
    "csrf_exempt": false,
    "db_init": null,
    "mcp_tools": [],
    "frontend": {
        "route": "/oitracker",
        "component": "OITracker",
        "import_path": "@/pages/OITracker",
        "nav_category": "tools"
    }
}
```

### 4.7 `tools/oitracker/__init__.py` (NEW — migrated from `blueprints/oitracker.py`)
Migrated blueprint with:
- `blueprint = Blueprint("oitracker", __name__)`
- All routes using `@tool_auth` instead of manual auth boilerplate
- All responses using `tool_success()` / `tool_error()` instead of raw `jsonify`
- Same URL structure maintained

### 4.8 `docs/tools-plugin-framework.md` (NEW — ~200 lines)
Documentation covering:
- Architecture overview
- plugin.json schema reference
- How to create a new tool
- How to migrate an existing blueprint
- How MCP tools are declared
- Testing guide

## 5. Migration Checklist (per tool)

For each tool being migrated from `blueprints/` to `tools/`:

- [ ] Read existing `blueprints/<tool>.py`
- [ ] Identify all routes, auth pattern, services called
- [ ] Create `tools/<tool>/plugin.json`
- [ ] Create `tools/<tool>/__init__.py` with blueprint + routes
- [ ] Replace manual auth with `@tool_auth`
- [ ] Replace `jsonify({"status": "success", ...})` with `tool_success()`
- [ ] Replace `jsonify({"status": "error", ...})` with `tool_error()`
- [ ] If tool has services, create `tools/<tool>/service.py` (or keep using existing `services/`)
- [ ] If tool needs DB init, configure in plugin.json
- [ ] Remove old blueprint from manual registration in app.py (keep file in blueprints/)
- [ ] Mark old blueprint file with `# DEPRECATED: migrated to tools/<tool>/`
- [ ] Test: all endpoints work identically
- [ ] Update `plugin.json` `mcp_tools` field when MCP tools are added later

## 6. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Old and new blueprints both registered → URL conflict | During Step 1, remove old `register_blueprint()` for oitracker BEFORE registering new one |
| Import failures crash app | `_loader.py` wraps each tool load in try/except, logs warning, continues |
| Auth behavior change | `tool_auth` delegates to same `check_session_validity` + `get_api_key_for_tradingview` — identical behavior |
| Frontend breaks | No URL changes — same routes, same response format |
| Session state differences | No change — same Flask session, same `session.get("user")` |

## 7. Files Modified in Step 1

| File | Action | Lines Changed |
|------|--------|--------------|
| `tools/__init__.py` | CREATE | ~5 |
| `tools/_base.py` | CREATE | ~80 |
| `tools/_registry.py` | CREATE | ~100 |
| `tools/_loader.py` | CREATE | ~120 |
| `tools/oitracker/__init__.py` | CREATE | ~100 (migrated) |
| `tools/oitracker/plugin.json` | CREATE | ~15 |
| `app.py` | MODIFY | ~5 lines added |
| `docs/tools-plugin-framework.md` | CREATE | ~200 |
| **Total** | | ~625 lines new/modified |

## 8. Verification Plan

1. **Startup**: App starts without errors, logs "Loaded 1 tool plugins"
2. **Old endpoint still works**: `POST /oitracker/api/oi` returns same response
3. **Auth**: Unauthenticated request returns 401, authenticated request succeeds
4. **Response format**: Same JSON structure as before
5. **Registry**: `list_tools()` returns oitracker with correct metadata
6. **MCP-ready**: `get_mcp_tool_map()` returns empty dict (no MCP tools yet for oitracker)
7. **No regressions**: All other tools still work via old blueprints

## 9. What's NOT in Step 1

- Migrating other tools (that's Steps 2-5)
- Dynamic frontend discovery (Tools page remains static for now)
- MCP tool registration (plugin.json declares names, actual MCP wiring is Step 7)
- Removing old blueprint files (just deprecated, not deleted)
- New UI for Tools page
