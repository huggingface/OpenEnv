# Custom Web UI

When `ENABLE_WEB_INTERFACE=true`, the server serves a default Gradio app at `/web` with Reset/Step/Get state, Quick Start, and README. Environment authors can **add** a custom tab by providing a custom Gradio builder.

## Extension point: `gradio_builder`

`create_app()` accepts an optional **`gradio_builder`** callable. When set, the UI at `/web` is built with [Gradio’s TabbedInterface](https://www.gradio.app/4.44.1/docs/gradio/tabbedinterface): the **first tab (“Playground”)** is the default OpenEnv UI, and the **second tab (“Custom”)** is the `gr.Blocks` returned by your builder. Users can switch between the default Playground and your custom interface without losing either. The same `/web/reset`, `/web/step`, `/web/state`, and `/web/metadata` API routes remain available; your custom tab can use the provided `web_manager` in-process or call those endpoints.

### Builder signature

```python
def my_gradio_builder(
    web_manager,      # WebInterfaceManager: .reset_environment(), .step_environment(), .get_state()
    action_fields,    # list[dict]: from action schema for form generation
    metadata,        # EnvironmentMetadata | None: name, readme_content, etc.
    is_chat_env,     # bool: True if single message input
    title,           # str: app title (e.g. metadata.name)
    quick_start_md,  # str: Quick Start markdown (class names already replaced)
) -> gr.Blocks:
    ...
```

Return a `gr.Blocks` instance. It is shown in the **“Custom”** tab of a tabbed interface; the **“Playground”** tab always shows the default OpenEnv UI. Core applies the same theme/css when mounting.

---

## Option 1: Add a custom tab

Provide a builder that returns your own `gr.Blocks`; it appears as the second tab (“Custom”) next to the default “Playground” tab:

```python
# server/app.py
from openenv.core.env_server.http_server import create_app
from .my_environment import MyEnvironment
from ..models import MyAction, MyObservation
from .gradio_ui import build_my_gradio_app  # your module

app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="my_env",
    gradio_builder=build_my_gradio_app,
)
```

In `server/gradio_ui.py` implement `build_my_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md)` returning a `gr.Blocks` (e.g. env-specific visualizations, extra controls). Use `web_manager.reset_environment()`, `web_manager.step_environment(action_data)`, and `web_manager.get_state()` in your Gradio event handlers. The default Playground tab remains available in the first tab.

---

## Option 2: Custom tab that wraps or reuses the default

Your builder can call the core `build_gradio_app` to get a Blocks instance and embed it inside your custom tab (e.g. in a `gr.Tabs` or as one section). That way your “Custom” tab can show both the default layout and additional content in one place.

---

## Option 3: Custom Quick Start or README only

You don’t need a custom builder only to change text. The default UI uses:

- **Quick Start**: generated from `get_quick_start_markdown(metadata, action_cls, observation_cls)` (init-style class names).
- **README**: `metadata.readme_content` (loaded from the env’s README).

So you can influence the default UI by ensuring `metadata` and README are correct. To change the Quick Start template itself (e.g. different wording or placeholders), you would use a custom `gradio_builder` that calls `build_gradio_app` with a custom `quick_start_md` string you build yourself (or by copying and adapting the default template from the core).

---

## Migration from custom HTML override (e.g. wildfire)

Environments that currently override `/web` with custom HTML (e.g. by removing the default route and adding a GET `/web` that returns HTML) should migrate to a **gradio_builder** that returns a `gr.Blocks` app. The custom UI then appears in the **“Custom”** tab alongside the default **“Playground”** tab. Benefits:

- Single, supported extension point using [TabbedInterface](https://www.gradio.app/4.44.1/docs/gradio/tabbedinterface).
- No need to remove or override routes; the default UI stays in the first tab.
- Same `/web` path; both tabs can use `web_manager` or `/web/reset`, `/web/step`, `/web/state`.

If you need a non-Gradio custom UI (e.g. static HTML/JS), you can still register your own route after `create_app` (e.g. at `/web/custom` or another path), but the main `/web` slot is the Gradio tabbed app when `ENABLE_WEB_INTERFACE=true`.

---

## Summary

| Goal                         | Approach                                                                 |
|-----------------------------|---------------------------------------------------------------------------|
| Use default UI only         | Do not pass `gradio_builder`.                                            |
| Add a custom tab            | Pass `gradio_builder=my_builder`; return your own `gr.Blocks` (shown in “Custom” tab). |
| Custom tab + default inside | In your builder, call `build_gradio_app(...)` and embed or wrap it in your Blocks. |
| Change Quick Start / README | Rely on metadata/README, or custom builder that builds custom markdown.  |

The default Playground tab is built with `openenv.core.env_server.gradio_ui.build_gradio_app`; you can import and call it with the same arguments if your custom tab needs to embed or extend it.
