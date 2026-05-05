# Core API

The `openenv.core` package provides the core abstractions for building and running environments. For an end-to-end tutorial on building environments with OpenEnv, see the [building an environment](auto_getting_started/environment-builder.md) guide.

If you are trying to understand when OpenEnv exposes the training loop versus direct MCP access, see the [simulation vs production mode](simulation-vs-production.md) guide.

For a high-level explanation of how MCP-backed environments move through `step()`, `step_async()`, and convenience tool helpers, see the [MCP environment lifecycle](mcp-environment-lifecycle.md) guide.

## Server

### Environment server primitives

```{eval-rst}
.. automodule:: openenv.core.env_server.interfaces
   :members:
   :undoc-members:
   :show-inheritance:
```

### Types

```{eval-rst}
.. automodule:: openenv.core.env_server.types
   :members:
   :undoc-members:
   :show-inheritance:
```

### Exceptions

```{eval-rst}
.. automodule:: openenv.core.env_server.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
```

### HTTP server utilities

```{eval-rst}
.. automodule:: openenv.core.env_server.http_server
   :members:
   :undoc-members:
   :show-inheritance:
```

### Web interface helpers

```{eval-rst}
.. automodule:: openenv.core.env_server.web_interface
   :members:
   :undoc-members:
   :show-inheritance:
```

### Serialization

```{eval-rst}
.. automodule:: openenv.core.env_server.serialization
   :members:
   :undoc-members:
   :show-inheritance:
```

### Transforms

```{eval-rst}
.. automodule:: openenv.core.env_server.base_transforms
   :members:
   :undoc-members:
   :show-inheritance:
```

### Route configuration

```{eval-rst}
.. automodule:: openenv.core.env_server.route_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## Clients

### Base client

```{eval-rst}
.. automodule:: openenv.core.env_client
   :members:
   :undoc-members:
   :show-inheritance:
```

### Synchronous client

```{eval-rst}
.. automodule:: openenv.core.sync_client
   :members:
   :undoc-members:
   :show-inheritance:
```

### Generic client

```{eval-rst}
.. automodule:: openenv.core.generic_client
   :members:
   :undoc-members:
   :show-inheritance:
```

### LLM client

```{eval-rst}
.. automodule:: openenv.core.llm_client
   :members:
   :undoc-members:
   :show-inheritance:
```

### Shared dataclasses

```{eval-rst}
.. automodule:: openenv.core.client_types
   :members:
   :undoc-members:
   :show-inheritance:
```

## MCP (Model Context Protocol)

### MCP environment

```{eval-rst}
.. automodule:: openenv.core.env_server.mcp_environment
   :members:
   :undoc-members:
   :show-inheritance:
```

### MCP types

```{eval-rst}
.. automodule:: openenv.core.env_server.mcp_types
   :members:
   :undoc-members:
   :show-inheritance:
```

### MCP client

```{eval-rst}
.. automodule:: openenv.core.mcp_client
   :members:
   :undoc-members:
   :show-inheritance:
```

## Rubrics

```{eval-rst}
.. automodule:: openenv.core.rubrics.base
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: openenv.core.rubrics.containers
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: openenv.core.rubrics.trajectory
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: openenv.core.rubrics.llm_judge
   :members:
   :undoc-members:
   :show-inheritance:
```

## Tools

```{eval-rst}
.. automodule:: openenv.core.tools.git_server_client
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: openenv.core.tools.local_python_executor
   :members:
   :undoc-members:
   :show-inheritance:
```

## Container providers

```{eval-rst}
.. automodule:: openenv.core.containers.runtime.providers
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: openenv.core.containers.runtime.uv_provider
   :members:
   :undoc-members:
   :show-inheritance:
```
