import { Type } from "typebox";

const bridgeUrl = process.env.OPENENV_PI_BRIDGE_URL;

async function callBridge(method, params = {}, id = method) {
  if (!bridgeUrl) {
    throw new Error("OPENENV_PI_BRIDGE_URL is not set");
  }
  const response = await fetch(bridgeUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id,
      method,
      params,
    }),
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error?.message || response.statusText);
  }
  return payload.result || {};
}

export default async function(pi) {
  const { tools = [] } = await callBridge("tools/list");
  for (const tool of tools) {
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description || tool.name,
      parameters: Type.Unsafe(tool.inputSchema || { type: "object", properties: {} }),
      async execute(toolCallId, params) {
        const result = await callBridge(
          "tools/call",
          { name: tool.name, arguments: params || {} },
          toolCallId,
        );
        return {
          content: [{ type: "text", text: JSON.stringify(result.data ?? result) }],
          details: result,
        };
      },
    });
  }
}
