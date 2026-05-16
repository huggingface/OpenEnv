// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// swe-answer.ts — Pi extension that registers the `answer` tool.
//
// During SWE training, this extension is deployed into the sandbox at
// ~/.pi/agent/extensions/swe-answer.ts so Pi auto-discovers it on startup.
//
// The `answer` tool:
//   1. Runs swe-grade.sh (applies test patch, runs eval, reverts patch).
//   2. Parses stdout for the GRADE line (resolved=true/false).
//   3. Returns "Resolved: true/false" to Pi as a feedback signal.
//   4. Pi either stops (done) or continues if not resolved.
//
// NOTE: This extension does NOT write or read reward.txt.  The
// authoritative training reward is computed host-side by
// SWESession.verify() using swebench grading.  The result returned
// here is only a fast feedback signal for the agent.
//
// Environment variables (set by the harness):
//   SWE_GRADE_SCRIPT  — path to swe-grade.sh
//   SWE_INSTANCE_ID   — task instance id (for logging)

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execSync } from "child_process";

export default function (pi: ExtensionAPI) {
  const GRADE_SCRIPT = process.env.SWE_GRADE_SCRIPT || "/home/user/swe-grade.sh";
  const INSTANCE_ID = process.env.SWE_INSTANCE_ID || "unknown";

  pi.registerTool({
    name: "answer",
    label: "Submit Answer",
    description:
      "Submit your solution for grading. Runs the test suite against your changes " +
      "and returns whether the issue is resolved. Call this when you believe your " +
      "fix is complete. Returns the grading result (Resolved: true/false).",
    parameters: Type.Object({}),

    async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
      let gradeOutput = "";
      let error: string | null = null;

      try {
        gradeOutput = execSync(`bash ${GRADE_SCRIPT}`, {
          encoding: "utf-8",
          timeout: 300_000, // 5 minutes
          stdio: ["pipe", "pipe", "pipe"],
          env: { ...process.env },
        });
      } catch (e: any) {
        // Grade script may exit non-zero if tests fail — that's expected.
        gradeOutput = (e.stdout || "") + "\n" + (e.stderr || "");
        error = e.message?.slice(0, 500) || "grading failed";
      }

      // Parse the GRADE line from stdout.
      const gradeLine = gradeOutput.split("\n").find(l => l.startsWith("GRADE:")) || "";
      const resolved = gradeLine.includes("resolved=true");

      const summary = resolved
        ? `✅ Resolved: true`
        : `❌ Resolved: false`;

      const details = [
        `Instance: ${INSTANCE_ID}`,
        `Resolved: ${resolved}`,
        error ? `Error: ${error}` : null,
        `--- Grade Output (last 2000 chars) ---`,
        gradeOutput.slice(-2000),
      ]
        .filter(Boolean)
        .join("\n");

      return {
        content: [{ type: "text", text: `${summary}\n\n${details}` }],
        details: { resolved, instance_id: INSTANCE_ID },
      };
    },
  });
}
