import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = fileURLToPath(new URL("../../..", import.meta.url));
// Non-CPU command snapshots from upstream 626b95d; no Git history needed in CI.
const commandDigests = {
  "ibm-granite/granite-4.2.jsx": "8c5c901d865063c5968cec87b047c8a4b453e66126439d580aaf56ff4a901481",
  "deepreinforce-ai/ornith-1.0.jsx": "2eb0869b23811c8f45c25ecd6c1504ac45c075a8477031333e79402680829a43",
  "ornith-ai/ornith-1.5.jsx": "69a8ef736f274775c2410dec39ebee6fafe040f774463429a205cec75a9bc0ed",
  "IFM/k2-horizon.jsx": "ca80fa456d250417f6eb32cae409f307c24c904d54c7efb0cd633cfcacc08a30",
  "IFM/jais-3.jsx": "7581cf5bea5568989223471c115ab3bb3208a5574f56d6a18d09fb94afc786c0",
  "poolside/laguna-xs21.jsx": "adbe322201ea8aa8dc7043e98506945efa3bdce7545fb2e01ad618788c02c949",
  "openbmb/minicpm5-2b.jsx": "5c6d6cdae9c6c7d3de43073745ab1ca6c32d43310df0767dae20c3bd256a60f6",
};
const engineSource = readFileSync(join(root, "docs/src/snippets/_deployment.jsx"), "utf8");
const start = engineSource.indexOf("  // ==== COMMAND ENGINE ====");
const end = engineSource.indexOf("  // ==== end COMMAND ENGINE ====");
assert.ok(start > 0 && end > start);
const engine = new Function(
  "config", `${engineSource.slice(start, end)}\nreturn { renderCommand, findCell };`,
);
const load = async (source) =>
  (await import("data:text/javascript," + encodeURIComponent(source))).config;
const modes = (config, selection) =>
  (typeof config.runModes === "function" ? config.runModes(selection) : config.runModes)
  || ["python", "docker"];

for (const [path, digest] of Object.entries(commandDigests)) {
  const relative = `docs/src/snippets/configs/${path}`;
  const config = await load(readFileSync(join(root, relative), "utf8"));
  const currentEngine = engine(config);
  const env = Object.fromEntries(
    Object.entries(config.placeholders || {}).map(([key, spec]) => [key, spec.default ?? ""]),
  );
  const overlays = Object.fromEntries(
    (config.overlayDims || []).map((spec) => [spec.id, spec.default ?? spec.options?.[0]?.id ?? ""]),
  );

  test(`${path}: existing platforms retain byte-identical commands`, () => {
    const snapshots = [];
    for (const cell of config.cells.filter((cell) => cell.match.hw !== "xeon")) {
      const base = { ...cell.match, ...overlays };
      const selections = [
        base,
        ...(config.overlayDims || []).flatMap((spec) =>
          (spec.options || []).map((option) => ({ ...base, [spec.id]: option.id }))),
      ];
      for (const selection of selections) {
        const runModes = modes(config, selection);
        snapshots.push([selection, runModes, runModes.map((mode) =>
          currentEngine.renderCommand(currentEngine.findCell(config.cells, selection), selection, env, mode),
        )]);
      }
    }
    assert.equal(createHash("sha256").update(JSON.stringify(snapshots)).digest("hex"), digest);
  });

  test(`${path}: Xeon recipes are CPU-only and cannot inherit Playground overrides`, () => {
    const cells = config.cells.filter((cell) => cell.match.hw === "xeon");
    assert.ok(cells.length > 0);
    for (const cell of cells) {
      const selection = { ...cell.match, ...overlays };
      assert.deepEqual(modes(config, selection), ["python"]);
      assert.equal(cell.verified, false);
      assert.equal(cell.verificationStatus, "unverified");
      const command = currentEngine.renderCommand(cell, selection, env, "python");
      assert.match(command, /--device cpu/);
      assert.match(command, /--disable-overlap-schedule/);
      assert.doesNotMatch(command, /docker|--ep|--speculative|--kv-cache-dtype fp8/);
      for (const spec of config.overlayDims || []) {
        assert.equal(spec.showWhen?.(selection), false);
        for (const option of spec.options || []) {
          assert.equal(currentEngine.renderCommand(
            cell, { ...selection, [spec.id]: option.id }, env, "python",
          ), command);
        }
      }
      if (path === "openbmb/minicpm5-2b.jsx") {
        assert.match(command, /--default-chat-template-kwargs '\{"enable_thinking":false\}'/);
        assert.doesNotMatch(command, /--reasoning-parser|--tool-call-parser/);
      }
      assert.deepEqual(config.playgroundFeatures.normalizeDeltas(
        { attention: { tp: 8 }, speculative: { mode: "dflash" } }, {}, selection,
      ), {});
    }
  });
}
