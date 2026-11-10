import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = fileURLToPath(new URL("../../..", import.meta.url));
const paths = [
  "ibm-granite/granite-4.2.jsx",
  "deepreinforce-ai/ornith-1.0.jsx",
  "ornith-ai/ornith-1.5.jsx",
  "IFM/k2-horizon.jsx",
  "poolside/laguna-xs21.jsx",
];
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

for (const path of paths) {
  const relative = `docs/src/snippets/configs/${path}`;
  const config = await load(readFileSync(join(root, relative), "utf8"));
  const original = await load(execFileSync(
    "git", ["show", `626b95d503321e01bd145b12b2f82aac43d78fc0:${relative}`],
    { cwd: root, encoding: "utf8" },
  ));
  const currentEngine = engine(config);
  const originalEngine = engine(original);
  const env = Object.fromEntries(
    Object.entries(config.placeholders || {}).map(([key, spec]) => [key, spec.default ?? ""]),
  );
  const overlays = Object.fromEntries(
    (config.overlayDims || []).map((spec) => [spec.id, spec.default ?? spec.options?.[0]?.id ?? ""]),
  );

  test(`${path}: existing platforms retain byte-identical commands`, () => {
    for (const cell of original.cells) {
      const base = { ...cell.match, ...overlays };
      const selections = [
        base,
        ...(config.overlayDims || []).flatMap((spec) =>
          (spec.options || []).map((option) => ({ ...base, [spec.id]: option.id }))),
      ];
      for (const selection of selections) {
        assert.deepEqual(modes(config, selection), modes(original, selection));
        for (const mode of modes(original, selection)) {
          assert.equal(
            currentEngine.renderCommand(currentEngine.findCell(config.cells, selection), selection, env, mode),
            originalEngine.renderCommand(originalEngine.findCell(original.cells, selection), selection, env, mode),
          );
        }
      }
    }
  });

  test(`${path}: Xeon recipes are CPU-only and cannot inherit Playground overrides`, () => {
    const cells = config.cells.filter((cell) => cell.match.hw === "xeon");
    for (const cell of cells) {
      const selection = { ...cell.match, ...overlays };
      assert.deepEqual(modes(config, selection), ["python"]);
      assert.equal(cell.verified, false);
      assert.equal(cell.verificationStatus, "unverified");
      const command = currentEngine.renderCommand(cell, selection, env, "python");
      assert.match(command, /--device cpu/);
      assert.match(command, /--disable-overlap-schedule/);
      assert.doesNotMatch(command, /docker|--ep|--speculative|--kv-cache-dtype fp8/);
      assert.deepEqual(config.playgroundFeatures.normalizeDeltas(
        { attention: { tp: 8 }, speculative: { mode: "dflash" } }, {}, selection,
      ), {});
    }
  });
}
