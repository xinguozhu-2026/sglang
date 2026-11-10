import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

function builder(name, selection) {
  const source = readFileSync(new URL(
    `../../src/snippets/autoregressive/${name}-deployment.jsx`, import.meta.url,
  ), "utf8");
  const body = source.slice(source.indexOf("{") + 1, source.indexOf("  const containerStyle"));
  let state = selection;
  let calls = 0;
  const useState = () => calls++ === 0
    ? [selection, (update) => { state = update(state); }]
    : [false, () => {}];
  const api = new Function("useState", "useEffect", `${body}
    return { generateCommand, handleRadioChange, options,
      resolveItems: typeof resolveItems === "undefined" ? null : resolveItems };
  `)(useState, () => {});
  return { ...api, state: () => state };
}

test("legacy builders preserve existing platform commands", () => {
  const entries = [];
  for (const hardware of ["h200", "b200"]) {
    for (const quantization of ["bf16", "fp8", "nvfp4"]) {
      for (const reasoning of ["enabled", "disabled"]) {
        for (const toolcall of ["enabled", "disabled"]) {
          for (const dpAttention of ["enabled", "disabled"]) {
            const selection = { hardware, quantization, reasoning, toolcall, dpAttention };
            entries.push([selection, builder("laguna-xs2", selection).generateCommand()]);
          }
        }
      }
    }
  }
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "32fd81a6353d5bb9f7ce0b233e65f896eb95b826af5d16e22206208ad2f37439");
  entries.length = 0;
  for (const hardware of ["h200", "b200", "b300", "mi300x", "arc_b", "xeon"]) {
    for (const modelSize of ["e2b", "e4b", "12b", "31b", "26b-a4b"]) {
      for (const checkpoint of hardware === "xeon" ? ["standard"] : ["standard", "qat"]) {
        if (hardware === "xeon" && modelSize === "12b") continue;
        for (const reasoning of ["enabled", "disabled"]) {
          for (const toolcall of ["enabled", "disabled"]) {
            for (const speculative of ["enabled", "disabled"]) {
              const selection = { hardware, modelSize, checkpoint, reasoning, toolcall, speculative };
              entries.push([selection, builder("gemma4", selection).generateCommand(selection)]);
            }
          }
        }
      }
    }
  }
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "cdae95e0ed878d4c4894f9851ee0d356006b770d5b89fffc54adc431b2e05241");
});

test("Gemma 12B Xeon exposes only the full-checkpoint validated baseline", () => {
  const selection = {
    hardware: "xeon", modelSize: "12b", checkpoint: "standard",
    reasoning: "enabled", toolcall: "enabled", speculative: "enabled",
  };
  const api = builder("gemma4", selection);
  const command = api.generateCommand(selection);
  assert.match(command, /--model google\/gemma-4-12B-it/);
  assert.match(command, /--device cpu/);
  assert.match(command, /--tp 6/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.doesNotMatch(command, /--speculative|--reasoning-parser|--tool-call-parser|--mem-fraction/);
  assert.equal(api.options.hardware.getDynamicItems(selection)
    .find((item) => item.id === "xeon").disabled, false);
  for (const key of ["reasoning", "toolcall", "speculative"]) {
    assert.equal(api.options[key].condition(selection), false);
  }
  const switcher = builder("gemma4", { ...selection, hardware: "h200", checkpoint: "qat" });
  switcher.handleRadioChange("hardware", "xeon");
  assert.deepEqual(switcher.state(), {
    ...selection, checkpoint: "standard",
    reasoning: "disabled", toolcall: "disabled", speculative: "disabled",
  });
});

test("GLM-OCR adds only the validated CPU baseline and preserves GPU commands", () => {
  for (const hardware of ["h100", "h200", "b200"]) {
    for (const strategy of [[], ["mtp"]]) {
      const selection = { hardware, strategy };
      const expected = "SGLANG_USE_CUDA_IPC_TRANSPORT=1 python -m sglang.launch_server \\\n  --model zai-org/GLM-OCR" +
        (strategy.length ? " \\\n  --speculative-algorithm EAGLE \\\n  --speculative-num-steps 3 \\\n  --speculative-eagle-topk 1 \\\n  --speculative-num-draft-tokens 4" : "");
      assert.equal(builder("glm-ocr", selection).generateCommand(), expected);
    }
  }
  const selection = { hardware: "xeon", strategy: ["mtp"] };
  const api = builder("glm-ocr", selection);
  const command = api.generateCommand();
  assert.match(command, /--model zai-org\/GLM-OCR/);
  assert.match(command, /--device cpu/);
  assert.match(command, /--tp 6/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.doesNotMatch(command, /CUDA|--speculative/);
  assert.equal(api.options.strategy.condition(selection), false);
  const switcher = builder("glm-ocr", { ...selection, hardware: "h100" });
  switcher.handleRadioChange("hardware", "xeon");
  assert.deepEqual(switcher.state(), { hardware: "xeon", strategy: [] });
});

test("Laguna Xeon disables and clears unvalidated options", () => {
  const selection = {
    hardware: "xeon", quantization: "bf16",
    reasoning: "enabled", toolcall: "enabled", dpAttention: "enabled",
  };
  const api = builder("laguna-xs2", selection);
  const command = api.generateCommand();
  assert.match(command, /--device cpu/);
  assert.match(command, /--tp 4/);
  assert.doesNotMatch(command, /--attn-dp|--reasoning-parser|--tool-call-parser|--speculative/);
  for (const [key, option] of Object.entries(api.options)) {
    if (key === "hardware") continue;
    const enabled = api.resolveItems(option, selection).filter((item) => !item.disabled);
    assert.deepEqual(enabled.map((item) => item.id), [key === "quantization" ? "bf16" : "disabled"]);
  }
  const switcher = builder("laguna-xs2", { ...selection, hardware: "b200", quantization: "fp8" });
  switcher.handleRadioChange("hardware", "xeon");
  assert.deepEqual(switcher.state(), {
    hardware: "xeon", quantization: "bf16",
    reasoning: "disabled", toolcall: "disabled", dpAttention: "disabled",
  });
});

test("Gemma Xeon QAT uses only the validated baseline", () => {
  for (const modelSize of ["e2b", "e4b", "31b"]) {
    const selection = {
      hardware: "xeon", modelSize, checkpoint: "qat",
      reasoning: "enabled", toolcall: "enabled", speculative: "enabled",
    };
    const api = builder("gemma4", selection);
    const command = api.generateCommand(selection);
    assert.match(command, /--device cpu/);
    assert.match(command, /--tp 4/);
    assert.match(command, /-qat-q4_0-unquantized/);
    if (modelSize === "31b") {
      assert.match(command, /--model google\/gemma-4-31B-it-qat-q4_0-unquantized/);
      assert.match(command, /--max-total-tokens 8192/);
    }
    assert.doesNotMatch(command, /--speculative|--reasoning-parser|--tool-call-parser/);
    for (const key of ["reasoning", "toolcall", "speculative"]) {
      assert.equal(api.options[key].condition(selection), false);
    }
    const switcher = builder("gemma4", { ...selection, hardware: "h200" });
    switcher.handleRadioChange("hardware", "xeon");
    assert.equal(switcher.state().checkpoint, "qat");
    assert.equal(switcher.state().speculative, "disabled");
  }
  const invalid = { hardware: "xeon", modelSize: "12b", checkpoint: "qat" };
  assert.match(builder("gemma4", invalid).generateCommand(invalid), /^# Error:/);
});

test("Intern-S2-Preview preserves GPU commands and gates CPU options", () => {
  const entries = [];
  for (const reasoning of ["enabled", "disabled"]) {
    for (const toolcall of ["enabled", "disabled"]) {
      for (const mtp of ["enabled", "disabled"]) {
        const selection = { hardware: "h200", reasoning, toolcall, mtp };
        entries.push([selection, builder("intern-s2-preview", selection).generateCommand()]);
      }
    }
  }
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "3067e357c77e45bcd7573298f7ab1bb224ad9eae8b10c2029a6befc67802e17a");
  const selection = {
    hardware: "xeon", reasoning: "enabled", toolcall: "enabled", mtp: "enabled",
  };
  const api = builder("intern-s2-preview", selection);
  assert.match(api.generateCommand(), /--device cpu/);
  assert.match(api.generateCommand(), /--tp 4/);
  assert.doesNotMatch(api.generateCommand(), /--speculative|--reasoning-parser|--tool-call-parser/);
  api.handleRadioChange("hardware", "xeon");
  assert.deepEqual(api.state(), {
    hardware: "xeon", reasoning: "disabled", toolcall: "disabled", mtp: "disabled",
  });
});
