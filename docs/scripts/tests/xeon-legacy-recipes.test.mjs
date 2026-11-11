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
      handleCheckboxChange: typeof handleCheckboxChange === "undefined" ? null : handleCheckboxChange,
      resolveItems: typeof resolveItems === "undefined" ? null : resolveItems };
  `)(useState, () => {});
  return { ...api, state: () => state };
}

test("DeepSeek-V3.1 adds only its validated TP6 CPU baseline and preserves existing recipes", () => {
  const entries = [];
  for (const hardware of ["h200", "b200", "mi300x", "mi325x", "mi355x", "xeon"]) {
    for (const modelname of hardware === "xeon" ? ["v31terminusint8"] : ["v31", "v31terminus"]) {
      const strategies = [["tp"], ["tp", "dp"],
        ...(hardware === "xeon" ? [] : [["tp", "ep"], ["tp", "mtp"], ["tp", "dp", "ep", "mtp"]])];
      for (const strategy of strategies) {
        for (const reasoningParser of ["disabled", "enabled"]) {
          for (const toolcall of ["disabled", "enabled"]) {
            const selection = { hardware, modelname, strategy, reasoningParser, toolcall };
            entries.push([selection, builder("deepseek-v31", selection).generateCommand()]);
          }
        }
      }
    }
  }
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "e7e6aa7d2d49db81d4272386c124a0dbfb8cfe92b4593dfc0313638af877df7c");
  const selection = {
    hardware: "xeon", modelname: "v31", strategy: ["tp", "dp", "ep", "mtp"],
    reasoningParser: "enabled", toolcall: "enabled",
  };
  const api = builder("deepseek-v31", selection);
  const command = api.generateCommand();
  assert.match(command, /^SGLANG_USE_CPU_ENGINE=1 python3 -m sglang.launch_server/);
  assert.match(command, /--model deepseek-ai\/DeepSeek-V3\.1(?:\s|$)/);
  assert.match(command, /--device cpu/);
  assert.match(command, /--tp 6/);
  assert.match(command, /--dtype bfloat16/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.doesNotMatch(command, /Terminus|--dp|--ep|--quantization|--speculative|--reasoning-parser|--tool-call-parser/);
  for (const key of ["strategy", "reasoningParser", "toolcall"]) {
    assert.equal(api.options[key].condition(selection), false);
  }
  assert.equal(api.options.modelname.items.find((item) => item.id === "v31terminus")
    .disabledWhen(selection), true);
  assert.match(builder("deepseek-v31", { ...selection, modelname: "v31terminus" })
    .generateCommand(), /^# Error:/);
  const switcher = builder("deepseek-v31", { ...selection, modelname: "v31terminusint8" });
  switcher.handleRadioChange("modelname", "v31");
  const baseline = {
    ...selection, strategy: ["tp"], reasoningParser: "disabled", toolcall: "disabled",
  };
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleRadioChange("modelname", "v31terminus");
  switcher.handleRadioChange("reasoningParser", "enabled");
  switcher.handleCheckboxChange("strategy", "dp", true);
  assert.deepEqual(switcher.state(), baseline);
});

test("GLM-4.5V preserves GPU commands and exposes only its validated BF16 CPU baseline", () => {
  const entries = [];
  for (const hardware of ["b200", "h100", "h200", "mi300x", "mi325x", "mi355x"]) {
    for (const quantization of ["bf16", "fp8"]) {
      for (const reasoning of ["enabled", "disabled"]) {
        for (const toolcall of ["enabled", "disabled"]) {
          const selection = { hardware, quantization, reasoning, toolcall };
          entries.push([selection, builder("glm-45v", selection).generateCommand()]);
        }
      }
    }
  }
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "62cd7d44cc3ba3c0e00efc88af8e8cb84e46dfb9234509641b3e9f3bcecc1fcb");
  const selection = {
    hardware: "xeon", quantization: "bf16", reasoning: "enabled", toolcall: "enabled",
  };
  const api = builder("glm-45v", selection);
  const command = api.generateCommand();
  assert.match(command, /^SGLANG_USE_CPU_ENGINE=1 python -m sglang.launch_server/);
  assert.match(command, /--model zai-org\/GLM-4\.5V(?:\s|$)/);
  assert.match(command, /--device cpu/);
  assert.match(command, /--tp 4/);
  assert.match(command, /--dtype bfloat16/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.doesNotMatch(command, /CUDA|FP8|--reasoning-parser|--tool-call-parser|--mm-/);
  assert.ok(api.options.hardware.items.some((item) => item.id === "xeon"));
  for (const key of ["quantization", "reasoning", "toolcall"]) {
    assert.equal(api.options[key].condition(selection), false);
  }
  assert.match(builder("glm-45v", { ...selection, quantization: "fp8" })
    .generateCommand(), /^# Error:/);
  const switcher = builder("glm-45v", { ...selection, hardware: "b200", quantization: "fp8" });
  switcher.handleRadioChange("hardware", "xeon");
  const baseline = {
    ...selection, reasoning: "disabled", toolcall: "disabled",
  };
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleRadioChange("quantization", "fp8");
  switcher.handleRadioChange("reasoning", "enabled");
  switcher.handleRadioChange("toolcall", "enabled");
  assert.deepEqual(switcher.state(), baseline);
});

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

test("GLM4.6V preserves GPU commands and exposes only validated BF16 CPU baselines", () => {
  const entries = [];
  for (const hardware of ["b200", "b300", "h100", "h200", "mi300x", "mi325x", "mi355x"]) {
    for (const modelsize of ["106b", "9b"]) {
      for (const quantization of ["bf16", "fp8"]) {
        for (const reasoning of ["enabled", "disabled"]) {
          for (const toolcall of ["enabled", "disabled"]) {
            const selection = { hardware, modelsize, quantization, reasoning, toolcall };
            entries.push([selection, builder("glm-46v", selection).generateCommand()]);
          }
        }
      }
    }
  }
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "babf679f8b7af75024b886ae174097eb134a2f668c32a85d1ed42595d303195f");
  const selection = {
    hardware: "xeon", modelsize: "9b", quantization: "bf16",
    reasoning: "enabled", toolcall: "enabled",
  };
  const api = builder("glm-46v", selection);
  const command = api.generateCommand();
  assert.match(command, /--model zai-org\/GLM-4.6V-Flash/);
  assert.match(command, /--device cpu/);
  assert.match(command, /--tp 2/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.doesNotMatch(command, /CUDA|--reasoning-parser|--tool-call-parser|--mm-enable/);
  for (const key of ["quantization", "reasoning", "toolcall"]) {
    assert.equal(api.options[key].condition(selection), false);
  }
  const switcher = builder("glm-46v", { ...selection, hardware: "b200", modelsize: "106b", quantization: "fp8" });
  switcher.handleRadioChange("hardware", "xeon");
  assert.deepEqual(switcher.state(), {
    ...selection, reasoning: "disabled", toolcall: "disabled",
  });
  switcher.handleRadioChange("modelsize", "106b");
  assert.equal(switcher.state().modelsize, "106b");
  const fullCommand = builder("glm-46v", switcher.state()).generateCommand();
  assert.match(fullCommand, /--model zai-org\/GLM-4.6V \\/);
  assert.match(fullCommand, /--tp 4/);
  assert.match(fullCommand, /--dtype bfloat16/);
  assert.match(fullCommand, /--max-total-tokens 8192/);
  assert.doesNotMatch(fullCommand, /Flash|CUDA|--reasoning-parser|--tool-call-parser|--mm-enable/);
  for (const delta of [{ modelsize: "unknown" }, { quantization: "fp8" }]) {
    assert.match(builder("glm-46v", { ...selection, ...delta }).generateCommand(), /^# Error:/);
  }
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
