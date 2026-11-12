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
      getInitialState: typeof getInitialState === "undefined" ? null : getInitialState,
      computeConstraints: typeof computeConstraints === "undefined" ? null : computeConstraints,
      handleCheckboxChange: typeof handleCheckboxChange === "undefined" ? null : handleCheckboxChange,
      resolveItems: typeof resolveItems === "undefined" ? null : resolveItems };
  `)(useState, () => {});
  return { ...api, state: () => state };
}

test("MiMo-V2.5 preserves CUDA/TPU commands and exposes only its validated Pro TP4 CPU topology", () => {
  const toggles = ["eagleMtp", "dpAttention", "expertParallelism", "deepep", "reasoningParser", "toolcall"];
  const entries = [];
  for (const hardware of ["h200", "h100", "b200", "gb300", "tpu-v7x", "tpu-v6e"]) {
    for (const modelVariant of hardware.startsWith("tpu") ? ["pro"] : ["pro", "base"]) {
      for (let mask = 0; mask < 64; mask++) {
        const selection = { hardware, modelVariant };
        toggles.forEach((key, index) => {
          selection[key] = mask & (1 << index) ? "enabled" : "disabled";
        });
        entries.push([selection, builder("mimo-v25", selection).generateCommand()]);
      }
    }
  }
  assert.equal(entries.length, 640);
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "6fe001624987c47f80af468eb93f7cd450d40831268b839a8f3b2a82beea729b");
  const stale = { hardware: "xeon", modelVariant: "pro",
    ...Object.fromEntries(toggles.map((name) => [name, "enabled"])) };
  const api = builder("mimo-v25", stale);
  const command = api.generateCommand();
  assert.match(command, /^SGLANG_USE_CPU_ENGINE=1 SGLANG_CPU_OMP_THREADS_BIND='[^']+' sglang serve/);
  assert.match(command, /--model XiaomiMiMo\/MiMo-V2\.5-Pro \\\n  --device cpu \\\n  --disable-overlap-schedule/);
  assert.match(command, /--tp 4/);
  assert.match(command, /--dtype bfloat16/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.match(command, /--max-mamba-cache-size 64/);
  assert.match(command, /--trust-remote-code/);
  assert.match(command, /--port 30000/);
  assert.doesNotMatch(command, /docker|--model-path|--speculative|--attn-dp|--ep |--reasoning-parser|--tool-call-parser|--moe-a2a|--attention-backend/);
  const groups = command.match(/SGLANG_CPU_OMP_THREADS_BIND='([^']+)'/)[1]
    .split("|").map((group) => group.split(",").map(Number));
  assert.deepEqual(groups, Array.from({ length: 4 }, (_, rank) =>
    Array.from({ length: 48 }, (_, index) => rank + 4 * index)));
  assert.equal(new Set(groups.flat()).size, 192);
  const constraints = api.computeConstraints("pro", "xeon");
  for (const key of ["modelVariant", ...toggles]) {
    const items = api.resolveItems(api.options[key], constraints);
    assert.ok(items.filter((item) => item.id !== constraints[key].force)
      .every((item) => item.disabled && item.disabledReason));
  }
  assert.match(builder("mimo-v25", { ...stale, modelVariant: "base" }).generateCommand(), /^# Error:/);
  const baseline = { hardware: "xeon", modelVariant: "pro",
    ...Object.fromEntries(toggles.map((name) => [name, "disabled"])) };
  const switcher = builder("mimo-v25", { ...stale, hardware: "b200", modelVariant: "base" });
  switcher.handleRadioChange("hardware", "xeon");
  assert.deepEqual(switcher.state(), baseline);
  for (const key of toggles) switcher.handleRadioChange(key, "enabled");
  switcher.handleRadioChange("modelVariant", "base");
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleRadioChange("hardware", "b200");
  assert.deepEqual(switcher.state(), { ...baseline, hardware: "b200" });
});

test("Kimi-K2 preserves GPU commands and exposes only its validated TP6 Instruct CPU recipe", () => {
  const entries = [];
  for (const hardware of ["h200", "b200", "b300", "mi300x", "mi325x", "mi355x"]) {
    for (const modelname of ["instruct", "thinking"]) {
      for (let mask = 0; mask < 8; mask++) {
        for (const reasoning of ["disabled", "enabled"]) {
          for (const toolcall of ["disabled", "enabled"]) {
            const selection = {
              hardware, modelname, strategy: ["tp", "dp", "ep"].filter((_, i) => mask & (1 << i)),
              reasoning, toolcall,
            };
            entries.push([selection, builder("kimi-k2", selection).generateCommand(selection)]);
          }
        }
      }
    }
  }
  assert.equal(entries.length, 384);
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "d4bca90229294cdd99b468493b29475277747672e844cac9f4571f1e4a00592a");
  const baseline = {
    hardware: "xeon", modelname: "instruct", strategy: ["tp"],
    reasoning: "disabled", toolcall: "disabled",
  };
  const api = builder("kimi-k2", baseline);
  assert.deepEqual(api.getInitialState(), { ...baseline, hardware: "h200" });
  const stale = { ...baseline, strategy: ["tp", "dp", "ep"], reasoning: "enabled", toolcall: "enabled" };
  const command = api.generateCommand(stale);
  assert.match(command, /^SGLANG_USE_CPU_ENGINE=1 python3 -m sglang.launch_server/);
  assert.match(command, /--model moonshotai\/Kimi-K2-Instruct(?:\s|$)/);
  assert.match(command, /--device cpu \\\n  --disable-overlap-schedule/);
  assert.match(command, /--tp 6/);
  assert.match(command, /--dtype bfloat16/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.match(command, /--max-mamba-cache-size 64/);
  assert.match(command, /--trust-remote-code/);
  assert.doesNotMatch(command, /docker|--model-path|--attn-dp|--ep|--reasoning-parser|--tool-call-parser|--speculative|--kv-cache-dtype/);
  assert.match(api.generateCommand({ ...baseline, modelname: "thinking" }), /^# Error:/);
  assert.equal(api.options.modelname.getDynamicItems(baseline)
    .find((item) => item.id === "thinking").disabled, true);
  for (const key of ["strategy", "reasoning", "toolcall"]) {
    assert.equal(api.options[key].condition(baseline), false);
  }
  const switcher = builder("kimi-k2", { ...stale, hardware: "h200", modelname: "thinking" });
  switcher.handleRadioChange("hardware", "xeon");
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleRadioChange("modelname", "thinking");
  switcher.handleRadioChange("reasoning", "enabled");
  switcher.handleRadioChange("toolcall", "enabled");
  switcher.handleCheckboxChange("strategy", "dp", true);
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleRadioChange("hardware", "b200");
  assert.deepEqual(switcher.state(), { ...baseline, hardware: "b200" });
  assert.equal(api.options.modelname.getDynamicItems({ ...baseline, hardware: "b200" })
    .find((item) => item.id === "thinking").disabled, undefined);
});

test("MiMo-V2-Flash preserves all GPU commands and exposes only validated TP4 CPU baseline", () => {
  const entries = [];
  for (const hardware of ["h200", "h100", "mi355x"]) {
    for (let mask = 0; mask < 16; mask++) {
      for (let reasoningMask = 0; reasoningMask < 4; reasoningMask++) {
        const selection = {
          hardware, modelname: "mimo-v2-flash",
          strategy: ["tp", "dp", "mtp", "optimization"].filter((_, i) => mask & (1 << i)),
          reasoning: ["reasoning", "toolcall"].filter((_, i) => reasoningMask & (1 << i)),
        };
        entries.push([selection, builder("mimo-v2-flash", selection).generateCommand()]);
      }
    }
  }
  assert.equal(createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    "d04ea751428e24506dfeb379ce29069b4d109842d1153953383364c37230cbde");
  const selection = {
    hardware: "xeon", modelname: "mimo-v2-flash",
    strategy: ["tp", "dp", "mtp", "optimization"], reasoning: ["reasoning", "toolcall"],
  };
  const api = builder("mimo-v2-flash", selection);
  const command = api.generateCommand();
  assert.match(command, /^SGLANG_USE_CPU_ENGINE=1 sglang serve/);
  assert.match(command, /--model XiaomiMiMo\/MiMo-V2-Flash \\\n  --device cpu \\\n  --disable-overlap-schedule/);
  assert.match(command, /--tp-size 4/);
  assert.match(command, /--dtype bfloat16/);
  assert.match(command, /--max-total-tokens 8192/);
  assert.match(command, /--max-mamba-cache-size 64/);
  assert.match(command, /--port 30000/);
  assert.match(command, /--trust-remote-code/);
  assert.doesNotMatch(command, /docker|CUDA|--model-path|--attn-dp|--speculative|--reasoning-parser|--tool-call-parser|--attention-backend|--mem-fraction/);
  const switcher = builder("mimo-v2-flash", { ...selection, hardware: "h200" });
  switcher.handleRadioChange("hardware", "xeon");
  const baseline = { ...selection, strategy: ["tp"], reasoning: [] };
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleCheckboxChange("strategy", "mtp", true);
  switcher.handleCheckboxChange("reasoning", "reasoning", true);
  switcher.handleRadioChange("modelname", "other");
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleRadioChange("hardware", "h100");
  assert.deepEqual(switcher.state(), { ...selection, hardware: "h100" });
});

test("DeepSeek-V3.1 variants expose only validated TP6 CPU baselines and preserve existing recipes", () => {
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
  assert.equal(entries.length, 208);
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
  const terminus = builder("deepseek-v31", { ...selection, modelname: "v31terminus" });
  assert.match(terminus.generateCommand(),
    /--model deepseek-ai\/DeepSeek-V3\.1-Terminus(?:\s|$)/);
  assert.equal(terminus.generateCommand(),
    command.replace("--model deepseek-ai/DeepSeek-V3.1 \\\n",
      "--model deepseek-ai/DeepSeek-V3.1-Terminus \\\n"));
  for (const key of ["strategy", "reasoningParser", "toolcall"]) {
    assert.equal(terminus.options[key].condition({ ...selection, modelname: "v31terminus" }), false);
  }
  assert.equal(terminus.options.modelname.items.find((item) => item.id === "v31terminus")
    .disabledWhen, undefined);
  const switcher = builder("deepseek-v31", { ...selection, modelname: "v31terminusint8" });
  switcher.handleRadioChange("modelname", "v31");
  const baseline = {
    ...selection, strategy: ["tp"], reasoningParser: "disabled", toolcall: "disabled",
  };
  assert.deepEqual(switcher.state(), baseline);
  switcher.handleRadioChange("modelname", "v31terminus");
  const terminusBaseline = { ...baseline, modelname: "v31terminus" };
  assert.deepEqual(switcher.state(), terminusBaseline);
  switcher.handleRadioChange("reasoningParser", "enabled");
  switcher.handleRadioChange("toolcall", "enabled");
  switcher.handleCheckboxChange("strategy", "dp", true);
  assert.deepEqual(switcher.state(), terminusBaseline);
  switcher.handleRadioChange("modelname", "v31");
  assert.deepEqual(switcher.state(), baseline);
  const hardwareSwitch = builder("deepseek-v31", { ...selection, hardware: "h200", modelname: "v31terminus" });
  hardwareSwitch.handleRadioChange("hardware", "xeon");
  assert.equal(hardwareSwitch.state().modelname, "v31terminusint8");
  hardwareSwitch.handleRadioChange("modelname", "v31terminus");
  assert.deepEqual(hardwareSwitch.state(), terminusBaseline);
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
