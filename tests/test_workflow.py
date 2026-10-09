"""Regression checks through CLI outputs and the dashboard page."""

import hashlib
from dataclasses import asdict
from contextlib import contextmanager
import json
import os
import socket
import time
import subprocess
import sys
import tempfile
import unittest
from urllib.request import urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def files(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="lingxi-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def cli(self, module, *arguments):
        return subprocess.run(
            [sys.executable, "-m", f"lingxi.{module}", *map(str, arguments)],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )

    def tiny_run(self, *options):
        import mlx.core as mx
        from mlx_lm.models.qwen3 import Model, ModelArgs
        from mlx_lm.utils import quantize_model, save_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        base = self.root / "base-8bit"
        base.mkdir()
        args = ModelArgs(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            rms_norm_eps=1e-6,
            vocab_size=64,
            max_position_embeddings=128,
            rope_theta=10000,
            tie_word_embeddings=True,
        )
        mx.random.seed(42)
        model = Model(args)
        model.set_dtype(mx.bfloat16)
        model, config = quantize_model(model, asdict(args), group_size=64, bits=8)
        save_model(base, model)
        (base / "config.json").write_text(json.dumps(config))
        vocab = {
            "[UNK]": 0,
            "user": 1,
            "assistant": 2,
            "hi": 3,
            "answer": 4,
            "yes": 5,
            "[EOS]": 6,
            "[PAD]": 7,
        }
        tokens = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokens.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokens,
            unk_token="[UNK]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            chat_template="{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {{ eos_token }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}",
        )
        tokenizer.save_pretrained(base)
        data = self.root / "data"
        data.mkdir()
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer yes"},
            ]
        }
        for split in ("train", "valid"):
            (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        (data / "stats.json").write_text(json.dumps({"total": 2, "by_source": {"fixture": 2}}))
        run = self.root / "first-run"
        result = self.cli(
            "train",
            "--model",
            base,
            "--data",
            data,
            "--out",
            run,
            "--num-layers",
            1,
            "--iters",
            1,
            "--save-every",
            1,
            "--steps-per-report",
            1,
            "--steps-per-eval",
            1,
            "--val-batches",
            1,
            "--grad-accumulation-steps",
            1,
            "--max-seq-length",
            32,
            "--no-grad-checkpoint",
            *options,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return base, data, run

    def test_training_records_the_requested_lora_configuration(self):
        _, _, run = self.tiny_run("--lora-rank", 4, "--lora-scale", 4)
        config = json.loads((run / "adapters/adapter_config.json").read_text())
        self.assertEqual(config["lora_parameters"], {"rank": 4, "scale": 4.0, "dropout": 0.0})

    def test_bundled_smoke_data_trains_and_saves_an_adapter(self):
        _, _, run = self.tiny_run("--data", ROOT / "docs/smoke")
        self.assertTrue((run / "adapters/adapters.safetensors").stat().st_size > 0)
        stats = json.loads((run / "data_stats.json").read_text())
        self.assertEqual((stats["train"], stats["valid"]), (12, 4))
        self.assertEqual(stats["license"], "Apache-2.0")

    def test_learning_rate_schedule_uses_optimizer_updates(self):
        _, _, run = self.tiny_run(
            "--learning-rate",
            1e-4,
            "--iters",
            8,
            "--grad-accumulation-steps",
            2,
            "--warmup-updates",
            1,
            "--cosine-schedule",
        )
        config = json.loads((run / "run_config.json").read_text())
        self.assertEqual(
            config["lr_schedule"],
            {
                "name": "cosine_decay",
                "arguments": [1e-4, 3, 1e-5],
                "warmup": 1,
                "warmup_init": 0.0,
            },
        )

    def test_training_rejects_consecutive_assistant_turns_before_loading_weights(self):
        data = self.root / "bad-data"
        data.mkdir()
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yes"},
                {"role": "assistant", "content": "another answer"},
            ]
        }
        for split in ("train", "valid"):
            (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        result = self.cli(
            "train",
            "--model",
            self.root / "missing-model",
            "--data",
            data,
            "--out",
            self.root / "bad-run",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("alternate user and assistant", result.stderr)

    def test_quantized_training_exports_real_four_bit_weights(self):
        from mlx_lm import load
        from lingxi.persona import NAME

        _, _, run = self.tiny_run()
        out = self.root / "export"
        result = self.cli("publish", "--run", run, "--out", out)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        exported, _, config = load(out / NAME, return_config=True)
        bits = {module.bits for _, module in exported.named_modules() if hasattr(module, "bits")}
        self.assertEqual(bits, {4})
        self.assertEqual(config["quantization"]["bits"], 4)
        float_config = json.loads((out / "bf16/config.json").read_text())
        self.assertNotIn("quantization", float_config)

    def test_model_card_reports_the_scope_of_recorded_validation_losses(self):
        run = self.root / "run"
        (run / "adapters").mkdir(parents=True)
        (run / "adapters/adapters.safetensors").touch()
        config = {
            "model": "models/Qwen3-1.7B",
            "data": "data/processed",
            "iters": 100,
            "lora_parameters": {"rank": 8, "scale": 20},
            "num_layers": 16,
            "batch_size": 1,
            "grad_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "max_seq_length": 1536,
        }
        (run / "run_config.json").write_text(json.dumps(config))
        (run / "data_stats.json").write_text('{"total":2,"by_source":{"fixture":2}}')
        rows = [
            {"type": "val", "iteration": 0, "val_loss": 1},
            {"type": "val", "iteration": 0, "val_loss": 5},
        ]
        (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
        before = files(run)
        out = self.root / "export"
        result = self.cli("publish", "--run", run, "--out", out, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Best recorded validation loss: 1", result.stdout)
        self.assertIn("sampler=make_sampler(temp=0.7, top_p=0.8, top_k=20)", result.stdout)
        self.assertIn("/no_think", result.stdout)
        self.assertIn("Learning rate schedule: `null`", result.stdout)
        self.assertEqual(files(run), before)
        self.assertFalse(out.exists())

    @contextmanager
    def dashboard(self, run):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = subprocess.Popen(
            [sys.executable, "-m", "lingxi.dashboard", "--run", str(run), "--port", str(port)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                try:
                    with urlopen(url + "/api/metrics", timeout=1) as response:
                        self.assertEqual(response.status, 200)
                    break
                except OSError:
                    if server.poll() is not None:
                        self.fail(server.communicate()[1])
                    time.sleep(0.05)
            else:
                self.fail("dashboard did not start")
            yield url
        finally:
            server.terminate()
            server.communicate(timeout=10)

    def test_dashboard_displays_every_retained_source_and_its_share(self):
        from playwright.sync_api import sync_playwright

        run = self.root / "run"
        run.mkdir()
        sources = {
            "chatml/dolphin": 20,
            "chatml/openhermes": 20,
            "chatml/airoboros": 20,
            "roleplay-zh": 20,
            "emotional-zh": 20,
        }
        (run / "data_stats.json").write_text(json.dumps({"total": 100, "by_source": sources}))
        with self.dashboard(run) as url:
            with urlopen(url + "/api/metrics") as response:
                self.assertEqual(json.load(response)["stats"]["by_source"], sources)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    page = browser.new_page()
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url)
                    page.wait_for_function(
                        "document.querySelector('#mixTotal').textContent.includes('100')"
                    )
                    labels = page.locator(".legend").inner_text().lower()
                    self.assertIn("openhermes", labels)
                    self.assertIn("airoboros", labels)
                    shares = page.locator("#mixBar > div").evaluate_all(
                        "nodes => nodes.map(node => parseFloat(node.style.width))"
                    )
                    self.assertEqual(shares, [20, 20, 20, 20, 20])
                    self.assertEqual(errors, [])
                finally:
                    browser.close()

    def test_dashboard_shows_pipeline_stages_and_optimizer_updates(self):
        from playwright.sync_api import sync_playwright

        run = self.root / "lr1e-06.model"
        run.mkdir()
        (run / "run_config.json").write_text(
            json.dumps({"iters": 128, "grad_accumulation_steps": 16})
        )
        (run / "status.json").write_text(json.dumps({"state": "COMPLETED"}))
        (run / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "type": "train",
                    "iteration": 100,
                    "train_loss": 2,
                    "iterations_per_second": 2,
                    "peak_memory": 5.5,
                }
            )
            + "\n"
        )
        pipeline = run.with_name(run.name + ".pipeline.json")
        pipeline.write_text(json.dumps({"state": "EVALUATING", "pid": os.getpid()}))
        with self.dashboard(run) as url, sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url)
                for state in (
                    "EVALUATING",
                    "EXPORTING",
                    "EVALUATING_4BIT",
                    "QUALITY_REVIEW_REQUIRED",
                    "COMPLETED",
                ):
                    pipeline.write_text(json.dumps({"state": state, "pid": os.getpid()}))
                    with urlopen(url + "/api/metrics") as response:
                        payload = json.load(response)
                    self.assertEqual(payload["status"], state)
                    self.assertEqual(payload["progress"]["optimizer_updates"], 8)
                    self.assertEqual(payload["progress"]["total_updates"], 8)
                    self.assertIsNone(payload["progress"]["eta_sec"])
                    page.wait_for_function(
                        "state => document.querySelector('#statusText').textContent.includes(state)",
                        arg=state,
                    )
                self.assertIn("8 / 8", page.locator("#updates").inner_text())
                page.get_by_role("button", name="双语 / 中 / EN").click()
                page.get_by_role("button", name="双语 / 中 / EN").click()
                self.assertIn("MLX peak", page.locator("#kpis").inner_text())
                self.assertNotIn("当前损失", page.locator("#kpis").inner_text())
                (run / "status.json").write_text(
                    json.dumps({"state": "TRAINING", "pid": os.getpid()})
                )
                pipeline.write_text(json.dumps({"state": "TRAINING", "pid": os.getpid()}))
                with urlopen(url + "/api/metrics") as response:
                    progress = json.load(response)["progress"]
                self.assertEqual(progress["optimizer_updates"], 6)
                self.assertEqual(progress["total_updates"], 8)
                self.assertEqual(progress["eta_sec"], 14)
                (run / "status.json").write_text(json.dumps({"state": "STOPPED"}))
                with urlopen(url + "/api/metrics") as response:
                    self.assertEqual(json.load(response)["status"], "STOPPED")
                pipeline.write_text(json.dumps({"state": "EVALUATING", "pid": -1}))
                with urlopen(url + "/api/metrics") as response:
                    self.assertEqual(json.load(response)["status"], "STOPPED")
            finally:
                browser.close()

    def test_dashboard_compares_raw_replies_and_flags_mismatched_settings(self):
        from playwright.sync_api import sync_playwright

        run = self.root / "comparison"
        run.mkdir()
        config = {
            "seed": 42,
            "max_tokens": 384,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "enable_thinking": False,
            "system_prompt": "same persona",
            "questions": [[11, "17 × 23?"]],
            "validation_sha256": "same split",
            "validation_indices": [0],
            "mask_prompt": True,
            "max_seq_length": 1536,
        }
        for name, text in (("base-assessment", "<b>391</b>"), ("final-assessment", "391")):
            assessment = run / name
            assessment.mkdir()
            (assessment / "config.json").write_text(json.dumps(config))
            (assessment / "result.json").write_text(json.dumps({"validation_loss": 2.5}))
            (assessment / "answers.jsonl").write_text(
                json.dumps(
                    {
                        "id": 11,
                        "prompt": "17 × 23?",
                        "text": text,
                    }
                )
                + '\n{"id":12'
            )
        with self.dashboard(run) as url, sync_playwright() as playwright:
            with urlopen(url + "/api/assessments") as response:
                payload = json.load(response)
            self.assertTrue(payload["matched"])
            self.assertEqual(payload["models"]["base"]["answers"][0]["text"], "<b>391</b>")
            self.assertEqual(payload["models"]["4bit"]["answers"], [])
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url)
                page.wait_for_function(
                    "document.querySelector('#reply-base').textContent.includes('<b>391</b>')"
                )
                self.assertEqual(page.locator("#reply-base b").count(), 0)
                self.assertEqual(page.locator("#reply-adapter").inner_text(), "391")
                config["max_tokens"] = 128
                (run / "final-assessment/config.json").write_text(json.dumps(config))
                with urlopen(url + "/api/assessments") as response:
                    self.assertFalse(json.load(response)["matched"])
                page.reload()
                page.wait_for_function(
                    "document.querySelector('#comparisonHint').textContent.includes('不一致')"
                )
            finally:
                browser.close()

    def test_resume_in_a_new_directory_preserves_the_original_run(self):
        base, data, original = self.tiny_run()
        before = files(original)
        resumed = self.root / "resumed-run"
        result = self.cli(
            "train",
            "--model",
            base,
            "--data",
            data,
            "--out",
            resumed,
            "--resume",
            original / "adapters/adapters.safetensors",
            "--num-layers",
            1,
            "--iters",
            1,
            "--save-every",
            1,
            "--steps-per-report",
            1,
            "--steps-per-eval",
            1,
            "--val-batches",
            1,
            "--grad-accumulation-steps",
            1,
            "--max-seq-length",
            32,
            "--no-grad-checkpoint",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(files(original), before)
        status = json.loads((resumed / "status.json").read_text())
        self.assertEqual(status["state"], "COMPLETED")
        old_weights = (original / "adapters/adapters.safetensors").read_bytes()
        new_weights = (resumed / "adapters/adapters.safetensors").read_bytes()
        self.assertNotEqual(old_weights, new_weights)

    def test_resume_rejects_an_existing_run_without_changing_its_files(self):
        run = self.root / "old-run"
        (run / "adapters").mkdir(parents=True)
        checkpoint = run / "adapters/0000600_adapters.safetensors"
        checkpoint.write_bytes(b"existing checkpoint")
        (run / "metrics.jsonl").write_text('{"type":"train","iteration":600}\n')
        (run / "run_config.json").write_text('{"iters":600}')
        (run / "data_stats.json").write_text('{"total":100,"seed":42}')
        data = self.root / "new-data"
        data.mkdir()
        (data / "stats.json").write_text('{"total":5,"seed":99}')
        model = self.root / "missing-model"
        model.mkdir()
        before = files(run)
        result = self.cli(
            "train", "--out", run, "--resume", checkpoint, "--data", data, "--model", model
        )
        self.assertEqual(files(run), before, result.stderr)
        self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
