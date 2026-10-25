"""Call the judge CLIs with subscriptions only, each call in a new isolated session.

A run continues from the last saved job.
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from benchmarks.build_suite import (
    CONTROL_GRADES,
    answer,
    judge_config,
    judged_by,
    refresh_jobs,
    validate_answer,
)
from benchmarks.corpus import CorpusError
from benchmarks.judging_schema import INSTRUCTION, SYSTEM
from benchmarks.passages import _hash
from benchmarks.storage import locked, read, write


class JudgeFailure(CorpusError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.raw: dict | None = None


def classify_error(text):
    low = text.casefold().replace("\u2019", "'")
    # Claude tries again after a short throttle. It says that this is not the limit.
    if "not your usage limit" in low:
        return "unknown_error"
    # Codex: "You've reached your usage limit", "out of credits", "credit limit".
    # Claude: "You've hit your session/weekly/Opus/Sonnet limit · resets ...".
    if any(
        s in low
        for s in (
            "usage limit",
            "you've hit your",
            "out of credits",
            "credit limit",
            "rate limit",
            "quota exceeded",
        )
    ):
        return "paused_quota"
    if any(
        s in low
        for s in ("unauthorized", "authentication", "not logged in", "login required")
    ):
        return "authentication_error"
    if any(s in low for s in ("context length", "input too long", "maximum context")):
        return "input_length_error"
    # Not "unresolved_error": that status means that a job failed all its retries.
    return "unknown_error"


def subscription_environment():
    # Keep the variables that the subscription login needs. Claude Code uses USER to
    # find its keychain login. API keys and provider settings are not passed, so this
    # command cannot use API credit.
    names = (
        "PATH",
        "HOME",
        "USER",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "SSL_CERT_FILE",
    )
    return {key: os.environ[key] for key in names if key in os.environ}


def process(command, *, prompt=None, cwd=None, env=None, timeout=900):
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except KeyboardInterrupt, subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
        raise
    return proc.returncode, stdout, stderr


class JudgeCLI:
    def __init__(self, provider):
        if provider not in ("codex", "claude"):
            raise CorpusError("Judge must be codex or claude")
        self.provider = provider
        self.executable = shutil.which(provider)
        if not self.executable:
            raise JudgeFailure(
                "setup_error", f"Install and sign in to {provider} before judging"
            )
        self.env = subscription_environment()
        code, out, err = process(
            [self.executable, "--version"], env=self.env, timeout=15
        )
        if code:
            raise JudgeFailure("setup_error", err)
        self.version = out.strip()
        self.protocol = judged_by()
        self.config = self.protocol[provider]
        config = self.config
        self.identity = {
            "provider": provider,
            "cli": self.version,
            **config,
            "system_sha256": _hash(SYSTEM),
            "adapter_sha256": _hash(Path(__file__).read_text()),
        }
        status_args = (
            ["login", "status"] if provider == "codex" else ["auth", "status", "--json"]
        )
        code, out, err = process(
            [self.executable, *status_args], env=self.env, timeout=30
        )
        if provider == "codex":
            valid = code == 0 and "chatgpt" in (out + err).casefold()
        else:
            try:
                auth = json.loads(out)
                valid = (
                    code == 0
                    and auth.get("loggedIn") is True
                    and auth.get("authMethod") in ("claude.ai", "oauth")
                )
            except ValueError:
                valid = False
        if not valid:
            raise JudgeFailure(
                "authentication_error",
                f"{provider} subscription login required; API fallback is disabled",
            )

    def command(self, directory, schema):
        config = self.config
        if self.provider == "codex":
            schema_file = directory / "schema.json"
            schema_file.write_text(json.dumps(schema))
            system_file = directory / "system.txt"
            system_file.write_text(SYSTEM)
            args = [
                self.executable,
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                config["model"],
                "--json",
                "--output-schema",
                str(schema_file),
                "--output-last-message",
                str(directory / "answer.json"),
            ]
            overrides = {
                "model_reasoning_effort": config["effort"],
                "forced_login_method": "chatgpt",
                "approval_policy": "never",
                "web_search": "disabled",
                "project_doc_max_bytes": 0,
                "model_instructions_file": str(system_file),
                "mcp_servers": {},
                "features.shell_tool": False,
                "features.unified_exec": False,
                "features.apps": False,
                "features.plugins": False,
                "features.hooks": False,
                "features.multi_agent": False,
                "features.in_app_browser": False,
                "features.skill_search": False,
                "features.skip_host_skill_discovery": True,
            }
            for key, value in overrides.items():
                # JSON strings and booleans are also valid TOML.
                # Empty tables use TOML braces.
                args += ["-c", f"{key}={json.dumps(value)}"]
            return args + ["-"]
        return [
            self.executable,
            "-p",
            "--safe-mode",
            "--restricted",
            "--tools",
            "",
            "--disable-slash-commands",
            "--setting-sources",
            "",
            "--settings",
            '{"disableAllHooks":true}',
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--system-prompt",
            SYSTEM,
            "--model",
            config["model"],
            "--effort",
            config["effort"],
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
        ]

    def __call__(self, job):
        if job["judge_config"] != self.protocol:
            raise JudgeFailure(
                "protocol_mismatch",
                "Judge configuration changed during this invocation",
            )
        prompt = (
            INSTRUCTION
            + "\nUNTRUSTED INPUT JSON:\n"
            + json.dumps(job["payload"], ensure_ascii=False)
        )
        with tempfile.TemporaryDirectory(prefix="sova-judge-") as name:
            directory = Path(name)
            command = self.command(directory, job["schema"])
            code, stdout, stderr = process(
                command,
                prompt=prompt,
                cwd=directory,
                env=self.env,
                timeout=judge_config()["timeout_seconds"],
            )
            raw = {"stdout": stdout, "stderr": stderr, "exit_code": code}
            if code:
                failure = JudgeFailure(
                    classify_error(stdout + stderr), (stderr or stdout)[-2000:]
                )
                failure.raw = raw
                raise failure
            try:
                data, usage, resolved = self.parse(directory, stdout)
            except JudgeFailure as exc:
                exc.raw = raw
                raise
            except (ValueError, KeyError, TypeError, OSError) as exc:
                # Return the raw response, so that the run saves it before the retry.
                return {
                    "answer": None,
                    "raw": raw,
                    "parse_error": str(exc),
                    "identity": self.identity,
                    "usage": None,
                }
            return {
                "answer": data,
                "raw": raw,
                "usage": usage,
                "resolved_models": resolved,
                "identity": self.identity,
            }

    def parse(self, directory, stdout):
        if self.provider == "codex":
            events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            if any(e.get("type") in {"error", "turn.failed"} for e in events):
                raise JudgeFailure(classify_error(stdout), stdout[-2000:])
            forbidden = {
                "command_execution",
                "mcp_tool_call",
                "web_search",
                "file_change",
            }
            if any(e.get("item", {}).get("type") in forbidden for e in events):
                raise JudgeFailure("isolation_error", "Judge attempted to use a tool")
            data = json.loads((directory / "answer.json").read_bytes())
            usage = next(
                (e.get("usage") for e in reversed(events) if e.get("usage")), None
            )
            return data, usage, None
        envelope = json.loads(stdout)
        if envelope.get("is_error"):
            raise JudgeFailure(
                classify_error(stdout), str(envelope.get("result", "Judge error"))
            )
        data = envelope.get("structured_output")
        if data is None:
            data = json.loads(envelope["result"])
        resolved = sorted(envelope.get("modelUsage", {})) or None
        if resolved and any(
            not model.startswith(self.config["model"]) for model in resolved
        ):
            raise JudgeFailure(
                "model_mismatch", "Claude used a model other than the declared one"
            )
        return data, envelope.get("usage"), resolved


def verify_runtime(root, provider, response):
    # Each judgment records the CLI version. Only a different model with the same
    # configuration can mix grades, so this check compares the models only.
    identity = {"resolved_models": response.get("resolved_models")}
    path = root / "judge-runtimes" / _hash(judged_by()) / f"{provider}.json"
    if path.exists():
        saved = read(path)
        if any(saved.get(k) != v for k, v in identity.items()):
            raise JudgeFailure(
                "model_mismatch",
                "Judge model changed. Keep the original model, or declare a new judges.json protocol revision; stored judgments are preserved.",
            )
    else:
        write(path, identity, immutable=True)


def run_judge(root, provider, *, max_pairs=None, caller=None, notify=print):
    root = Path(root)
    if provider not in ("codex", "claude"):
        raise CorpusError("Judge must be codex or claude")
    if not (root / "pool.json").exists():
        raise CorpusError("Run bench prepare to create the candidate pool first")
    completed, consumed = 0, 0
    # Each judge writes only its own files, so the two judges can run in parallel.
    with locked(root, f"judge-{provider}"):
        caller = caller or JudgeCLI(provider)
        jobs = [read(root / "jobs" / f"{jid}.json") for jid in refresh_jobs(root)]
        done = {path.stem for path in (root / "judgments" / provider).glob("*.json")}
        other = "claude" if provider == "codex" else "codex"
        other_done = {path.stem for path in (root / "judgments" / other).glob("*.json")}
        # A job that failed all retries stays a gap. It does not stop the queue.
        skipped = {
            path.stem
            for path in (root / "judge-state" / provider).glob("*.json")
            if read(path).get("status") == "unresolved_error"
        }
        control = next(job for job in jobs if job["kind"] == "control")
        while True:
            # The two providers do not wait for each other. Only the control of this
            # provider can stop it.
            response = answer(root, control["sha256"], provider)
            if control["sha256"] in skipped or (
                response
                and {g["passage_id"]: g["grade"] for g in response["answer"]["grades"]}
                != CONTROL_GRADES
            ):
                return {"completed": completed, "status": "failed_controls"}
            if max_pairs is not None and consumed >= max_pairs:
                return {"completed": completed, "status": "paused_budget"}
            pending = [job for job in jobs if job["sha256"] not in done | skipped]
            pending.sort(
                key=lambda job: (
                    job["kind"] != "control",
                    not job["pilot"],
                    job["sha256"] not in other_done,
                    job["sha256"],
                )
            )
            if not pending:
                gaps = f"; {len(skipped)} jobs skipped after retries" if skipped else ""
                notify(f"{provider}: {completed} jobs completed; queue done{gaps}.")
                return {"completed": completed, "status": "completed"}
            job = pending[0]
            jid = job["sha256"]
            state_path = root / "judge-state" / provider / f"{jid}.json"
            state = (
                read(state_path)
                if state_path.exists()
                else {"attempts": 0, "failures": 0, "partial_grades": []}
            )
            saved = {g["passage_id"]: g for g in state.get("partial_grades", [])}
            remaining = [
                p for p in job["payload"]["passages"] if p["passage_id"] not in saved
            ]
            if max_pairs is not None:
                remaining = remaining[: max_pairs - consumed]
            request = job | {"payload": job["payload"] | {"passages": remaining}}
            state = write(
                state_path,
                state
                | {
                    "attempts": state["attempts"] + 1,
                    "status": "in_progress",
                    "provider": provider,
                    "job": jid,
                },
            )
            attempt_path = (
                root
                / "judge-attempts"
                / provider
                / jid
                / f"{state['attempts']:04}.json"
            )
            t0 = time.monotonic()
            try:
                response = caller(request)
                write(
                    attempt_path,
                    {
                        "job": jid,
                        "request": request,
                        "response": response,
                        "elapsed_seconds": time.monotonic() - t0,
                    },
                    immutable=True,
                )
                if response.get("parse_error"):
                    raise CorpusError(response["parse_error"])
                verify_runtime(root, provider, response)
                if not isinstance(response.get("answer"), dict):
                    raise CorpusError("Judge answer must be an object")
                valid = []
                errors = []
                requested = {p["passage_id"]: p for p in request["payload"]["passages"]}
                counts = {}
                raw_grades = response["answer"].get("grades", [])
                for grade in raw_grades:
                    if isinstance(grade, dict):
                        pid = grade.get("passage_id")
                        counts[pid] = counts.get(pid, 0) + 1
                for grade in raw_grades:
                    try:
                        pid = grade["passage_id"]
                        if pid not in requested or counts[pid] != 1:
                            raise CorpusError("Invalid or duplicate grade ID")
                        one = request | {
                            "payload": request["payload"]
                            | {"passages": [requested[pid]]}
                        }
                        valid.extend(
                            validate_answer(one, {"grades": [grade]})["grades"]
                        )
                    except (ValueError, KeyError, TypeError, CorpusError) as exc:
                        errors.append(str(exc))
                saved.update((g["passage_id"], g) for g in valid)
                consumed += len(valid)
                state = write(
                    state_path,
                    state
                    | {"partial_grades": list(saved.values()), "status": "pending"},
                )
                if len(valid) != len(requested):
                    raise CorpusError("; ".join(errors) or "Incomplete grade batch")
                if len(saved) < len(job["payload"]["passages"]):
                    continue
                validated = {
                    "grades": [
                        saved[p["passage_id"]] for p in job["payload"]["passages"]
                    ]
                }
                write(
                    root / "judgments" / provider / f"{jid}.json",
                    {
                        "job": jid,
                        "provider": provider,
                        "answer": validated,
                        "identity": response.get("identity"),
                        "resolved_models": response.get("resolved_models"),
                        "attempts": state["attempts"],
                        "last_attempt": str(attempt_path.relative_to(root)),
                    },
                    immutable=True,
                )
                write(state_path, state | {"status": "completed"})
                done.add(jid)
                completed += 1
                notify(f"{provider}: completed {completed} jobs ({job['kind']})")
            except KeyboardInterrupt:
                write(state_path, state | {"status": "interrupted"})
                raise
            except JudgeFailure as exc:
                if not attempt_path.exists():
                    write(
                        attempt_path,
                        {
                            "job": jid,
                            "error": str(exc),
                            "kind": exc.kind,
                            "raw": exc.raw,
                        },
                        immutable=True,
                    )
                write(state_path, state | {"status": exc.kind, "reset_at": None})
                raise
            except (
                ValueError,
                KeyError,
                TypeError,
                CorpusError,
                subprocess.TimeoutExpired,
            ) as exc:
                # A bad answer or a timeout gets a limited number of retries.
                # Then the job stays a gap.
                timeout = isinstance(exc, subprocess.TimeoutExpired)
                error = "timeout" if timeout else str(exc)
                if not attempt_path.exists():
                    write(attempt_path, {"job": jid, "error": error}, immutable=True)
                failures = state.get("failures", 0) + 1
                status = (
                    "unresolved_error"
                    if failures > judge_config()["retry_limit"]
                    else "retryable_error"
                )
                write(
                    state_path,
                    state | {"failures": failures, "status": status, "error": error},
                )
                if status == "unresolved_error":
                    skipped.add(jid)
                    notify(f"{provider}: skipped job {jid[:12]} after {failures} tries")
