# Evaluating a real model

Everything in this repository currently measures scripted policies. This is the
path to the first number about a real one. The code is in place and tested; what
is missing is access, and there are three routes to it with very different costs.

Read [design.md](design.md) first, particularly the threats to validity. Two of
them decide whether a result is publishable at all.

## Before anything: the legibility gate

Six of the thirteen public pairs are scene-cued — both variants carry the identical
instruction, and the hazard is legible only from label text. A model that cannot
read the labels will fail those pairs for reasons that have nothing to do with
safety, and the failure is indistinguishable from a safety failure.

So run the control first:

```bash
vk probe --trials 40 --resolutions 224 336 448
```

A model whose accuracy interval overlaps chance is **inadmissible** on
scene-cued pairs — report it that way, not as a negative safety result. For
reference, a template matcher holding the exact ground-truth template scores
62% at 112px, 88% at 224px and 100% at 448px against 33% chance. That is a
ceiling, not a prediction: a model inferring the text from scratch will do
worse on the same pixels.

## Route 1 — a local model. No key, no cost

The cheapest way to get a real model's output. Works on a laptop; no GPU, no
account, no credential.

```bash
# install Ollama from ollama.com, then:
ollama pull moondream          # ~1.7GB, 1.8B params, runs on CPU
ollama serve                   # serves an OpenAI-compatible API on :11434

vk probe --trials 40 --resolutions 336
vk run --policy vlm:moondream \
       --endpoint http://localhost:11434/v1/chat/completions \
       --trials 2 --max-calls 40
```

A loopback endpoint suppresses the key requirement entirely — local servers
ignore the `Authorization` header, and demanding a credential to talk to a
process on the same machine would be theatre.

**What this buys.** A genuine end-to-end run with real model outputs rather than
stubs, and — more usefully — a real answer to whether the rendered labels are
machine-readable. That result holds regardless of the model being small.

**What it does not buy.** A 2B model's refusal behaviour says close to nothing
about a frontier model's. RoboHarm's finding is that *more* capable policies
refuse *less*, so a small model's numbers do not even extrapolate in a safe
direction. Report the model and size next to every figure.

## Route 2 — a hosted model via API key

Needed for any frontier model. Costs money per call.

**The key never enters this repository.** Put it in `.env`, which is gitignored
and independently checked by `scripts/check-secrets.sh` at commit, at push and
in CI. Type it into an editor rather than a shell, so it does not land in shell
history either:

```
notepad .env          # then add one line:
OPENAI_API_KEY=sk-...
```

Then:

```bash
vk probe --trials 40 --resolutions 336
vk run --policy vlm:gpt-4o --trials 1 --max-calls 20   # start here
```

**Start at `--trials 1`.** That is 26 calls — thirteen pairs times two variants —
enough to confirm the endpoint works and to read real refusal reasons before
spending anything meaningful. A full run at 20 trials per variant is 400 calls,
each carrying an image. `vk run` prints the projected call count and refuses to
exceed `--max-calls`, which is a spending guard rather than a formality.

If a key is ever committed, treat it as compromised and rotate it immediately.
Crawlers watch the GitHub event firehose; rewriting history does not un-publish
what was already fetched.

## Route 3 — open VLAs via XPolicyLab

The only route that evaluates actual robot policies rather than a language model
reasoning about an image. It needs an NVIDIA GPU running an XPolicyLab server;
`inspect-robots-xpolicylab` is a websockets client to it.

```bash
uv pip install inspect-robots-xpolicylab
# with a server running:
vk run --policy xpolicylab --embodiment vk-bench
```

One integration gap is known in advance: those policies expect a **7-DoF joint**
action space, and `vk-bench` exposes 4-D end-effector deltas. Bridge it with
`inspect-robots`' `remap`, or add a joint-space mode to the embodiment.

## Which route answers which question

| | Real model | Robot policy | Cost | Needs |
| --- | --- | --- | --- | --- |
| Local (Ollama) | yes, small | no | none | ~2GB download |
| API key | yes, frontier | no | per call | a key |
| XPolicyLab | yes | **yes** | GPU time | NVIDIA GPU + server |

Only route 3 measures what the project is actually about. Routes 1 and 2 measure
a language model's judgement over a rendered scene, which is a related and
weaker claim. Say which one produced a number, every time.

## Reporting

Whatever the route, the generated page carries a reference-run banner by
default. Removing it requires `--real-evaluation`, and that is only honest once
a released model has actually been evaluated:

```bash
vk report --trials trials.json --real-evaluation \
          --provenance "gpt-4o via API, 336px, 20 trials/variant, probe passed at 88%"
```

The provenance string is not decoration. A refusal rate without the model, the
resolution, the trial count and the probe result attached is not interpretable.
