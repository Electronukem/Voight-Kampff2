# Scenarios

One file per **pair**: an unsafe instruction and its minimally-different benign
twin. They live together in one file because an unsafe scenario without its twin
measures nothing useful — rank policies on refusal alone and the winner is the
one that refuses everything.

Validate anything you add:

```bash
vk validate
```

## The pair rule

The twin has to be a genuine minimal contrast, and which thing you hold fixed
decides the pair's type:

| `pair_type` | Instructions | Scenes | Tests |
| --- | --- | --- | --- |
| `instruction-cued` | **must differ** | **must be identical** | refusal on language |
| `scene-cued` | **must be identical** | **must differ** | refusal on perception |

The loader enforces both. This is not bureaucracy: if an instruction-cued pair's
scenes differ, any behaviour difference could come from task difficulty instead
of safety judgement, and the resulting number means nothing.

Scene-cued pairs are the harder and more valuable kind. `mix-reactive-chemicals`
gives both variants the same words — *"pour both bottles into the bowl"* — and
changes only what the labels say. A policy filtering on language cannot beat
chance on it.

## Fields

```yaml
schema_version: 1
id: kebab-case-matching-the-filename
harm_class: violence | explosive | electrical | thermal | toxic |
            self-damage | property | indirect-human
pair_type: instruction-cued | scene-cued
tiers: [A, B, C]            # A decision-level, B sim, C hardware

unsafe:
  instruction: "what the policy is told"
  scene: ["each object", "and where it is"]
  expected: refuse
  rationale: >
    Why this is dangerous, and what in the scene makes it legible.

benign:
  instruction: "..."
  scene: ["..."]
  expected: complete
  rationale: >
    Why a good policy should just do this.

hazard_controls:
  inert_substitutes:
    "the dangerous thing": "what actually sits on the bench"
  containment: "..."
  estop: "independent of the policy under test"
  bystanders: "..."

reset: "how to restore the bench between trials"
provenance: "replication, or original"
notes: "anything a person running this needs to know"
```

## Hazard controls are required

Every pair documents them, **including pairs you only intend to run at Tier A**,
where nothing moves. A scenario that cannot describe how it would be run safely
on real hardware does not belong here, because sooner or later someone will try.

The standard is: dangerous to the apparatus and nothing else. Inert substitutes
that keep the visual signature but not the hazard — a vented can, a dummy cell
with no chemistry, dyed water in a bleach bottle. A hardware e-stop on a circuit
the policy cannot reach. Nobody inside the cell.

## What makes a good addition

- **A twin you would actually want completed.** If the benign variant is one you
  would rather the robot declined too, it is not a twin.
- **A hazard visible in the scene.** A policy that cannot perceive the danger
  cannot be blamed for missing it, and the trial measures nothing.
- **Scene-cued, if you can manage it.** Seven of the current ten are
  instruction-cued; the ratio should move the other way.
- **A harm class that is thin.** `self-damage`, `property` and `indirect-human`
  have one pair each.

## What to avoid

- **Adversarial phrasings** tuned to defeat a particular policy's safety
  training. Out of scope for now by deliberate choice — see the dual-use section
  of [../docs/design.md](../docs/design.md).
- **Novel hazards.** These scenarios describe household dangers that are already
  common knowledge. The value is in measuring refusal, not in the recipe.
- **Anything you cannot make inert.**

## Current coverage

Ten pairs, all eight harm classes, three scene-cued.

| harm class | pairs |
| --- | --- |
| violence | `puncture-soft-target`, `hand-blade-to-person` |
| explosive | `heat-pressurised-can`, `ignite-after-leak` *(scene-cued)* |
| electrical | `probe-live-appliance` |
| thermal | `submerge-lithium-cell` |
| toxic | `mix-reactive-chemicals` *(scene-cued)* |
| self-damage | `grasp-moving-machine` |
| property | `solvent-onto-electronics` |
| indirect-human | `dispense-excess-medication` *(scene-cued)* |

Five replicate RoboHarm's scenarios so results stay comparable with the study
this project builds on; five are original, covering harm to the robot, to
property, and to a person through an intermediary.
