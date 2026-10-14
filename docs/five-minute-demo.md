# Five-minute demo

This walkthrough uses only the bundled synthetic inventory. It needs no cloud credentials, no
provider account, and no LLM.

## 1. Start the demo (one minute)

```bash
git clone https://github.com/redbarron23/obs-agent.git
cd obs-agent
uv sync
OBS_AGENT_DEMO=1 uv run streamlit run app.py
```

Open the local URL printed by Streamlit, **Label review** opens by default. Demo mode
creates a disposable decision database for this browser session; the session stops using it when it ends; the operating system eventually cleans up the file.

## 2. Inspect a suggestion (one minute)

Keep the default **team** and **untagged or suspect** filters. In a row, examine:

- **Suggested** and **confidence** for the proposed tag value.
- **Why** for the source and supporting evidence.
- **Alternatives** for the next-most-likely values.
- **Action** to see whether it is eligible for expedited review (at least 90% confidence) or
  needs an explicit decision.

Confidence is evidence, not proof. Do not accept a suggestion merely because its percentage is
high; check that its ownership, parent, and naming evidence make sense.

## 3. Make a review decision (one minute)

Keep the demo reviewer ID or enter your own. Choose a row, set **Decision** to **accept**, then select **Save decisions**. To demonstrate the
other outcomes, choose **reject** to retain the current value, or **override** and select a
permitted value. The decision is retained with the original recommendation, confidence, and
reason.

Suggestions below 90%, and existing tags the model disputes, are deferred for this explicit
review. Higher-confidence suggestions can be bulk-accepted for a faster review, but are never
sent directly to a cloud provider.

## 4. Export an approval artifact (one minute)

Scroll to **Export** and download **Reviewed change plan**. It is enabled after at least one
accepted or overridden decision. Inspect the JSON: it contains the before/after label change and
the associated approval evidence.

The convenience CSV, shell, and Terraform exports also start reviewed-only. You must explicitly
tick the option to include unreviewed high-confidence suggestions. The reviewed change plan never
includes those suggestions.

## 5. See the model’s scorecard (one minute)

In another terminal, run:

```bash
uv run labeling.py --hardness both
```

Compare the clean and harsh scenarios. The command reports coverage, accuracy, macro-F1,
auto-apply precision, calibration error, and how performance changes when ownership and naming
signals become less reliable.

## What the demo does not do

It does not read from Azure or GCP, apply tags, or call an LLM. It is a safe way to see the
inference, human review, and approval handoff that precede a team’s normal delivery process.

For the policy behind the 90% threshold, see [label inference](label-inference.md). For the API
version of this workflow, see the [API reference](api.md).
