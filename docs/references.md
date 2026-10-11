# References

Sources behind the design decisions in this project, grouped by what they informed.

## Label inference and evaluation

- [scikit-learn — Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) and [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — the inference and out-of-fold mislabel check
- [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) — the macro-F1 and precision metrics used in the README

## Cloud tagging semantics

- [Google Cloud — Labeling projects](https://cloud.google.com/resource-manager/docs/creating-managing-labels) and [Azure — Tag resources](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/tag-resources) — the tag semantics the export targets

## LLM tool use

- [Anthropic — Tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) and [OpenAI — Function calling](https://platform.openai.com/docs/guides/function-calling) — the two tool-call shapes the provider abstraction unifies

## Evaluation patterns for LLM systems

- [RAG 101: Demystifying Retrieval-Augmented Generation Pipelines](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/) — NVIDIA blog on evaluation patterns
- [Evaluating RAG: A Comprehensive Guide to Metrics and Methods](https://www.rungalileo.io/blog/evaluating-rag-a-comprehensive-guide-to-metrics-and-methods) — Galileo guide covering dual-attribution eval (source + fact checks)
- [LangSmith RAG Evaluation](https://docs.smith.langchain.com/faq/evaluation/eval_rag) — production-grade RAG eval patterns (correctness, faithfulness, relevance)

These last three are RAG-oriented, but the fact-presence checking strategy they describe is what
`evals.py` uses for tool-calling answers — see [docs/testing.md](testing.md).
