# Overnight Related Work Notes - Selective Risk Framing

Date: 2026-06-25

Branch/worktree:

- Branch: `codex/overnight-autonomous-research-20260624`
- Worktree: `/Users/chenmohan/gits/darjeeling-overnight-research-20260624`

## Research Question

Can Darjeeling's L1/L2 acceptance gates be treated as a selective prediction system: maximize accepted coverage only under a bounded accepted-risk constraint?

This matters because the 2026-06-24 L1 and L2 CLINC150 experiments both produced high visible coverage pressure with insufficient private/locked safety:

- L1: visible validation accepted precision reached 100% at useful coverage, but the selected candidate later exposed 24 locked-test wrong accepts.
- L2: an AutoResearch candidate reached perfect visible inner validation and train-audit acceptance, but did not improve private selection holdout accuracy and was not adopted.

## Sources Read

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks": https://arxiv.org/abs/1705.08500
- Geifman and El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option": https://arxiv.org/abs/1901.09192
- Hendrickx et al., "Machine Learning with a Reject Option: A survey": https://arxiv.org/abs/2107.11277
- Xin et al., "The Art of Abstention: Selective Prediction and Error Regularization for NLP": https://aclanthology.org/2021.acl-long.84/

## Mapping To Darjeeling

The selective-classification framing maps cleanly onto Darjeeling without adding a new framework:

- Accepted precision is the risk constraint.
- Accepted coverage is the objective after the risk constraint is satisfied.
- L4 fallback is the reject/abstain route.
- L1 and L2 candidate selection should reject candidates before private or locked exposure when visible evidence shows any accepted-error signal that is likely to transfer.
- Precision/coverage plots are the standard evidence surface for comparing candidates, but candidate adoption still needs hard gates because operating curves alone can hide small accepted-error counts.

## Design Implications Used In This Sprint

1. Treat "zero visible wrong accepts on a narrow validation slice" as insufficient when a larger train-dev slice already shows wrong accepts.
2. Promote count-based accepted-error gates alongside precision ratios. A small precision drop can still correspond to many locked wrong accepts.
3. Keep scratch search outputs separate from active target configs. Search should produce auditable candidates first, then apply them only through an explicit writeback path.
4. Keep the changes target-local. The CLINC150/NLU fields and diagnostics stay in `darjeeling.targets.nlu`; core remains target independent.

