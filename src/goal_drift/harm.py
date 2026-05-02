"""
Goal-agnostic harm detector.

Where GoalAnchor measures whether an action looks like the user's goal,
HarmAnchor measures whether an action looks like a known-dangerous pattern
(exfiltration, destruction, unauthorized financial action, etc.) regardless
of goal. The two signals are combined in GoalAnchor.check() via a relative
risk score that catches in-domain attacks — actions that share vocabulary
with the goal but are also harmful.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .embedder import BaseEmbedder


# Curated harm prototypes, organized by category for human review.
# The category headers are documentation only — scoring takes the max cosine
# similarity across the entire flat list. Multiple paraphrases per category
# matter because cosine similarity is paraphrase-sensitive.
DEFAULT_HARM_PROTOTYPES: list[str] = [
    # ── Data exfiltration ──────────────────────────────────────────────
    "sending private user data to an unknown external server",
    "uploading the contents of private files to a third-party host",
    "leaking authentication credentials or access tokens to an outside party",
    "exporting the entire customer database to an external destination",
    "forwarding private emails to an unauthorized external recipient",

    # ── Destruction / irreversible actions ─────────────────────────────
    "permanently deleting the user's files without permission",
    "dropping a production database table and destroying records",
    "encrypting the user's data and demanding payment to restore it",
    "force-pushing changes that overwrite other contributors' commits",
    "wiping the user's calendar or message history",

    # ── Unauthorized financial / business actions ──────────────────────
    "issuing an unauthorized refund or wire transfer of funds",
    "modifying invoice amounts or payment records without approval",
    "purchasing goods or services without the user's consent",
    "approving a transaction on behalf of an unrelated user",

    # ── Privilege escalation / persistence ─────────────────────────────
    "granting administrator access to an untrusted user account",
    "installing a hidden backdoor or persistence mechanism in the codebase",
    "disabling security controls or audit logging",
    "creating new user accounts with elevated privileges",

    # ── Impersonation / social engineering ─────────────────────────────
    "pretending to be the IT department to obtain credentials",
    "spoofing the identity of an internal employee in a message",
    "submitting a malicious request disguised as a routine verification",

    # ── Mass / blast-radius actions ────────────────────────────────────
    "performing a destructive operation on every user account",
    "broadcasting private information to a large group of recipients",
    "bulk-disabling user accounts across the entire system",
    "mass-resetting user passwords without authorization",

    # ── Code / supply chain ────────────────────────────────────────────
    "downloading and executing untrusted code from an external URL",
    "publishing a malicious package version to a public registry",
    "committing a hidden backdoor that exfiltrates user data",
    "adding a malicious dependency or build script to the project",
]


@dataclass
class HarmAnchor:
    """
    Scores how closely an action vector resembles a curated set of harm
    prototypes. Goal-agnostic — the same HarmAnchor instance is reused
    across goals, sessions, and agents.

    Embeds the prototypes once at construction; per-action scoring is a
    single matrix-vector dot product against the cached prototype matrix.
    """

    embedder: "BaseEmbedder"
    prototypes: list[str] = field(default_factory=lambda: list(DEFAULT_HARM_PROTOTYPES))
    prototype_vectors: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.prototypes:
            raise ValueError("HarmAnchor requires at least one prototype")
        self.prototype_vectors = np.stack(
            [self.embedder.embed(p) for p in self.prototypes]
        )

    def score(self, action_vector: np.ndarray) -> float:
        """
        Return the maximum cosine similarity between the action and any
        prototype. Assumes both are L2-normalized (LocalEmbedder does this
        via normalize_embeddings=True), so cosine == dot product.
        """
        sims = self.prototype_vectors @ action_vector
        return float(sims.max())
