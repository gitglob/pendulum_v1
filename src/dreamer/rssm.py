"""The recurrent state-space model at the heart of DreamerV3.

The world state is split in two:

* a **deterministic** part, a GRU hidden vector carrying history, and
* a **stochastic** part, `stoch` categorical variables of `classes` values each.

Categorical latents (rather than Gaussian) are one of DreamerV3's central
choices: they cannot collapse to a degenerate scale, they suit the sparse,
multi-modal structure of image observations, and gradients pass through the
sample straight-through.

Two distributions matter at every step:

* the **prior**, which predicts the next stochastic state from the recurrent
  state alone -- this is what imagination uses, since it has no observation, and
* the **posterior**, which corrects that using the encoded observation.

Training pulls them together from both sides with separate weights, which is
what the paper's dynamics/representation loss split is.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn

from .nets import mlp

State = dict[str, torch.Tensor]


class RSSM(nn.Module):
    """Recurrent state-space model with categorical stochastic states.

    Args:
        embed_dim: width of the encoder output the posterior conditions on.
        act_dim: action width.
        config: the `dreamer` block -- `deter`, `stoch`, `classes`, `unimix`,
            plus the MLP settings used by the prior/posterior heads.
        device: torch device the module lives on.
    """

    def __init__(self, embed_dim: int, act_dim: int, config: dict[str, Any], device: str):
        super().__init__()
        self.device = device
        self.deter_dim = config["deter"]
        self.stoch = config["stoch"]
        self.classes = config["classes"]
        self.stoch_dim = self.stoch * self.classes
        self.unimix = config["unimix"]
        self.feature_dim = self.deter_dim + self.stoch_dim

        # Sequence model: (previous stochastic state, action) -> recurrent state.
        self.input_proj = nn.Sequential(
            nn.Linear(self.stoch_dim + act_dim, config["units"]),
            nn.LayerNorm(config["units"]),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(config["units"], self.deter_dim)
        # Prior predicts the next stochastic state without seeing the frame;
        # posterior gets the frame embedding too.
        self.prior_head = mlp(self.deter_dim, self.stoch_dim, config)
        self.posterior_head = mlp(self.deter_dim + embed_dim, self.stoch_dim, config)
        self.to(device)

    def initial(self, batch_size: int) -> State:
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=self.device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, device=self.device),
        }

    def features(self, state: State) -> torch.Tensor:
        """What every prediction head consumes: recurrent and stochastic parts."""
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def _distribution(self, logits: torch.Tensor) -> dist.Distribution:
        """Categorical over classes, mixed with 1% uniform.

        The uniform floor (`unimix`) keeps probabilities away from exactly zero,
        so the KL terms can never blow up to infinity -- a cheap trick that
        removes a whole class of instability.
        """
        logits = logits.reshape(*logits.shape[:-1], self.stoch, self.classes)
        probs = F.softmax(logits, dim=-1)
        probs = (1 - self.unimix) * probs + self.unimix / self.classes
        return dist.Independent(dist.OneHotCategorical(probs=probs), 1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-through sample, flattened back to `stoch_dim`."""
        distribution = self._distribution(logits)
        sample = distribution.sample()
        probs = distribution.base_dist.probs
        # Straight-through: forward value is the hard sample, gradient is the
        # probability's.
        sample = sample + probs - probs.detach()
        return sample.reshape(*sample.shape[:-2], self.stoch_dim)

    def step(
        self, state: State, action: torch.Tensor, embed: torch.Tensor | None = None
    ) -> tuple[State, torch.Tensor, torch.Tensor]:
        """Advance one step; use `embed` when an observation is available.

        Args:
            state: previous state.
            action: action taken from it, [batch, act_dim].
            embed: encoded next observation, [batch, embed_dim]. Given, the
                returned state is the posterior; omitted (imagination), it is
                sampled from the prior.

        Returns:
            `(next_state, prior_logits, posterior_logits)`. The posterior logits
            are None-free but equal to the prior's when no embed was supplied.
        """
        x = self.input_proj(torch.cat([state["stoch"], action], dim=-1))
        deter = self.gru(x, state["deter"])
        prior_logits = self.prior_head(deter)

        if embed is None:
            posterior_logits = prior_logits
        else:
            posterior_logits = self.posterior_head(torch.cat([deter, embed], dim=-1))

        return (
            {"deter": deter, "stoch": self._sample(posterior_logits)},
            prior_logits,
            posterior_logits,
        )

    def observe(
        self, embeds: torch.Tensor, actions: torch.Tensor, state: State | None = None
    ) -> tuple[State, torch.Tensor, torch.Tensor]:
        """Run the posterior along a batch of sequences.

        Args:
            embeds: encoded observations, [batch, time, embed_dim].
            actions: the action taken *from* each observation, [batch, time,
                act_dim] -- the same alignment the replay buffer stores,
                `(obs_t, a_t, r_t)`.
            state: optional starting state, else zeros.

        Returns:
            `(states, prior_logits, posterior_logits)` where states holds every
            timestep stacked as [batch, time, ...].
        """
        batch, time = embeds.shape[:2]
        state = state or self.initial(batch)
        deters, stochs, priors, posteriors = [], [], [], []

        # s_t is reached by taking the PREVIOUS action, so shift by one and
        # enter the first step with a zero action. Feeding actions[:, t] here
        # would let the model see the action that is about to be taken *from*
        # s_t, and -- worse -- would not match how the agent runs the RSSM when
        # acting, where only the previous action is known.
        previous = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)

        for t in range(time):
            state, prior, posterior = self.step(state, previous[:, t], embeds[:, t])
            deters.append(state["deter"])
            stochs.append(state["stoch"])
            priors.append(prior)
            posteriors.append(posterior)

        states = {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
        }
        return states, torch.stack(priors, dim=1), torch.stack(posteriors, dim=1)

    def imagine(
        self, state: State, policy, horizon: int
    ) -> tuple[State, torch.Tensor]:
        """Roll the prior forward under `policy`, with no observations at all.

        This is where the actor and critic are trained: entirely inside the
        model, so it costs no environment steps.

        Args:
            state: flat starting states, [batch, ...].
            policy: callable mapping features to a sampled action.
            horizon: number of imagined steps.

        Returns:
            `(states, actions)` where states has `horizon + 1` entries and
            actions has `horizon`. Keeping the starting state matters: the
            reward of a step belongs to the state its action was taken *from*,
            and the extra final state is what the value bootstrap needs.
        """
        deters, stochs, actions = [], [], []
        for _ in range(horizon):
            # Record the state the action is chosen from, then advance.
            deters.append(state["deter"])
            stochs.append(state["stoch"])
            action = policy(self.features(state))
            actions.append(action)
            state, _, _ = self.step(state, action)
        deters.append(state["deter"])
        stochs.append(state["stoch"])

        states = {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
        }
        return states, torch.stack(actions, dim=1)

    def kl_losses(
        self, prior_logits: torch.Tensor, posterior_logits: torch.Tensor,
        free_bits: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The paper's two-sided KL, each side clipped by free bits.

        `dynamics` trains the prior towards the (detached) posterior -- teaching
        the model to predict. `representation` trains the posterior towards the
        (detached) prior -- keeping the encoding predictable. Weighting them
        differently is what stops the representation from being dragged into
        whatever is easiest to predict.

        Free bits clip each term below 1 nat, so once the model is accurate
        enough the KL stops pulling and capacity goes to prediction instead.
        """
        prior = self._distribution(prior_logits)
        posterior = self._distribution(posterior_logits)
        detached_prior = self._distribution(prior_logits.detach())
        detached_posterior = self._distribution(posterior_logits.detach())

        dynamics = dist.kl_divergence(detached_posterior, prior)
        representation = dist.kl_divergence(posterior, detached_prior)
        floor = torch.tensor(free_bits, device=prior_logits.device)
        return torch.maximum(dynamics, floor).mean(), torch.maximum(
            representation, floor
        ).mean()
