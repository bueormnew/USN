"""USN architecture submodules.

This package contains the 7 individual submodules that compose a USN block,
each implementing one step of the state-update-and-readout pipeline:

1. InputProjection - Linear transformation of input embeddings
2. TemporalMixing - Local temporal blending with one-step lookback
3. ExponentialGating - Bounded decay factor computation
4. SelectiveWriting - Content-dependent write gate
5. StateUpdate - Unified state transition (semantic + relational)
6. StateReadout - State extraction with confidence gating
7. ChannelMixing - Inter-channel feedforward with residual
"""

from usn.modules.channel_mixing import ChannelMixing
from usn.modules.exponential_gating import ExponentialGating
from usn.modules.input_projection import InputProjection
from usn.modules.selective_writing import SelectiveWriting
from usn.modules.state_readout import StateReadout
from usn.modules.state_update import StateUpdate
from usn.modules.temporal_mixing import TemporalMixing

__all__ = [
    "InputProjection",
    "TemporalMixing",
    "ExponentialGating",
    "SelectiveWriting",
    "StateUpdate",
    "StateReadout",
    "ChannelMixing",
]
