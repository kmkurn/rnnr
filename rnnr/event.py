# Copyright 2019 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum, auto


class Event(Enum):
    """An enumeration of events.

    Attributes:
        STARTED: Emitted once at the start of a run.
        EPOCH_STARTED: Emitted at the start of every epoch.
        BATCH: Emitted on every batch.
        EPOCH_FINISHED: Emitted at the end of every epoch.
        FINISHED: Emitted once at the end of a run.
    """
    STARTED = auto()
    EPOCH_STARTED = auto()
    BATCH = auto()
    EPOCH_FINISHED = auto()
    FINISHED = auto()

    # Events for EpochTimer attachment
    _ETIMER_STARTED = auto()
    _ETIMER_FINISHED = auto()

    # Events for ProgressBar attachment
    _PBAR_CREATED = auto()
    _PBAR_UPDATED = auto()
    _PBAR_CLOSED = auto()

    # Events for LambdaReducer attachment
    _REDUCER_RESET = auto()
    _REDUCER_UPDATED = auto()
    _REDUCER_COMPUTED = auto()
