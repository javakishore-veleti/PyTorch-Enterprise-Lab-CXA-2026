"""DependencyContainer — wires all classes together for FastAPI DI."""

from __future__ import annotations

from functools import lru_cache

from quantedge_services.core.checkpointing import CheckpointManager
from quantedge_services.core.device import DeviceManager
from quantedge_services.core.reproducibility import ReproducibilityManager
from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
from quantedge_services.services.wfs.cfpb_complaints.cfpb_service import CFPBComplaintsService
from quantedge_services.services.wfs.cfpb_complaints.tasks.dataset_task import CFPBDatasetTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.ingest_task import CFPBIngestionTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.preprocess_task import CFPBPreprocessTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.training_task import CFPBTrainingTask
from quantedge_services.services.wfs.forex_eurusd.forex_service import ForexEURUSDService
from quantedge_services.services.wfs.forex_eurusd.tasks.autograd_task import ForexAutogradTask
from quantedge_services.services.wfs.forex_eurusd.tasks.ingest_task import ForexIngestionTask
from quantedge_services.services.wfs.forex_eurusd.tasks.preprocess_task import ForexPreprocessTask
from quantedge_services.services.wfs.forex_eurusd.tasks.tensor_ops_task import ForexTensorOpsTask


class DependencyContainer:
    """Constructs and wires all application dependencies.

    Use get_container() to obtain the singleton instance.
    """

    def __init__(self) -> None:
        # Core
        self.device_manager = DeviceManager(prefer_gpu=True)
        self.reproducibility_manager = ReproducibilityManager(seed=42)
        self.checkpoint_manager = CheckpointManager(base_dir="data/checkpoints")

        # Forex tasks
        _forex_ingest = ForexIngestionTask()
        _forex_preprocess = ForexPreprocessTask(device_manager=self.device_manager)
        _forex_autograd = ForexAutogradTask(device_manager=self.device_manager)
        _forex_tensor_ops = ForexTensorOpsTask()

        # CFPB tasks
        _cfpb_ingest = CFPBIngestionTask()
        _cfpb_preprocess = CFPBPreprocessTask()
        _cfpb_dataset = CFPBDatasetTask()
        _cfpb_training = CFPBTrainingTask(
            device_manager=self.device_manager,
            reproducibility_manager=self.reproducibility_manager,
        )

        # Services
        self.forex_service = ForexEURUSDService(
            ingest_task=_forex_ingest,
            preprocess_task=_forex_preprocess,
            autograd_task=_forex_autograd,
            tensor_ops_task=_forex_tensor_ops,
        )
        self.cfpb_service = CFPBComplaintsService(
            ingest_task=_cfpb_ingest,
            preprocess_task=_cfpb_preprocess,
            dataset_task=_cfpb_dataset,
            training_task=_cfpb_training,
        )

        # Facade
        self.foundations_facade = FoundationsServiceFacade(
            forex_service=self.forex_service,
            cfpb_service=self.cfpb_service,
        )


@lru_cache(maxsize=1)
def get_container() -> DependencyContainer:
    """Return the application-scoped singleton DependencyContainer."""
    return DependencyContainer()
