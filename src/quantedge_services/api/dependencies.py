"""DependencyContainer — wires all classes together for FastAPI DI."""
from __future__ import annotations
from functools import lru_cache
from quantedge_services.core.checkpointing import CheckpointManager
from quantedge_services.core.device import DeviceManager
from quantedge_services.core.jobs import JobRegistry
from quantedge_services.core.reproducibility import ReproducibilityManager
from quantedge_services.services.facade.foundations_facade import FoundationsServiceFacade
from quantedge_services.services.wfs.cfpb_complaints.cfpb_service import CFPBComplaintsService
from quantedge_services.services.wfs.cfpb_complaints.tasks.dataset_task import CFPBDatasetTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.download_task import CFPBDataDownloadTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.ingest_task import CFPBIngestionTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.preprocess_task import CFPBPreprocessTask
from quantedge_services.services.wfs.cfpb_complaints.tasks.training_task import CFPBTrainingTask
from quantedge_services.services.wfs.cic_iot.cic_iot_service import CICIoTService
from quantedge_services.services.wfs.cic_iot.tasks.download_task import CICIoTDownloadTask
from quantedge_services.services.wfs.cic_iot.tasks.ingest_task import CICIoTIngestionTask
from quantedge_services.services.wfs.forex_eurusd.forex_service import ForexEURUSDService
from quantedge_services.services.wfs.forex_eurusd.tasks.autograd_task import ForexAutogradTask
from quantedge_services.services.wfs.forex_eurusd.tasks.download_task import ForexDataDownloadTask
from quantedge_services.services.wfs.forex_eurusd.tasks.ingest_task import ForexIngestionTask
from quantedge_services.services.wfs.forex_eurusd.tasks.preprocess_task import ForexPreprocessTask
from quantedge_services.services.wfs.forex_eurusd.tasks.tensor_ops_task import ForexTensorOpsTask
from quantedge_services.services.wfs.forex_neuralnet.forex_nn_service import ForexNeuralNetService
from quantedge_services.services.wfs.forex_neuralnet.tasks.eval_task import NNEvalTask
from quantedge_services.services.wfs.forex_neuralnet.tasks.predict_task import NNPredictTask
from quantedge_services.services.wfs.forex_neuralnet.tasks.train_task import NNTrainTask
from quantedge_services.services.wfs.profiling.profiling_service import ProfilingService
from quantedge_services.services.wfs.profiling.tasks.dataloader_tune_task import DataloaderTuneTask
from quantedge_services.services.wfs.profiling.tasks.memory_summary_task import MemorySummaryTask
from quantedge_services.services.wfs.profiling.tasks.profiler_run_task import ProfilerRunTask


class DependencyContainer:
    """Constructs and wires all application dependencies.

    Use get_container() to obtain the singleton instance.
    """

    def __init__(self) -> None:
        # Core
        self.device_manager = DeviceManager(prefer_gpu=True)
        self.reproducibility_manager = ReproducibilityManager(seed=42)
        self.checkpoint_manager = CheckpointManager(base_dir="data/checkpoints")
        self.job_registry = JobRegistry()

        # Forex tasks
        _forex_download = ForexDataDownloadTask()
        _forex_ingest = ForexIngestionTask()
        _forex_preprocess = ForexPreprocessTask(device_manager=self.device_manager)
        _forex_autograd = ForexAutogradTask(device_manager=self.device_manager)
        _forex_tensor_ops = ForexTensorOpsTask()

        # CFPB tasks
        _cfpb_download = CFPBDataDownloadTask()
        _cfpb_ingest = CFPBIngestionTask()
        _cfpb_preprocess = CFPBPreprocessTask()
        _cfpb_dataset = CFPBDatasetTask()
        _cfpb_training = CFPBTrainingTask(
            device_manager=self.device_manager,
            reproducibility_manager=self.reproducibility_manager,
        )

        # Services
        self.forex_service = ForexEURUSDService(
            download_task=_forex_download,
            ingest_task=_forex_ingest,
            preprocess_task=_forex_preprocess,
            autograd_task=_forex_autograd,
            tensor_ops_task=_forex_tensor_ops,
        )
        self.cfpb_service = CFPBComplaintsService(
            download_task=_cfpb_download,
            ingest_task=_cfpb_ingest,
            preprocess_task=_cfpb_preprocess,
            dataset_task=_cfpb_dataset,
            training_task=_cfpb_training,
        )

        # NN tasks + service
        _nn_train = NNTrainTask()
        _nn_eval = NNEvalTask()
        _nn_predict = NNPredictTask()
        self.nn_service = ForexNeuralNetService(
            train_task=_nn_train,
            eval_task=_nn_eval,
            predict_task=_nn_predict,
        )

        # CIC IoT tasks + service
        _cic_iot_download = CICIoTDownloadTask()
        _cic_iot_ingest = CICIoTIngestionTask()
        self.cic_iot_service = CICIoTService(
            download_task=_cic_iot_download,
            ingest_task=_cic_iot_ingest,
        )

        # Profiling tasks + service
        _profiler_run = ProfilerRunTask()
        _memory_summary = MemorySummaryTask()
        _dataloader_tune = DataloaderTuneTask()
        self.profiling_service = ProfilingService(
            profiler_task=_profiler_run,
            memory_task=_memory_summary,
            dataloader_task=_dataloader_tune,
        )

        # Facade
        self.foundations_facade = FoundationsServiceFacade(
            forex_service=self.forex_service,
            cfpb_service=self.cfpb_service,
            nn_service=self.nn_service,
            cic_iot_service=self.cic_iot_service,
            profiling_service=self.profiling_service,
        )


@lru_cache(maxsize=1)
def get_container() -> DependencyContainer:
    """Return the application-scoped singleton DependencyContainer."""
    return DependencyContainer()
