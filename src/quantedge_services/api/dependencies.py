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
from quantedge_services.services.wfs.cmapss.cmapss_service import CMAPSSService
from quantedge_services.services.wfs.cmapss.tasks.download_task import CMAPSSDownloadTask
from quantedge_services.services.wfs.cmapss.tasks.ingest_task import CMAPSSIngestionTask
from quantedge_services.services.wfs.forex_attention.forex_attention_service import ForexAttentionService
from quantedge_services.services.wfs.forex_attention.tasks.attention_train_task import AttentionTrainTask
from quantedge_services.services.wfs.forex_attention.tasks.attention_eval_task import AttentionEvalTask
from quantedge_services.services.wfs.forex_attention.tasks.attention_predict_task import AttentionPredictTask
from quantedge_services.services.wfs.attention_viz.tasks.attention_extract_task import AttentionExtractTask
from quantedge_services.services.wfs.attention_viz.tasks.attention_heatmap_task import AttentionHeatmapTask
from quantedge_services.services.wfs.attention_viz.tasks.arch_decision_task import ArchDecisionTask
from quantedge_services.services.wfs.attention_viz.attention_viz_service import AttentionVizService
from quantedge_services.services.wfs.oasst1.tasks.download_task import OAsst1DownloadTask
from quantedge_services.services.wfs.oasst1.tasks.ingest_task import OAsst1IngestionTask
from quantedge_services.services.wfs.oasst1.oasst1_service import OAsst1Service
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_train_task import LoRATrainTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_eval_task import LoRAEvalTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_predict_task import LoRAPredictTask
from quantedge_services.services.wfs.lora_finetuning.tasks.lora_merge_task import LoRAMergeTask
from quantedge_services.services.wfs.lora_finetuning.lora_service import LoRAService


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

        # CMAPSS tasks + service
        _cmapss_download = CMAPSSDownloadTask()
        _cmapss_ingest = CMAPSSIngestionTask()
        self.cmapss_service = CMAPSSService(
            download_task=_cmapss_download,
            ingest_task=_cmapss_ingest,
        )

        # Attention tasks + service
        _attention_train = AttentionTrainTask()
        _attention_eval = AttentionEvalTask()
        _attention_predict = AttentionPredictTask()
        self.attention_service = ForexAttentionService(
            train_task=_attention_train,
            eval_task=_attention_eval,
            predict_task=_attention_predict,
        )

        # Attention viz tasks + service
        _attention_extract = AttentionExtractTask()
        _attention_heatmap = AttentionHeatmapTask()
        _arch_decision = ArchDecisionTask()
        self.attention_viz_service = AttentionVizService(
            extract_task=_attention_extract,
            heatmap_task=_attention_heatmap,
            arch_decision_task=_arch_decision,
        )

        # OAsst1 tasks + service
        _oasst1_download = OAsst1DownloadTask()
        _oasst1_ingest = OAsst1IngestionTask()
        self.oasst1_service = OAsst1Service(
            download_task=_oasst1_download,
            ingest_task=_oasst1_ingest,
        )

        # LoRA tasks + service
        _lora_train = LoRATrainTask()
        _lora_eval = LoRAEvalTask()
        _lora_predict = LoRAPredictTask()
        _lora_merge = LoRAMergeTask()
        self.lora_service = LoRAService(
            train_task=_lora_train,
            eval_task=_lora_eval,
            predict_task=_lora_predict,
            merge_task=_lora_merge,
        )

        # Facade
        self.foundations_facade = FoundationsServiceFacade(
            forex_service=self.forex_service,
            cfpb_service=self.cfpb_service,
            nn_service=self.nn_service,
            cic_iot_service=self.cic_iot_service,
            profiling_service=self.profiling_service,
            cmapss_service=self.cmapss_service,
            attention_service=self.attention_service,
            attention_viz_service=self.attention_viz_service,
            oasst1_service=self.oasst1_service,
            lora_service=self.lora_service,
        )


@lru_cache(maxsize=1)
def get_container() -> DependencyContainer:
    """Return the application-scoped singleton DependencyContainer."""
    return DependencyContainer()
