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
from quantedge_services.services.wfs.stackoverflow.tasks.download_task import StackOverflowDownloadTask
from quantedge_services.services.wfs.stackoverflow.tasks.ingest_task import StackOverflowIngestionTask
from quantedge_services.services.wfs.stackoverflow.stackoverflow_service import StackOverflowService
from quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_train_task import DomainAdaptTrainTask
from quantedge_services.services.wfs.domain_adaptation.tasks.domain_adapt_eval_task import DomainAdaptEvalTask
from quantedge_services.services.wfs.domain_adaptation.domain_adapt_service import DomainAdaptService
from quantedge_services.services.wfs.ollama_serving.tasks.ollama_infer_task import OllamaInferTask
from quantedge_services.services.wfs.ollama_serving.tasks.ollama_merge_task import OllamaMergeTask
from quantedge_services.services.wfs.ollama_serving.ollama_service import OllamaService
from quantedge_services.services.wfs.model_export.tasks.torchscript_export_task import TorchScriptExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_export_task import ONNXExportTask
from quantedge_services.services.wfs.model_export.tasks.onnx_validate_task import ONNXValidateTask
from quantedge_services.services.wfs.model_export.tasks.tensorrt_export_task import TensorRTExportTask
from quantedge_services.services.wfs.model_export.tasks.benchmark_task import BenchmarkTask
from quantedge_services.services.wfs.model_export.export_service import ModelExportService
from quantedge_services.services.wfs.quantization.tasks.static_quant_task import StaticQuantTask
from quantedge_services.services.wfs.quantization.tasks.dynamic_quant_task import DynamicQuantTask
from quantedge_services.services.wfs.quantization.tasks.qat_task import QATTask
from quantedge_services.services.wfs.quantization.tasks.quant_compare_task import QuantCompareTask
from quantedge_services.services.wfs.quantization.quantization_service import QuantizationService
from quantedge_services.services.wfs.model_serving.tasks.model_infer_task import ModelInferTask
from quantedge_services.services.wfs.model_serving.tasks.serving_benchmark_task import ServingBenchmarkTask
from quantedge_services.services.wfs.model_serving.serving_service import ModelServingService
from quantedge_services.services.wfs.experiment_tracking.tasks.mlflow_log_task import MLflowLogTask
from quantedge_services.services.wfs.experiment_tracking.tasks.mlflow_register_task import MLflowRegisterTask
from quantedge_services.services.wfs.experiment_tracking.tracking_service import ExperimentTrackingService
from quantedge_services.services.wfs.canary_deployment.tasks.canary_deploy_task import CanaryDeployTask
from quantedge_services.services.wfs.canary_deployment.tasks.canary_eval_task import CanaryEvalTask
from quantedge_services.services.wfs.canary_deployment.canary_service import CanaryService
from quantedge_services.services.wfs.model_registry.tasks.registry_list_task import ModelRegistryListTask
from quantedge_services.services.wfs.model_registry.tasks.registry_promote_task import ModelRegistryPromoteTask
from quantedge_services.services.wfs.model_registry.registry_service import ModelRegistryService


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

        # StackOverflow tasks + service
        _so_download = StackOverflowDownloadTask()
        _so_ingest = StackOverflowIngestionTask()
        self.stackoverflow_service = StackOverflowService(
            download_task=_so_download,
            ingest_task=_so_ingest,
        )

        # Domain Adaptation tasks + service
        _da_train = DomainAdaptTrainTask()
        _da_eval = DomainAdaptEvalTask()
        self.domain_adapt_service = DomainAdaptService(
            train_task=_da_train,
            eval_task=_da_eval,
        )

        # Ollama tasks + service
        _ollama_infer = OllamaInferTask()
        _ollama_merge = OllamaMergeTask()
        self.ollama_service = OllamaService(
            infer_task=_ollama_infer,
            merge_task=_ollama_merge,
        )

        # Model export tasks + service
        _ts_export = TorchScriptExportTask()
        _onnx_export = ONNXExportTask()
        _onnx_validate = ONNXValidateTask()
        _tensorrt_export = TensorRTExportTask()
        _benchmark = BenchmarkTask()
        self.export_service = ModelExportService(
            torchscript_task=_ts_export,
            onnx_export_task=_onnx_export,
            onnx_validate_task=_onnx_validate,
            tensorrt_task=_tensorrt_export,
            benchmark_task=_benchmark,
        )

        # Quantization tasks + service
        _static_quant = StaticQuantTask()
        _dynamic_quant = DynamicQuantTask()
        _qat = QATTask()
        _quant_compare = QuantCompareTask()
        self.quantization_service = QuantizationService(
            static_task=_static_quant,
            dynamic_task=_dynamic_quant,
            qat_task=_qat,
            compare_task=_quant_compare,
        )

        # Model serving tasks + service
        _model_infer = ModelInferTask()
        _serving_benchmark = ServingBenchmarkTask()
        self.serving_service = ModelServingService(
            infer_task=_model_infer,
            benchmark_task=_serving_benchmark,
        )

        # Experiment Tracking tasks + service
        _mlflow_log = MLflowLogTask()
        _mlflow_register = MLflowRegisterTask()
        self.tracking_service = ExperimentTrackingService(
            log_task=_mlflow_log,
            register_task=_mlflow_register,
        )

        # Canary Deployment tasks + service
        _canary_deploy = CanaryDeployTask()
        _canary_eval = CanaryEvalTask()
        self.canary_service = CanaryService(
            deploy_task=_canary_deploy,
            eval_task=_canary_eval,
        )

        # Model Registry tasks + service
        _registry_list = ModelRegistryListTask()
        _registry_promote = ModelRegistryPromoteTask()
        self.registry_service = ModelRegistryService(
            list_task=_registry_list,
            promote_task=_registry_promote,
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
            stackoverflow_service=self.stackoverflow_service,
            domain_adapt_service=self.domain_adapt_service,
            ollama_service=self.ollama_service,
            quantization_service=self.quantization_service,
            serving_service=self.serving_service,
            export_service=self.export_service,
            tracking_service=self.tracking_service,
            canary_service=self.canary_service,
            registry_service=self.registry_service,
        )


@lru_cache(maxsize=1)
def get_container() -> DependencyContainer:
    """Return the application-scoped singleton DependencyContainer."""
    return DependencyContainer()
