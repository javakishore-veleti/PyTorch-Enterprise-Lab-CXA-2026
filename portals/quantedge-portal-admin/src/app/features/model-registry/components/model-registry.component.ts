import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

/** Model Registry — bridges MLflow / SageMaker / AzureML backends (Week 11). */
@Component({
  selector: 'qe-admin-model-registry',
  standalone: true,
  imports: [CommonModule],
  template: `
    <h2>Model Registry</h2>
    <p>Backend: <code>MLflowRegistryClient</code> — configurable via <code>MODEL_REGISTRY_BACKEND</code> env var.</p>
    <ul>
      <li><code>mlflow</code> — local MLflow (default)</li>
      <li><code>sagemaker</code> — AWS SageMaker Model Registry</li>
      <li><code>azureml</code> — Azure ML Model Registry</li>
    </ul>
  `,
})
export class ModelRegistryComponent {}
