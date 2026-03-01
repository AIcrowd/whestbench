# SageMaker Sandboxed Evaluator Notes

Last updated: 2026-03-01

This document captures concrete AWS controls relevant to running participant submissions as isolated SageMaker jobs.

## Primary AWS Controls to Use

### 1. Network isolation

- SageMaker training jobs support `EnableNetworkIsolation`.
- IAM policies can require this via the condition key `sagemaker:NetworkIsolation`.

Sources:

- CreateTrainingJob API: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html
- SageMaker IAM condition keys/examples: https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_id-based-policy-examples.html
- Service authorization condition keys: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonsagemaker.html

### 2. VPC attachment and private data access

- Use job VPC config and private subnets.
- For network-isolated jobs that still need S3 artifacts, configure S3 VPC endpoints and (recommended) custom endpoint policies.
- Ensure private subnets have sufficient IPs for concurrent jobs.

Source:

- VPC setup for network-isolated training: https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html

### 3. Hard runtime and resource ceilings

- Training jobs support stop conditions such as `MaxRuntimeInSeconds` (and related wait-time controls).
- Processing jobs support `StoppingCondition` (`MaxRuntimeInSeconds`) as well.

Sources:

- CreateTrainingJob API: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html
- CreateProcessingJob API: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html

### 4. IAM policy guardrails for evaluator job creation

Use IAM condition keys to constrain what evaluator orchestrators can launch:

- `sagemaker:NetworkIsolation`
- `sagemaker:VpcSubnets`
- `sagemaker:VpcSecurityGroupIds`
- `sagemaker:InstanceTypes`
- `sagemaker:VolumeKmsKey`

Sources:

- Condition key list: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonsagemaker.html
- Policy examples: https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_id-based-policy-examples.html

### 5. Logging and artifact paths

For training-style containers:

- Logs are sent to CloudWatch (default log group `/aws/sagemaker/TrainingJobs`).
- Container paths/environment conventions include:
  - input channels via `SM_CHANNEL_*` and `/opt/ml/input/data/<channel>`
  - model output under `/opt/ml/model`
  - other outputs under `/opt/ml/output`

Sources:

- Toolkits overview and paths: https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-env-var-summary.html
- CloudWatch logs behavior: https://docs.aws.amazon.com/sagemaker/latest/dg/logging-cloudwatch.html

### 6. Throughput planning and quotas

Public quota docs indicate constraints such as:

- control-plane request rate limits (for example CreateTrainingJob requests per second),
- maximum job runtime limits.

Source:

- SageMaker endpoints/quotas: https://docs.aws.amazon.com/general/latest/gr/sagemaker.html

## Training Job vs Processing Job for Evaluator

Inference from AWS APIs/docs (not a platform mandate):

- Training jobs are well-supported and common for script-mode execution.
- Processing jobs are often a cleaner semantic fit for "run this evaluation workload and emit metrics/artifacts" when no model training lifecycle is needed.

For this challenge, either can work if isolation, VPC, IAM, and runtime controls are enforced consistently.

## Suggested Minimal Evaluator Architecture (Inference)

1. Submission intake service validates package shape and metadata.
2. Orchestrator writes immutable eval bundle to S3:
   - participant submission,
   - fixed evaluator code version,
   - fixed hidden test circuits.
3. Orchestrator launches SageMaker job with:
   - network isolation enabled,
   - restricted VPC/subnets/SG,
   - strict runtime cap,
   - allowlisted instance type,
   - dedicated IAM role with least privilege.
4. Job writes structured outputs to S3:
   - per-budget metrics,
   - runtime/resource stats,
   - failure reason codes.
5. Control plane ingests results and updates leaderboard asynchronously.

## Multi-Tenant Isolation Idea (Inference)

Use ABAC/session-tag patterns so each job role/session can be constrained to tenant-specific S3 prefixes and log access.

Source:

- ABAC with STS session tags: https://docs.aws.amazon.com/IAM/latest/UserGuide/tutorial_attribute-based-access-control.html

## Non-Negotiable Security Checklist (Proposed)

- Enforce no public internet egress from job containers.
- Enforce least-privilege IAM roles for job execution and orchestration.
- Restrict S3 access to evaluator input/output prefixes only.
- Use KMS keys for EBS/S3 encryption and enforce key usage in IAM policy.
- Set strict runtime and failure-handling policies (timeout, retries, dead-letter for failed orchestration events).

