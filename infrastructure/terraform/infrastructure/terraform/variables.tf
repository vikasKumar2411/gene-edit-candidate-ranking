variable "aws_region" {
  description = "AWS region for project resources."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"
}

variable "aws_account_id" {
  description = "AWS account ID."
  type        = string
  default     = "975050327570"
}

variable "project_bucket_name" {
  description = "Existing project S3 bucket."
  type        = string
  default     = "gene-edit-ranking-dev-975050327570-us-east-1-an"
}

variable "batch_image_uri" {
  description = "Immutable ECR URI for the batch-ranking image."
  type        = string

  default = "975050327570.dkr.ecr.us-east-1.amazonaws.com/gene-edit-ranking-batch@sha256:c4dcf6747a7ebf94d07bda46c00e89741f0ca0e2b6b004349a4153e438a15285"
}
